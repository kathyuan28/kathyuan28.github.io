import base64
import io
import json
import math
import os
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse

import boto3
import fastText
import numpy as np
import pandas as pd
from Crypto.Cipher import AES

from sage_maker_evaluation import segment_review_data


class GloveModel:
    """ Class that holds Glove embedding model
    """

    def __init__(self, model_path):
        """
        :param model_path: path of glove model artifact file
        """
        words, vectors = parse_glove_model(model_path)

        vocab_size = len(words)
        vocab = {w: idx for idx, w in enumerate(words)}
        ivocab = {idx: w for idx, w in enumerate(words)}

        vector_dim = len(vectors[ivocab[0]])
        W = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.items():
            W[vocab[word], :] = v

        # normalize each word vector to unit length
        W_norm = np.zeros(W.shape)
        d = np.sum(W ** 2, 1) ** (0.5)
        W_norm = (W.T / d).T

        self._vocab = vocab
        self._ivocab = ivocab
        self._W = W_norm
        self._vector_dim = vector_dim

    def get_dimension(self):
        """ Return Glove model's vector dimension. Compatible with fastText embedding
        model's get_dimension method.

        :return: word embedding vector dimension
        """
        return self._vector_dim

    def get_sentence_vector(self, sentence):
        """ Get sentence embedding (by averaging word embedding values). Compatible with
        fastText embedding model's get_sentence_vector method.

        :param sentence: sentence (in string type) to be calculated
        :return: sentence embedding vector
        """
        word_list = sentence.split()

        indices = np.array(
            [self._vocab[word] if word in self._vocab else self._vocab["<unk>"] for word in word_list]
        )
        sentence_embedding = np.sum(self._W[indices, :], axis=0) / indices.shape[0]

        return sentence_embedding


class ReviewEmbeddingBuilder:
    """ Training data (sentence embeddings) builder. Supports multiple word embedding algorithms.
    """

    REVIEW_RATING_THRESHOLD = 3

    def __init__(self, needs_segment=True):
        """
        :param needs_segment: indicate if we want to segment the customer review data.
        """
        self._s3 = boto3.resource("s3")
        self._client = boto3.client("s3")
        self._needs_segment = needs_segment
        if needs_segment:
            self._review_column = "review_text"
        else:
            self._review_column = "review_title_text_segmented"

    def load_model(self, model_path):
        """Load the word embedding model.

        :param model_path: path of word embedding model zip file (could be a s3 uri or local path)
        """
        dirpath = tempfile.mkdtemp()
        tar_path = os.path.join(dirpath, "model.tar.gz")

        if not model_path.startswith("s3"):
            # model artifact is locally present
            shutil.copyfile(model_path, tar_path)
        else:
            # model artifact is stored in s3
            bucket, key = split_s3_uri(model_path)
            self._client.download_file(bucket, key, tar_path)

        tar = tarfile.open(tar_path)
        tar.extractall(path=dirpath)
        tar.close()

        if os.path.isfile(os.path.join(dirpath, "vectors.bin")):
            # blazingtext model
            model_path = os.path.join(dirpath, "vectors.bin")
            self._embedding_model = fastText.load_model(model_path)
        else:
            # glove model
            model_path = os.path.join(dirpath, "vectors.txt")
            self._embedding_model = GloveModel(model_path)

        # local file cleanup
        shutil.rmtree(dirpath)

    def read_all_partitions(
        self, bucket, key_prefix, s3uri, kms_keyid, marketplace_IDs, downsample_rate=0.0005
    ):
        """Read all customer review partitions under a given s3 bucket and key prefix.

        :param bucket: s3 bucket uri containing customer review partitions
        :param key_prefix: s3 key prefix of customer review partitions
        :param s3uri: uri to store the (temporarily) converted SSE_KMS data
        :param kms_keyid: user's kms keyid
        :param marketplace_IDs: list of marketplace IDs to be kept in the dataframe
        :param downsample_rate: indicates how much we want to downsample customer review data. For example,
        if this variable is set to 0.01, then we downsample the review data to 1%.
        :return: dataframe containing filtered customer review data
        """
        df = None
        print("Getting pandas dataframe of raw training data.")

        s3_bucket = self._s3.Bucket(bucket)
        for partition in s3_bucket.objects.filter(Prefix=key_prefix):
            sub_df = read_and_filter_partition(
                bucket,
                partition.key,
                s3uri,
                kms_keyid,
                self._review_column,
                downsample_rate,
                marketplace_IDs,
            )
            if df is None:
                df = sub_df
            else:
                df = df.append(sub_df)

        return df

    def preprocess_review_and_rating(self, df, s3_uri, needs_stratified_sampling=True):
        """Preprocess raw review and rating data. This includes:

        (1): Converting rating data to 0/1 label
        (2): Creating space-segmented customer review data, if needed
        (3): Uploading segmented review and preprocessed label to s3

        :param df: dataframe that contains review data
        :param s3_uri: s3 path to store segmented review and preprocessed label
        :param needs_stratified_sampling: indicates whether we need to perform stratified sampling.
        :return: s3 path for segmented review data and label
        """
        df[self._review_column] = df[self._review_column].astype(str)
        df = convert_rating_to_binary_label(df, ReviewEmbeddingBuilder.REVIEW_RATING_THRESHOLD)
        if needs_stratified_sampling:
            n = df["label"].value_counts().min()
            df = df.groupby("label").apply(lambda x: x.sample(n))

        review_idx = df.columns.get_loc(self._review_column)
        label_idx = df.columns.get_loc("label")

        # raw_reviews and ratings are ndarray
        raw_reviews = df.iloc[:, review_idx].values
        labels = df.iloc[:, label_idx].values
        labels = labels.astype(int)

        if self._needs_segment:
            segmenter = segment_review_data.SpacyEnglishSegmenter()
            segmented_reviews = [" ".join(segmenter.segment(review)) for review in raw_reviews]
        else:
            segmented_reviews = raw_reviews
        
        segmented_reviews = np.array(segmented_reviews)

        # upload to s3
        s3_review_uri = s3_uri + "/" + "data.npy"
        s3_label_uri = s3_uri + "/" + "label.npy"
        save_numpy_and_upload_to_s3(segmented_reviews, s3_review_uri)
        save_numpy_and_upload_to_s3(labels, s3_label_uri)

        return s3_review_uri, s3_label_uri
        

    def download_and_calculate_sentence_embedding(self, s3_review_uri, s3_label_uri):
        """ Download segmented review and preprocessed label from s3, and calculate sentence embedding for each review

        :param s3_review_uri: s3 uri of segmented review data
        :param s3_label_uri: s3 uri of preprocessed label data
        :return: sentence embedding and label, both in ndarray
        """
        dirpath = tempfile.mkdtemp()

        review_path = os.path.join(dirpath, "data.npy")
        label_path = os.path.join(dirpath, "label.npy")

        download_file_from_s3(review_path, s3_review_uri)
        download_file_from_s3(label_path, s3_label_uri)

        reviews = np.load(review_path)
        labels = np.load(label_path)

        print("Calculating sentence embeddings.")

        reviews = [self.sentence_embedding(review) for review in reviews]
        reviews = np.array(reviews)

        print("Finished calculation.")

        # remove zero vectors (which only result from meaningless comments that only contain stopwords)
        nonzero_idx = np.where(reviews.any(axis=1))[0]
        reviews = reviews[nonzero_idx]
        labels = labels[nonzero_idx]

        shutil.rmtree(dirpath)

        return reviews, labels

    def sentence_embedding(self, sentence):
        """Get sentence embedding (by averaging word embedding values)

        :param sentence: sentence (in string type) to be calculated
        :return: sentence embedding vector
        """
        if len(sentence) == 0:
            return np.zeros(self._embedding_model.get_dimension())

        return self._embedding_model.get_sentence_vector(sentence)


""" utility functions """


def split_s3_uri(uri):
    """Splits s3 uri into bucket and key

    :param uri: the uri to split
    :return: bucket and key of that uri
    """
    o = urlparse(uri)
    return o.netloc, o.path.strip("/")


def csekms_to_ssekms(s3_csekms_data_path, s3_ssekms_data_path, kms_keyid):
    """Convert CSE_KMS data to SSE_KMS data.

    :param s3_csekms_data_path: uri of CSE_KMS data
    :param s3_ssekms_data_path: uri to store the (temporarily) converted SSE_KMS data
    :param kms_keyid: user's kms keyid
    """
    bucket, key = split_s3_uri(s3_csekms_data_path)
    s3 = boto3.resource("s3")
    s3_response = s3.Object(bucket, key).get()

    encrypted_data = s3_response["Body"].read()
    metadata = s3_response["Metadata"]
    if "x-amz-key-v2" in metadata:
        encrypted_data_key = base64.b64decode(metadata["x-amz-key-v2"])
        iv = base64.b64decode(metadata["x-amz-iv"])
        encryption_context = json.loads(metadata["x-amz-matdesc"])
        kms = boto3.client("kms")
        out = kms.decrypt(CiphertextBlob=encrypted_data_key, EncryptionContext=encryption_context)
        decrypted_data_key = out["Plaintext"]
        decryptor = AES.new(decrypted_data_key, AES.MODE_CBC, IV=iv)
        decrypted_data = decryptor.decrypt(encrypted_data)
    else:
        decrypted_data = encrypted_data

    bucket, key = split_s3_uri(s3_ssekms_data_path)
    s3.Object(bucket, key).put(
        Body=decrypted_data,
        ContentEncoding="utf-8",
        ContentType="text/plain",
        ServerSideEncryption="aws:kms",
        SSEKMSKeyId=kms_keyid,
    )


def save_numpy_and_upload_to_s3(data, s3_uri):
    """ Save np array as .npy file and upload it to s3.

    :param data: data in ndarray
    :param s3_uri: full s3 uri to store the .npy file
    """
    dirpath = tempfile.mkdtemp()
    data_file = os.path.join(dirpath, "tmp.npy")

    np.save(data_file, data)
    bucket, key = split_s3_uri(s3_uri)
    boto3.client("s3").upload_file(data_file, bucket, key)

    shutil.rmtree(dirpath)


def download_file_from_s3(file_name, s3_uri):
    """ Simple wrapper of boto3 s3 client download function

    :param file_name: local path to store the downloaded file
    :param s3_uri: s3 uri of the file to be downloaded
    """
    bucket, key = split_s3_uri(s3_uri)
    boto3.client("s3").download_file(bucket, key, file_name)


def read_and_filter_partition(
    bucket, key, s3uri, kms_keyid, review_column, downsample_rate, marketplace_IDs
):
    """Read customer review data into pandas df

    :param bucket: s3 bucket uri of CSE_KMS data
    :param key: s3 key uri of CSE_KMS data
    :param s3uri: uri to store the (temporarily) converted SSE_KMS data
    :param kms_keyid: user's kms keyid
    :param review_column: df column that contains review data
    :param downsample_rate: downsample rate of the review data
    :param marketplace_IDs: list of marketplace IDs to be kept in the dataframe
    :return: pandas dataframe that contains filtered review data
    """
    s3euri = "s3://" + bucket + "/" + key
    print("Reading from:", s3euri)
    df = read_customer_review_dataframe(s3euri, s3uri, kms_keyid)
    df = df[(df[review_column] != "nan") & (df[review_column] != "")]
    df = df[df["overall_rating"].astype(str).str.isdigit()]
    df = df[df["marketplace_id"].astype(int).isin(marketplace_IDs)]
    df = df.sample(n=math.ceil(downsample_rate * df.shape[0]))
    df = df[[review_column, "overall_rating"]]

    return df


def read_customer_review_dataframe(s3euri, s3uri, kms_keyid):
    """Get the pandas df from CSE_KMS encrypted customer review data

    :param s3euri: uri of CSE_KMS data
    :param s3uri: uri to store the (temporarily) converted SSE_KMS data
    :param kms_keyid: user's kms keyid
    :return: pandas dataframe that contains review data
    """
    csekms_to_ssekms(s3euri, s3uri, kms_keyid)

    bucket, key = split_s3_uri(s3uri)
    s3 = boto3.resource("s3")
    s3_response = s3.Object(bucket, key).get()
    data = s3_response["Body"].read()

    df = pd.read_csv(io.BytesIO(data), encoding="utf8", sep="\t", low_memory=False)
    return df


def convert_rating_to_binary_label(df, threshold):
    """Given a threshold, convert 1-5 rating to 0/1 binary label

    :param df: customer review dataframe
    :param threshold: threshold that decides how ratings are converted to 0/1.
    For example, if threshold = 3, then all reviews that have < 3 rating will have label 0.
    :return: dataframe with rating column renamed to label, which contains 0/1 values.
    """
    df["overall_rating"] = np.where(df["overall_rating"] < threshold, 0, df["overall_rating"])
    df["overall_rating"] = np.where(df["overall_rating"] >= threshold, 1, df["overall_rating"])
    df = df.rename({"overall_rating": "label"}, axis=1)

    return df


def parse_glove_model(vector_file="vectors.txt"):
    """Parse glove model file and read it into memory

    :param vector_file: local word embedding model file
    :return: two lists. One contains all the words. The other one contains all corresponding vectors.
    """
    with open(vector_file, "r") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(" ")
            vectors[vals[0]] = [float(x) for x in vals[1:]]

        f.seek(0)
        words = [x.rstrip().split(" ")[0] for x in f.readlines()]

        return words, vectors
