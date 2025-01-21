import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
import builtins
from inverted_index_gcp import *

# bucket name
bucket_name = 'hw3ir322'

def get_bucket(bucket_name):
    """Retrieve a Google Cloud Storage bucket."""
    client = storage.Client()
    return client.bucket(bucket_name)


def download_blob_as_bytes(bucket, path):
    """Download a blob from the specified bucket as bytes."""
    blob = bucket.blob(path)
    return blob.download_as_bytes()

bucket = get_bucket(bucket_name)


class BackendClass:
    def __init__(self):
        # indices paths
        index_name = 'index'
        text_idx_path = 'text_stemmed'
        title_idx_path = 'title_stemmed'
        anchor_idx_path = 'anchor_stemmed'

        # documents length dictionaries paths
        text_doc_len_path = 'text_stemmed/text_doc_lengths.pickle'
        title_doc_len_path = 'title_stemmed/title_doc_lengths.pickle'
        anchor_doc_len_path = 'anchor_stemmed/anchor_doc_lengths.pickle'

        # indices data members
        self.text_index = InvertedIndex.read_index(text_idx_path, index_name, bucket_name)
        self.title_index = InvertedIndex.read_index(title_idx_path, index_name, bucket_name)
        self.anchor_index = InvertedIndex.read_index(anchor_idx_path, index_name, bucket_name)

        # Document length dict data members
        text_doc_len_bytes = download_blob_as_bytes(bucket, text_doc_len_path)
        self.text_doc_len_dict = pickle.loads(text_doc_len_bytes)
        title_doc_len_bytes = download_blob_as_bytes(bucket, title_doc_len_path)
        self.title_doc_len_dict = pickle.loads(title_doc_len_bytes)
        anchor_doc_len_bytes = download_blob_as_bytes(bucket, anchor_doc_len_path)
        self.anchor_doc_len_dict = pickle.loads(anchor_doc_len_bytes)

        # corpus size and average doc length data members
        self.corpus_size = 6348910  # from the gcp ipynb notebook
        self.avg_doc_len = builtins.sum(self.text_doc_len_dict.values()) / self.corpus_size

        # doc_id - title dict data member
        # self.docID_title_dict =

        # PageRank PageView data members
        # self.page_rank =
        # self.page_view =

