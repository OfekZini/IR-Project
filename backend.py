import gzip
import re
import math
from io import BytesIO
import numpy as np
import pandas as pd
from collections import Counter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
import builtins
from inverted_index_gcp import *
from similarity_functions import *

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
        """
        Class to encapsulate and manage search indices and related data.

        Attributes:
            self.text_index (InvertedIndex): Inverted index for document text content.
            self.title_index (InvertedIndex): Inverted index for document titles.
            self.anchor_index (InvertedIndex): Inverted index for anchor text.
            self.text_doc_len_dict (dict): Dictionary mapping document IDs to their lengths (in terms of number of words) for text content.
            self.title_doc_len_dict (dict): Dictionary mapping document IDs to their lengths (in terms of words) for titles.
            self.anchor_doc_len_dict (dict): Dictionary mapping document IDs to their lengths (in terms of words) for anchor text.
            self.corpus_size (int): Total number of documents in the corpus.
            self.avg_doc_len (float): Average document length (in terms of words) for text content.
            self.doc_id_title_even_dict (dict): Dictionary mapping even document IDs to their corresponding titles.
            self.doc_id_title_odd_dict (dict): Dictionary mapping odd document IDs to their corresponding titles.
            self.page_rank (dict): Dictionary mapping document IDs to their normalized PageRank scores.
            self.page_views (dict): Dictionary mapping document IDs to their normalized PageView counts.
        """
        # indices paths
        index_name = 'index'
        text_idx_path = 'text_stemmed'
        title_idx_path = 'title_stemmed'
        anchor_idx_path = 'anchor_stemmed'

        # documents length dictionaries paths
        text_doc_len_path = 'text_stemmed/text_doc_lengths.pickle'
        title_doc_len_path = 'title_stemmed/title_doc_lengths.pickle'
        # anchor_doc_len_path = 'anchor_stemmed/anchor_doc_lengths.pickle'

        # indices data members
        self.text_index = InvertedIndex.read_index(text_idx_path, index_name, bucket_name)
        self.title_index = InvertedIndex.read_index(title_idx_path, index_name, bucket_name)
        self.anchor_index = InvertedIndex.read_index(anchor_idx_path, index_name, bucket_name)

        # Document length dict data members
        text_doc_len_bytes = download_blob_as_bytes(bucket, text_doc_len_path)
        self.text_doc_len_dict = pickle.loads(text_doc_len_bytes)
        title_doc_len_bytes = download_blob_as_bytes(bucket, title_doc_len_path)
        self.title_doc_len_dict = pickle.loads(title_doc_len_bytes)
        # anchor_doc_len_bytes = download_blob_as_bytes(bucket, anchor_doc_len_path)
        # self.anchor_doc_len_dict = pickle.loads(anchor_doc_len_bytes)

        # corpus size and average doc length data members
        self.corpus_size = 6348910  # from the gcp ipynb notebook
        self.avg_doc_len = builtins.sum(self.text_doc_len_dict.values()) / self.corpus_size

        # doc_id - title dict data member
        doc_id_title_even_path = 'id_title/even_id_title_dict.pkl'
        doc_id_title_odd_path = 'id_title/uneven_id_title_dict.pkl'
        doc_id_title_even_bytes = download_blob_as_bytes(bucket, doc_id_title_even_path)
        doc_id_title_odd_bytes = download_blob_as_bytes(bucket, doc_id_title_odd_path)
        self.doc_id_title_even_dict = pickle.loads(doc_id_title_even_bytes)
        self.doc_id_title_odd_dict = pickle.loads(doc_id_title_odd_bytes)

        # PageRank data member
        pageRank_path = 'pr/part-00000-65f8552b-1b0d-4846-8d4e-74cf90eec0b7-c000.csv.gz' # a pyspark csv
        pageRank_bytes = download_blob_as_bytes(bucket, pageRank_path)
        with gzip.GzipFile(fileobj=BytesIO(pageRank_bytes)) as f:
            page_ranks = pd.read_csv(f, header=None, index_col=0).squeeze("columns").to_dict()
        ranks_max = max(page_ranks.values())
        self.page_rank = {id: rank / ranks_max for id, rank in page_ranks.items()}

        # PageView data member
        pageViews_path = 'pv/pageview.pkl' # a pickle to a dictionary
        pageViews_bytes = download_blob_as_bytes(bucket, pageViews_path)
        page_view = pickle.loads(pageViews_bytes)
        views_max = max(page_view.values())
        self.page_views = {id: view / views_max for id, view in page_view.items()}

    def search(self, query):
        # tokenize the query and create candidates dictionaries for each index
        tokenized_query = tokenize(query)
        text_candidates = get_candidates(tokenized_query, self.text_index)
        title_candidates = get_candidates(tokenized_query, self.title_index)
        anchor_candidates = get_candidates(tokenized_query, self.anchor_index)

        # collect and merge scores for query in text index
        bm25_scores_text = BM25_score(text_candidates, self.text_index, self.corpus_size, self.text_doc_len_dict,
                                 self.avg_doc_len, k1=1.2, b=0.75)
        cos_sim_text = cosine_similarity(tokenized_query, text_candidates, self.text_index, self.text_doc_len_dict)
        merged_text = merge(bm25_scores_text, cos_sim_text)

        # collect and merge scores for query in title index
        word_count_scores_title = word_count_score(title_candidates)
        cos_sim_title = cosine_similarity(tokenized_query, title_candidates, self.title_index, self.title_doc_len_dict)
        merged_title = merge(word_count_scores_title, cos_sim_title)

        # collect and merge scores for query in anchor index
        word_count_scores_anchor = word_count_score(anchor_candidates)
        tf_count_scores_anchor = tf_count_score(anchor_candidates)
        merged_anchor = merge(word_count_scores_anchor, tf_count_scores_anchor)

        # combine the three indices scores
        combined_scores = cross_merge(merged_text, merged_title, merged_anchor)

        # sort the combined scores and return the top 100
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:100]
