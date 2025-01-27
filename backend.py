import time
from collections import Counter
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
nltk.download('stopwords')


class BackendClass:
    def __init__(self):
        """
        Class to encapsulate and manage search indices and related data.

        Attributes:
            self.index_name : string of index name.
            self.text_index (InvertedIndex): Inverted index for document text content.
            self.title_index (InvertedIndex): Inverted index for document titles.
            self.anchor_index (InvertedIndex): Inverted index for anchor text.
            self.text_doc_len_dict (dict): Dictionary mapping document IDs to their lengths (in terms of number of words) for text content.
            self.title_doc_len_dict (dict): Dictionary mapping document IDs to their lengths (in terms of words) for titles.
            self.corpus_size (int): Total number of documents in the corpus.
            self.text_avg_doc_len (float): Average document length (in terms of words) for text content.
            self.title_avg_doc_len (float): Average document length (in terms of words) for titles.
            self.doc_id_title_even_dict (dict): Dictionary mapping even document IDs to their corresponding titles.
            self.doc_id_title_odd_dict (dict): Dictionary mapping odd document IDs to their corresponding titles.
            self.page_rank (dict): Dictionary mapping document IDs to their normalized PageRank scores.
            self.page_views (dict): Dictionary mapping document IDs to their normalized PageView counts.
        """
        # indices paths
        print("init backend class")
        self.index_name = 'index'
        self.text_idx_path = 'text_stemmed'
        self.title_idx_path = 'title_stemmed'
        self.anchor_idx_path = 'anchor_stemmed'

        # indexes paths for specific query functions
        self.og_anchor_idx_path = 'og_anchor_idx'
        self.og_text_idx_path = 'og_text_idx'
        self.og_title_idx_path = 'og_title_idx'

        # documents length dictionaries paths
        text_doc_len_path = 'text_stemmed/text_doc_lengths.pickle'
        title_doc_len_path = 'title_stemmed/title_doc_lengths.pickle'

        # indices data members
        self.text_index = InvertedIndex.read_index(self.text_idx_path, self.index_name)
        self.title_index = InvertedIndex.read_index(self.title_idx_path, self.index_name)
        self.anchor_index = InvertedIndex.read_index(self.anchor_idx_path, self.index_name)

        # Document length dict data members
        with open(text_doc_len_path, "rb") as file:
            self.text_doc_len_dict = pickle.load(file)


        with open(title_doc_len_path, "rb") as file:
            self.title_doc_len_dict = pickle.load(file)


        # corpus size and average doc length data members
        self.corpus_size = 6348910  # from the gcp ipynb notebook
        self.text_avg_doc_len = builtins.sum(self.text_doc_len_dict.values()) / self.corpus_size
        self.title_avg_doc_len = builtins.sum(self.title_doc_len_dict.values()) / self.corpus_size

        # doc_id - title dict data member
        doc_id_title_even_path = 'id_title/even_id_title_dict.pkl'
        doc_id_title_odd_path = 'id_title/uneven_id_title_dict.pkl'

        with open(doc_id_title_even_path, "rb") as file:
            self.doc_id_title_even_dict = pickle.load(file)

        with open(doc_id_title_odd_path, "rb") as file:
            self.doc_id_title_odd_dict = pickle.load(file)

        # PageRank data member
        pageRank_path = 'pr/part-00000-65f8552b-1b0d-4846-8d4e-74cf90eec0b7-c000.csv.gz'
        page_ranks = pd.read_csv(pageRank_path, compression='gzip', header=None, index_col=0).squeeze(
            "columns").to_dict()
        ranks_max = max(page_ranks.values()) # Normalize the page ranks
        self.page_rank = {id: rank / ranks_max for id, rank in page_ranks.items()}

        # PageView data member
        pageViews_path = 'pv/pageview.pkl' # a pickle to a dictionary
        with open(pageViews_path, "rb") as file:
            self.page_views = pickle.load(file)

        self.views_max = max(self.page_views.values())

    def search(self, query):
        # tokenize the query and create candidates dictionaries for each index
        tokenized_query = tokenize(query)

        # collect scores for query in text index using bm25
        bm25_scores_text = BM25_score(tokenized_query, self.text_index, self.corpus_size,
                                        self.text_doc_len_dict, self.text_avg_doc_len, k1=1.2, b=0.6)
        text_bm25_scores_top_500 = bm25_scores_text.most_common(500)

        # normalize text scores
        text_max_score = text_bm25_scores_top_500[0][1]

        # collect scores for query in title index using binary word count
        word_count_scores_title = BM25_score(tokenized_query, self.title_index, self.corpus_size, self.title_doc_len_dict,
                                             self.title_avg_doc_len, k1=1.5, b=0.4)
        title_word_count_scores_top_500 = word_count_scores_title.most_common(500)

        # normalize title scores
        title_max_score = title_word_count_scores_top_500[0][1]

        text_bm25_dict = {key: value/text_max_score  for key, value in text_bm25_scores_top_500}
        title_word_count_dict = {key :value/title_max_score for key, value in title_word_count_scores_top_500}


        # combine the 500 most common doc_ids from the three indices scores with the page rank and page views
        text_weight = 1.9
        title_weight = 1.1
        pr_weight = 0.4
        pv_weight = 0.6


        weighted_scores = [
            (doc_id,
             text_bm25_dict.get(doc_id, 0.0) * text_weight +
             title_word_count_dict.get(doc_id, 0.0) * title_weight +
             self.page_rank.get(doc_id, 0.0) * pr_weight +
             self.page_views.get(doc_id, 0.0) * pv_weight / self.views_max)
            for doc_id in set(text_bm25_dict) | set(title_word_count_dict)
        ]


        # sort the combined scores, transform to a list of top 100 doc_ids
        sorted_scores = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
        res = []
        for doc_id, _ in sorted_scores[:100]:
            if doc_id % 2 == 0:
                res.append((str(doc_id), self.doc_id_title_even_dict.get(doc_id)))
            else:
                res.append((str(doc_id), self.doc_id_title_odd_dict.get(doc_id)))
        return res

    def search_body(self, query):
        og_text_index = InvertedIndex.read_index(self.og_text_idx_path, self.index_name)
        tokens = og_tokenize(query)
        scored = cosine_similarity(tokens, og_text_index)
        top_100 = scored.most_common(100)
        res = []
        for doc_id, _ in top_100:
            if doc_id % 2 == 0:
                res.append((str(doc_id), self.doc_id_title_even_dict.get(doc_id)))
            else:
                res.append((str(doc_id), self.doc_id_title_odd_dict.get(doc_id)))
        return res

    def search_title(self, query):
        og_title_index = InvertedIndex.read_index(self.og_title_idx_path, self.index_name)
        tokens = og_tokenize(query)
        scored = word_count_score(tokens, og_title_index)
        sorted = scored.most_common()
        res = []
        for doc_id, _ in sorted:
            if doc_id % 2 == 0:
                res.append((str(doc_id), self.doc_id_title_even_dict.get(doc_id)))
            else:
                res.append((str(doc_id), self.doc_id_title_odd_dict.get(doc_id)))
        return res

    def search_anchor(self, query):
        og_anchor_index = InvertedIndex.read_index(self.og_anchor_idx_path, self.index_name)
        tokens = og_tokenize(query)
        scored = word_count_score(tokens, og_anchor_index)
        sorted = scored.most_common()
        res = []
        for doc_id, _ in sorted:
            if doc_id % 2 == 0:
                res.append((str(doc_id), self.doc_id_title_even_dict.get(doc_id)))
            else:
                res.append((str(doc_id), self.doc_id_title_odd_dict.get(doc_id)))
        return res

    def get_pagerank(self, page_ids):
        res = []
        for id in page_ids:
            res.append(self.page_rank.get(id, 0.0))
        return res


    def get_pageview(self, page_ids):
        res = []
        for id in page_ids:
            res.append(self.page_views.get(id, 0.0))
        return res


    def _get_doc_titles(self,id_list):
        res = []
        for id in id_list:
            if id % 2 == 0:
                res.append(self.doc_id_title_even_dict.get(id))
            else:
                res.append(self.doc_id_title_odd_dict.get(id))
        return res


    def search_prm(self, query, in_text_weight = 0.65 ,in_title_weight = 0.25,in_anchor_weight = 0.1 ,in_pr_weight = 1 ,in_pv_weight = 1,k=1.2,b=0.5):
        # tokenize the query and create candidates dictionaries for each index
        tokenized_query = tokenize(query)

        # collect scores for query in text index using bm25
        bm25_scores_text = BM25_score(tokenized_query, self.text_index, self.corpus_size,
                                        self.text_doc_len_dict, self.text_avg_doc_len, k1=k, b=b)
        text_bm25_scores_top_500 = bm25_scores_text.most_common(500)

        # normalize text scores
        text_max_score = text_bm25_scores_top_500[0][1]
        text_bm25_scores_top_500 = [(pair[0], pair[1]/text_max_score) for pair in text_bm25_scores_top_500]

        # collect scores for query in title index using binary word count
        word_count_scores_title = word_count_score(tokenized_query, self.title_index)
        title_word_count_scores_top_500 = word_count_scores_title.most_common(500)

        # normalize title scores
        title_max_score = title_word_count_scores_top_500[0][1]
        title_word_count_scores_top_500 = [(pair[0], pair[1]/title_max_score) for pair in title_word_count_scores_top_500]

        # collect and merge scores for query in anchor index using word count
        word_count_scores_anchor = word_count_score(tokenized_query, self.anchor_index)
        anchor_word_count_scores_top_500 = word_count_scores_anchor.most_common(500)

        # normalize anchor scores
        anchor_max_score = anchor_word_count_scores_top_500[0][1]
        anchor_word_count_scores_top_500 = [(pair[0], pair[1]/anchor_max_score) for pair in anchor_word_count_scores_top_500]

        # Create a dict for quick lookup
        text_bm25_dict = dict(text_bm25_scores_top_500)
        title_word_count_dict = dict(title_word_count_scores_top_500)
        anchor_word_count_dict = dict(anchor_word_count_scores_top_500)

        # combine the 500 most common doc_ids from the three indices scores with the page rank and page views
        text_weight = in_text_weight
        title_weight = in_title_weight
        anchor_weight = in_anchor_weight
        pr_weight = in_pr_weight
        pv_weight = in_pv_weight
        weighted_scores = [
            (doc_id,
             text_bm25_dict.get(doc_id, 0.0) * text_weight +
             title_word_count_dict.get(doc_id, 0.0) * title_weight +
             anchor_word_count_dict.get(doc_id, 0.0) * anchor_weight +
             self.page_rank.get(doc_id, 0.0) * pr_weight +
             self.page_views.get(doc_id, 0.0) * pv_weight / self.views_max)
            for doc_id in set(text_bm25_dict) | set(title_word_count_dict) | set(anchor_word_count_dict)
        ]

        # sort the combined scores, transform to a list of top 100 doc_ids
        sorted_scores = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
        return [(str(doc_id),"res") for doc_id, score in sorted_scores[:100]]