from collections import Counter
import math
import numpy as np
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
import re
from inverted_index_gcp import *

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also”', 'links', 'extrenal',
                 'first', 'see', 'new', 'two', 'list', 'may', 'one', 'district',
                 'including', 'became', 'however', 'com', 'many', 'began',
                 'make', 'made', 'part', 'would', 'people', 'second', 'also',
                 'following', 'history', 'thumb', 'external']

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text):
    stemmer = PorterStemmer()
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    stemmed = [stemmer.stem(token) for token in tokens if token not in all_stopwords]
    return stemmed


def query_tfidf(query, index):
    """
    Calculates the TF-IDF for a given query.

    Args:
        query: A list of tokens representing the query.
        index: An inverted index object.

    Returns:
        A dictionary where keys are query tokens and values are their TF-IDF scores.
    """
    query_tfidf = {}
    N = len(index.df)  # Total number of documents
    for token in set(query):  # Iterate through unique query tokens
        tf = query.count(token)  # Term frequency in the query
        df = index.df.get(token, 0)  # Document frequency of the term
        if df == 0:
          idf = 0
        else:
          idf = math.log(N / df, 10)  # Inverse document frequency
        query_tfidf[token] = tf * idf

    return query_tfidf

def calc_cosine(query_tfidf, doc_tfidf):
    """
    Calculates the cosine similarity between a query and a document.
    """
    dot_product = 0
    query_magnitude = 0
    doc_magnitude = 0

    for term, tfidf in query_tfidf.items():
        dot_product += tfidf * doc_tfidf.get(term, 0)  # Handle terms not in the document
        query_magnitude += tfidf ** 2

    for tfidf in doc_tfidf.values():
        doc_magnitude += tfidf ** 2

    if query_magnitude == 0 or doc_magnitude == 0:
        return 0

    return dot_product / (math.sqrt(query_magnitude) * math.sqrt(doc_magnitude))


def cosine_similarity(query, candidates, index, doc_length_dict):
    """
    Returns a dictionary of candidates with cosine similarity scores.

    Output:
    - A dictionary where:
        - key: doc_id
        - value: cosine similarity score for the document
    """
    query_tfidf_scores = query_tfidf(query, index)
    N = len(index.df)
    # results = {}
    results = Counter()

    for term, posting_list in candidates.items():
        for doc_id, tf in posting_list:
            doc_tfidf = {}

            # Get document length from title_doc_lengths_dict
            doc_length = doc_length_dict.get(doc_id, 1)  # Default to 1 if not found

            # Calculate TF-IDF
            df = index.df.get(term, 0)  # Document frequency
            tf = tf  # Term frequency
            if df == 0:
                idf = 0
            else:
                idf = math.log(N / df, 10)  # Inverse document frequency

            tfidf_value = tf * idf  # TF-IDF value
            doc_tfidf[term] = tfidf_value

            cosine = calc_cosine(query_tfidf_scores, doc_tfidf)
            # results[doc_id] = cosine
            results[doc_id] += cosine

    return results


def BM25_score(tokenized_query, bucket_name, index, doc_num, doc_lengths, avg_doc_length, k1=1.2, b=0.75):
    """
    Calculates BM25 scores for documents based on a given query and an inverted index.

    Args:
        candidates (dict): A dictionary where keys are query terms and values are their posting lists.
        index: The inverted index object.
        doc_num (int): Total number of documents.
        doc_lengths (dict): A dictionary mapping document IDs to their lengths.
        avg_doc_length (float): The average document length.
        k1 (float, optional): BM25 tuning parameter. Defaults to 1.2.
        b (float, optional): BM25 tuning parameter. Defaults to 0.75.

    Returns:
        dict: A dictionary where keys are document IDs and values are their BM25 scores.
    """
    bm25_scores = Counter()  # Initialize a dictionary to store BM25 scores
    candidates_dict = {}  # Initialize a dictionary to store candidates of retrieval

    for token in tokenized_query:
        pl = index.read_a_posting_list(".", token, bucket_name)
        if pl == []:
            continue
        else:
            candidates_dict[token] = pl  # Store the posting list in the dictionary
            df = index.df[token]  # Document frequency
            idf = math.log(doc_num / df, 10)  # Inverse document frequency

            for doc_id, tf in candidates_dict[token]:  # Iterate through the posting list
                try:
                    norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_lengths[doc_id] / avg_doc_length)))
                    bm25_scores[doc_id] += idf * norm  # Accumulate scores
                except:
                    pass

    # bm25_scores = Counter()
    # for term in candidates.keys():
    #     # Calculate IDF
    #     df = index.df[term]
    #     idf = math.log(doc_num / df, 10)
    #
    #     # Calculate BM25 score for each document in the term's posting list
    #     for doc_id, tf in candidates[term]:
    #         try:
    #             norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_lengths[doc_id] / avg_doc_length)))
    #             # bm25_scores[doc_id] = bm25_scores.get(doc_id, 0) + idf * norm  # Accumulate scores
    #             bm25_scores[doc_id] += idf * norm  # for counter
    #         except:
    #             pass

    return bm25_scores


def word_count_score(candidates):
    """
    Calculates the number of query terms present in each candidate document.

    Args:
        candidates (dict): A dictionary where keys are query terms and values are
                           their corresponding posting lists (list of (doc_id, tf) tuples).

    Returns:
        dict: A dictionary where keys are document IDs and values are the number of
              query terms found in that document.
    """

    # doc_term_counts = {}  # Initialize a dictionary to store term counts for each document
    doc_term_counts = Counter()
    for term, postings in candidates.items():
        for doc_id, _ in postings:
            doc_term_counts[doc_id] = doc_term_counts.get(doc_id, 0) + 1  # Increment the term count for the document
    return doc_term_counts


def tf_count_score(candidates):
    """
    Calculates the total term frequency (tf) for each document in the candidates posting list.

    Args:
        candidates (dict): A dictionary where keys are query terms and values are
                           their corresponding posting lists (list of (doc_id, tf) tuples).

    Returns:
        dict: A dictionary where keys are document IDs and values are their total tf score.
    """
    # doc_tf_scores = {}  # Initialize a regular dictionary
    doc_tf_scores = Counter()
    for term, postings in candidates.items():
        for doc_id, tf in postings:
            doc_tf_scores[doc_id] = doc_tf_scores.get(doc_id, 0) + tf
    return doc_tf_scores


def get_candidates(tokenized_query, index, bucket_name):
    """
    Retrieves a dictionary mapping query tokens to their respective posting lists.

    Args:
    tokenized_query: A list of tokens representing the query.
    index: The inverted index object.

    Returns:
    A dictionary where keys are query tokens and values are their posting lists.
    """
    candidates_dict = {}  # Initialize the dictionary to store results

    for token in tokenized_query:
      pl = index.read_a_posting_list(".",token, bucket_name)
      if pl == []:
        continue
      else:
        candidates_dict[token] = pl
    return candidates_dict  # Return the dictionary


def merge(scores1, scores2, w1=0.5, w2=0.5):
    """
      Merges two dictionaries of scores from the same index type, applying weights to each.

      Args:
        scores1: A dictionary where keys are document IDs and values are scores.
        scores2: A dictionary where keys are document IDs and values are scores.
        w1: Weight for scores1 (default: 0.5).
        w2: Weight for scores2 (default: 0.5).

      Returns:
        A dictionary where keys are document IDs and values are weighted sums of
        corresponding scores from scores1 and scores2.
      """
    merged_scores = {}
    for doc_id in scores1:
        merged_scores[doc_id] = scores1[doc_id] * w1 + scores2[doc_id] * w2
    return merged_scores


def cross_merge(text_res, title_res, anchor_res, w1=0.65, w2=0.25, w3=0.1):
    """
      Merges three dictionaries of scores with different keys.

      Args:
        text_res: A dictionary where keys are document IDs and values are scores for text similarity functions.
        title_res: A dictionary where keys are document IDs and values are scores for title similarity functions.
        anchor_res: A dictionary where keys are document IDs and values are scores for anchor similarity functions.
        w1: Weight for text_res (default: 0.65).
        w2: Weight for title_res (default: 0.25).
        w3: Weight for anchor_res (default: 0.1).

      Returns:
        A dictionary where keys are document IDs and values are weighted sums of
        corresponding scores from text_res, title_res, and anchor_res.
        If a key is missing in one of the dictionaries, its score is considered 0.
      """
    return {
        doc_id: text_res.get(doc_id, 0.0) * w1 +
                title_res.get(doc_id, 0.0) * w2 +
                anchor_res.get(doc_id, 0.0) * w3
        for doc_id in set(text_res) | set(title_res) | set(anchor_res)
    }

