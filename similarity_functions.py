from collections import Counter
import math
import numpy as np
import pandas as pd
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
import re
from inverted_index_gcp import *

nltk.download('stopwords')
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


def og_tokenize(query):
    english_sw = frozenset(stopwords.words('english'))
    corpus_sw = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_sw = english_sw.union(corpus_sw)
    RE_W = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_W.finditer(query.lower())]
    return [token for token in tokens if token not in all_sw]

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


def cosine_similarity(tokenized_query, index):
    """
    Returns a dictionary of candidates with cosine similarity scores.

    Args:
        tokenized_query (list): A list of tokens representing the query.
        index: The inverted index object.

    Output:
    - A dictionary where:
        - key: doc_id
        - value: cosine similarity score for the document
    """
    query_tfidf_scores = query_tfidf(tokenized_query, index)
    N = len(index.df)
    candidates_dict = {}
    results = Counter()

    for token in tokenized_query:
        pl = index.read_a_posting_list(".", token)
        if pl == []:
            continue
        else:
            candidates_dict[token] = pl
            for doc_id, tf in candidates_dict[token]:
                doc_tfidf = {}
                df = index.df.get(token, 0)
                if df == 0:
                    idf = 0
                else:
                    idf = math.log(N / df, 10)
                tfidf_value = tf * idf
                doc_tfidf[token] = tfidf_value
                cosine = calc_cosine(query_tfidf_scores, doc_tfidf)
                results[doc_id] += cosine

    return results

def BM25_score(tokenized_query, index, doc_num, doc_lengths, avg_doc_length, k1=1.2, b=0.75):
    """
    Calculates BM25 scores for documents based on a given query and an inverted index.

    Args:
        tokenized_query (list): A list of tokens representing the query.
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
        pl = index.read_a_posting_list(".", token)
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

    return bm25_scores


def word_count_score(tokenized_query, index):
    """
    Calculates the number of query terms present in each candidate document.

    Args:
        tokenized_query (list): A list of tokens representing the query.
        index: The inverted index object.

    Returns:
        dict: A dictionary where keys are document IDs and values are the number of
              query terms found in that document.
    """

    candidates_dict = {}  # Initialize a regular dictionary
    doc_term_counts = Counter()
    for token in tokenized_query:
        pl = index.read_a_posting_list(".", token)
        if pl == []:
            continue
        else:
            candidates_dict[token] = pl
            for doc_id, tf in candidates_dict[token]:
                doc_term_counts[doc_id] += 1
    return doc_term_counts


def tf_count_score(tokenized_query, index):
    """
    Calculates the total term frequency (tf) for each document in the candidates posting list.

    Args:
        tokenized_query (list): A list of tokens representing the query.
        index: The inverted index object.

    Returns:
        dict: A dictionary where keys are document IDs and values are their total tf score.
    """
    candidates_dict = {}  # Initialize a regular dictionary
    doc_tf_scores = Counter()

    for token in tokenized_query:
        pl = index.read_a_posting_list(".", token)
        if pl == []:
            continue
        else:
            candidates_dict[token] = pl
            for doc_id, tf in candidates_dict[token]:
                doc_tf_scores[doc_id] += tf

    return doc_tf_scores



