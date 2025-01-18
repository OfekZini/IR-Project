import math
import numpy as np
import pandas as pd


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

def cosine_similarity(query_tfidf, doc_tfidf):
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


def get_candidates_with_cosine(query, candidates, index, doc_length_dict):
    """
    Returns a dictionary of candidates with cosine similarity scores.

    Output:
    - A dictionary where:
        - key: doc_id
        - value: cosine similarity score for the document
    """
    query_tfidf_scores = query_tfidf(query, index)
    N = len(index.df)
    results = {}

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

            cosine = cosine_similarity(query_tfidf_scores, doc_tfidf)
            results[doc_id] = cosine

    return results

def BM25_score(candidates, index, doc_num, doc_lengths, avg_doc_length, k1=1.2, b=0.75):
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
    bm25_scores = {}  # Initialize an empty dictionary to store scores

    for term in candidates.keys():
        # Calculate IDF
        df = index.df[term]
        idf = math.log(doc_num / df, 10)

        # Calculate BM25 score for each document in the term's posting list
        for doc_id, tf in candidates[term]:
            try:
                norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_lengths[doc_id] / avg_doc_length)))
                bm25_scores[doc_id] = bm25_scores.get(doc_id, 0) + idf * norm  # Accumulate scores
            except:
                pass

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

    doc_term_counts = {}  # Initialize a dictionary to store term counts for each document
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
    doc_tf_scores = {}  # Initialize a regular dictionary
    for term, postings in candidates.items():
        for doc_id, tf in postings:
            doc_tf_scores[doc_id] = doc_tf_scores.get(doc_id, 0) + tf
    return doc_tf_scores