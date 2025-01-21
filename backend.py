import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
from inverted_index_gcp import *

class BackendClass:
    def __init__(self):
        pass
        # self.name = 'backend'

    @staticmethod
    def get_index():
        base_dir = 'postings_gcp'
        name = 'index'
        bucket_name = 'hw3ir322'

        text_index = InvertedIndex.read_index(base_dir, name, bucket_name)
        # text_index = pickle.load(open(text_index_pkl, 'rb'))
        print(text_index.df)
# nltk.download('stopwords')

