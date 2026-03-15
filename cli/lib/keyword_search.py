import os
import pickle
import string
from typing import Counter
from lib import search_utils
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
from lib.search_utils import BM25_K1

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, text):
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in remove_stopwords(tokenize(text))]
        for token in tokens:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

        # add to term_frequencies counter
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term):
        term = term.lower()
        ids = self.index.get(term, set())
        return sorted(ids)
    
    def build(self):
        movies = search_utils.load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self):
        # Check if cache dir exist -> create one
        if not os.path.exists('cache'):
            os.makedirs('cache')

        # Save index data (wb - write mode)
        with open('cache/index.pkl', 'wb') as f:
            pickle.dump(self.index, f)

        # Save docmap data (wb - write mode)
        with open('cache/docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)

        # Save term_frequencies data
        with open('cache/term_frequencies.pkl', 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        # Check if files exist
        paths = ["cache/index.pkl", "cache/docmap.pkl"]
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing cache file: {path}")
            
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        with open("cache/term_frequencies.pkl", 'rb') as f:
            self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id, term):
        # tokenize term
        stemmer = PorterStemmer()
        token = [stemmer.stem(t) for t in remove_stopwords(tokenize(term))]

        # check if there is only 1 term
        if len(token) != 1:
            raise ValueError("term must be a single token")

        return self.term_frequencies[doc_id][token[0]]
    
    def get_bm25_idf(self, term: str) -> float:
        # tokenize term
        stemmer = PorterStemmer()
        term = [stemmer.stem(t) for t in remove_stopwords(tokenize(term))]

        if len(term) != 1:
            raise ValueError("term must be a single token")

        # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1) - (N - total number of documents, df document frequency)
        # Numerator (N - df + 0.5): Count of documents WITHOUT the term (plus smoothing)
        # Denominator (df + 0.5): Count of documents WITH the term (plus smoothing)
        bm25 = math.log((len(self.docmap) - len(self.index[term[0]]) + 0.5) / (len(self.index[term[0]]) + 0.5) + 1)
        return bm25
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
       # get the raw term frequency
       tf = self.get_tf(doc_id, term)

       # BM25 daturation formula (tf * (k1 + 1)) / (tf + k1)
       tf_score = (tf * (k1 + 1)) / (tf + k1)
       return tf_score

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return []

    stemmer = PorterStemmer()
    query_tokens = [stemmer.stem(t) for t in remove_stopwords(tokenize(query))]

    results = []
    seen = set()
    for token in query_tokens:
        for doc_id in idx.get_documents(token):
            if doc_id not in seen:
                seen.add(doc_id)
                results.append(idx.docmap[doc_id])
            if len(results) == 5:
                break
        if len(results) == 5:
            break

    return results

def tf_command(doc_id, term):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return []
    
    return idx.get_tf(doc_id, term)


def tokenize(text):
    cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return cleaned_text.split()

def remove_stopwords(tokens):
    stopwords = search_utils.load_stopwords()
    return [word for word in tokens if word not in stopwords]

# Inverse Document Frequency
def idf_command(term):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return []
    
    # tokenize term
    stemmer = PorterStemmer()
    term = [stemmer.stem(t) for t in remove_stopwords(tokenize(term))]
    
    # calculate the IDF
    idf = math.log((len(idx.docmap) +1) / (len(idx.index[term[0]]) +1))
    return idf

# TF-IDF
def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return []

    tfidf = tf_command(doc_id, term) * idf_command(term)
    return tfidf

def bm25_idf_command(term):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return None

    bm25 = idx.get_bm25_idf(term)
    return bm25

def bm25_tf_command(doc_id, term, k1=BM25_K1):
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        return None
    
    bm25_tf_score = idx.get_bm25_tf(doc_id, term, k1)
    return bm25_tf_score