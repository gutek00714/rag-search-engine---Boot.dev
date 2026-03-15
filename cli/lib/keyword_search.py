import os
import pickle
import string
from typing import Counter
from lib import search_utils
from nltk.stem import PorterStemmer
from collections import defaultdict
import math

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
            raise Exception("Term isn't 1")

        return self.term_frequencies[doc_id][token[0]]
       

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
    print(f"IDF value: {idf}")
    return idf