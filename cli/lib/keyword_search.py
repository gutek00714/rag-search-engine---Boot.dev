import os
import pickle
import string
from typing import Counter
from lib import search_utils
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
from lib.search_utils import BM25_B, BM25_K1

class InvertedIndex:
    def __init__(self):
        self.index = {} #{"love": {1, 42, 99}, "dragon": {5, 42}}
        # Maps doc_id -> full movie dict (title, description, etc.)
        self.docmap = {}    #{1: {"id": 1, "title": "Titanic", "description": "..."}}
        # Maps doc_id -> Counter of how many times each token appears in that doc.
        self.term_frequencies = defaultdict(Counter)    #{1: Counter({"love": 3, "sea": 2})}
        # Maps doc_id -> total number of tokens in that document.
        self.doc_lengths = {}   #{1: 120, 42: 85}
        self.doc_lengths_path = os.path.join(search_utils.CACHE_DIR, "doc_lengths.pkl")

    # Example: doc_id=1, text="I love dragons and love fire"
    # tokens = ["love", "dragon", "fire"]  (after stemming/stopword removal)
    # index["love"].add(1), index["dragon"].add(1), index["fire"].add(1)
    # term_frequencies[1] = Counter({"love": 2, "dragon": 1, "fire": 1})
    # doc_lengths[1] = 4  (total tokens before dedup)
    def __add_document(self, doc_id, text) -> None:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in remove_stopwords(tokenize(text))]
        for token in tokens:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

        # add to term_frequencies counter
        self.term_frequencies[doc_id].update(tokens)

        # get tokens len
        self.doc_lengths[doc_id] = len(tokens)

    # Given a single stemmed token, returns a sorted list of doc IDs that contain it.
    # Example: get_documents("love") -> [1, 42, 99]
    def get_documents(self, term) -> list[int]:
        term = term.lower()
        ids = self.index.get(term, set())
        return sorted(ids)
    
    # Loads all movies from disk and indexes each one.
    def build(self) -> None:
        movies = search_utils.load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self) -> None:
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

        # Save doc_length data
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
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

        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)

    # TF = Term Frequency: how many times a term appears in a specific document.
    # Example: movie 1 has "love" 3 times -> get_tf(1, "love") -> 3
    def get_tf(self, doc_id, term) -> int:
        # tokenize term
        stemmer = PorterStemmer()
        token = [stemmer.stem(t) for t in remove_stopwords(tokenize(term))]

        # check if there is only 1 term
        if len(token) != 1:
            raise ValueError("term must be a single token")

        return self.term_frequencies[doc_id][token[0]]

    # BM25 IDF = Inverse Document Frequency (BM25 variant).
    # Measures how *rare* a term is across all documents.
    # Common words like "the" appear in many docs -> low IDF (less useful).
    # Rare words like "mowgli" appear in few docs -> high IDF (more useful). 
    def get_bm25_idf(self, term: str) -> float:
        # tokenize term
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in remove_stopwords(tokenize(term))]

        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1) - (N - total number of documents, df document frequency)
        # Numerator (N - df + 0.5): Count of documents WITHOUT the term (plus smoothing)
        # Denominator (df + 0.5): Count of documents WITH the term (plus smoothing)
        df = len(self.index.get(tokens[0], set()))
        bm25 = math.log((len(self.docmap) - df + 0.5) / (df + 0.5) + 1)
        return bm25
    
    # BM25 TF = saturated, length-normalized term frequency.
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
       # get the raw term frequency
       tf = self.get_tf(doc_id, term)

       # get doc_length
       doc_len = self.doc_lengths[doc_id]

       # get avg_doc_length
       doc_len_avg = self.__get_avg_doc_length()

       # calculate length_norm = 1 - b + b * (doc_length / avg_doc_length)
       length_norm = 1 - b + b * (doc_len / doc_len_avg)

       # BM25 daturation formula (tf * (k1 + 1)) / (tf + k1)
       # UPDATE add length_norm at the end
       tf_score = (tf * (k1 + 1)) / (tf + k1 * length_norm)
       return tf_score
    
    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        
        # calculate average from all docs
        all = 0
        for doc_id in self.doc_lengths:
            all += self.doc_lengths[doc_id]

        result = all / len(self.doc_lengths)
        return result

    # Combines BM25 TF and BM25 IDF into a single relevance score
    # for one (document, term) pair.
    # Score = bm25_tf * bm25_idf
    # Example: bm25_tf=1.82, bm25_idf=5.2 -> bm25 score = 9.46
    def bm25(self, doc_id, term) -> float:
        tf = self.get_bm25_tf(doc_id, term, k1=BM25_K1, b=BM25_B)
        idf = self.get_bm25_idf(term)
        return tf * idf
    
    # Full BM25 search over all documents for a multi-word query.
    # Steps:
    #   1. Tokenize the query into stemmed tokens.
    #      e.g. "love story" -> ["love", "stori"]
    #   2. For every document in the index, sum the bm25() score for each token.
    #      e.g. scores[doc_1] = bm25(doc_1, "love") + bm25(doc_1, "stori")
    #   3. Sort documents by total score, highest first.
    #   4. Return the top `limit` results with their doc_id, title, and score.
    def bm25_search(self, query, limit) -> list[tuple[int, str, float]]:
        # tokenize query
        stemmer = PorterStemmer()
        query = [stemmer.stem(t) for t in remove_stopwords(tokenize(query))]

        # initialize scores dict (map doc id to total bm25 scores)
        scores = {}

        for doc_id in self.docmap:
            total = 0
            for token in query:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total

        # sort the dict desc
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            results.append((doc_id, doc["title"], score))
        return results


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query) -> list[dict]:
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

def tf_command(doc_id, term) -> int:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        raise
    
    return idx.get_tf(doc_id, term)


def tokenize(text) -> list[str]:
    cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return cleaned_text.split()

def remove_stopwords(tokens) -> list[str]:
    stopwords = search_utils.load_stopwords()
    return [word for word in tokens if word not in stopwords]

# Inverse Document Frequency
def idf_command(term) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        raise
    
    # tokenize term
    stemmer = PorterStemmer()
    term = [stemmer.stem(t) for t in remove_stopwords(tokenize(term))]
    
    # calculate the IDF
    idf = math.log((len(idx.docmap) +1) / (len(idx.index[term[0]]) +1))
    return idf

# TF-IDF
def tfidf_command(doc_id, term) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        raise

    tfidf = tf_command(doc_id, term) * idf_command(term)
    return tfidf

def bm25_idf_command(term) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        raise

    bm25 = idx.get_bm25_idf(term)
    return bm25

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        raise
    
    bm25_tf_score = idx.get_bm25_tf(doc_id, term, k1, b)
    return bm25_tf_score

def bm25_search_command(query, limit) -> list[tuple[int,str,float]]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(e)
        raise
    
    return idx.bm25_search(query, limit)