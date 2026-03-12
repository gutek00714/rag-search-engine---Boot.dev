import os
import pickle
import string
from lib import search_utils
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for token in tokens:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

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

def keyword_search(query):
    # Load movie data from JSON storage
    movies_list = search_utils.load_movies()

    stemmer = PorterStemmer()
    results = []

    # Preprocess the user query into stemmed roots
    query_tokens = [stemmer.stem(token) for token in remove_stopwords(tokenize(query))]

    for item in movies_list:
        # Preprocess each title to match the query's format
        title_tokens = [stemmer.stem(token) for token in remove_stopwords(tokenize(item['title']))]

        # Check if any stemmed query word exists within the stemmed movie title tokens
        if any(q in t for q in query_tokens for t in title_tokens):
            results.append(item)

        # Limit retults
        if len(results) == 5:
            break

        # Print the result
        for i, item in enumerate(results, start=1):
            print(f"{i}. {item['title']}")

    return results

def tokenize(text):
    cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return cleaned_text.split()

def remove_stopwords(tokens):
    stopwords = search_utils.load_stopwords()
    return [word for word in tokens if word not in stopwords]
