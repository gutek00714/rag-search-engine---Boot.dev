from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from lib import search_utils

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if len(text) == 0 or text.strip() == "":
            raise ValueError("String is empty")
            
        # encode takes list and returns list
        text = [text]
        embedding = self.model.encode(text)

        return embedding[0]
    
    def build_embeddings(self, documents):  # documents - list of dicts
        self.documents = documents
        movies = []
        for document in self.documents:
            self.document_map.update({document["id"]: document})
            movie = f"{document['title']}: {document['description']}"
            movies.append(movie)

        self.embeddings = self.model.encode(movies, show_progress_bar = True)

        # save the embeddings
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map.update({document["id"]: document})

        # check the cache if the file exists
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                return self.build_embeddings(documents)
            
        return self.build_embeddings(documents)
    
def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")  

def verify_embeddings():
    ss = SemanticSearch()
    with open("data/movies.json", "r") as f:
        movies = search_utils.load_movies()

    embeddings = ss.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")