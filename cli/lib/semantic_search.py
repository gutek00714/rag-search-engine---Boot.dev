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
    
    def search(self, query, limit):
        # check if embeddings are loaded
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        # generate an embedding for a query
        embedding = self.generate_embedding(query)

        # calculate cosine similarity between query embedding and each document embedding
        similarity_list = []
        for i, value in enumerate(self.embeddings):
            similarity_score = cosine_similarity(embedding, value)
            corresponding_movie = self.documents[i]
            similarity_list.append((similarity_score, corresponding_movie))

        # sort the list by similarity score desc
        similarity_list.sort(key=lambda tup: tup[0], reverse=True)

        # return top retults as a list of dict (score, title, description)
        top_results = similarity_list[:limit]

        final_output = []
        for score, doc in top_results:
            final_output.append({
                "score": float(score),
                "title": doc['title'],
                "description": doc['description']
            })

        return final_output
    
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

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

# semantic search engine - find movies based on meaning rather than exact keyword matches
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def semantic_search(query, limit):
    ss = SemanticSearch()

    movies = search_utils.load_movies()

    ss.load_or_create_embeddings(movies)

    output = ss.search(query, limit)

    for i, item in enumerate(output, start=1):
        print(f"{i}. {item['title']} (score: {item['score']:.4f})\n{item['description'][:100]}...\n")

def chunk(text, chunk_size, overlap):
    # split text on whitespace
    text_split = text.split()

    # create chunks by chunk_size
    chunks = []
    for i in range(0, len(text_split), chunk_size - overlap):
        words = text_split[i:i + chunk_size]
        chunk_string = " ".join(words)
        chunks.append(chunk_string)

    # print result
    print(f"Chunking {len(text)} characters\n")
    for i, item in enumerate(chunks, start=1):
        print(f"{i}. {item}\n")
