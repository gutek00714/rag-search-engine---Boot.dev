import re

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from lib import search_utils

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents: list[dict] = []
        self.document_map = {}

    def generate_embedding(self, text) -> np.ndarray:
        if len(text) == 0 or text.strip() == "":
            raise ValueError("String is empty")
            
        # encode takes list and returns list
        text = [text]
        embedding = self.model.encode(text)

        return embedding[0]
    
    def build_embeddings(self, documents) -> np.ndarray:  # documents - list of dicts
        self.documents = documents
        movies = []
        for document in self.documents:
            self.document_map.update({document["id"]: document})
            movie = f"{document['title']}: {document['description']}"
            movies.append(movie)

        self.embeddings = self.model.encode(movies, show_progress_bar = True)

        # save the embeddings
        os.makedirs("cache", exist_ok=True)

        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents) -> np.ndarray:
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
    
    def search(self, query, limit) -> list[dict]:
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
    
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for document in self.documents:
            self.document_map.update({document["id"]: document})

        all_chunks = []
        chunk_meta = []   # hold metadata about each chunk

        for idx, document in enumerate(documents):
            # if description box is empty - skip
            if len(document['description'].strip()) == 0:
                continue
                
            # split description text into 4 sentence chunks with 1 sentence overlap
            chunks = semantic_chunk(document['description'], 4, 1)

            # add each chunk to the chunks list
            # for each chunk add a dict to chunk metadata
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_meta.append({
                    "movie_idx": idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                })

        # encode the entire list
        self.chunk_embeddings = self.model.encode(all_chunks)

        self.chunk_metadata = chunk_meta

        os.makedirs("cache", exist_ok=True)

        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for document in self.documents:
            self.document_map.update({document["id"]: document})

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    # search functionality that queries against chunk embeddings and aggregates results
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query) #vec1

        chunk_score = []

        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call load_or_create_chunk_embeddings first.")

        for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chunk_embedding, query_embedding)
            chunk_score.append({
                "chunk_idx": self.chunk_metadata[chunk_idx]["chunk_idx"],
                "movie_idx": self.chunk_metadata[chunk_idx]["movie_idx"],
                "score": score,
            })

        map_movie_index_to_score = {}

        for score in chunk_score:
            movie_idx = score["movie_idx"]
            if movie_idx not in map_movie_index_to_score or score["score"] > map_movie_index_to_score[movie_idx]:
                map_movie_index_to_score[movie_idx] = score["score"]

        # sort movie scores by score in descending order
        sorted_movie_scores = sorted(map_movie_index_to_score.items(), key=lambda item: item[1], reverse=True)

        # filter down to the top limit
        top_movies = sorted_movie_scores[:limit]

        results = []
        for movie_idx, score in top_movies:
            doc = self.documents[movie_idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, 4),
                "metadata": doc.get("metadata", {})
            })

        return results
    
    
def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text) -> None:
    ss = SemanticSearch()

    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")  

def verify_embeddings() -> None:
    ss = SemanticSearch()

    movies = search_utils.load_movies()

    embeddings = ss.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

# semantic search engine - find movies based on meaning rather than exact keyword matches
def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def semantic_search(query, limit) -> None:
    ss = SemanticSearch()

    movies = search_utils.load_movies()

    ss.load_or_create_embeddings(movies)

    output = ss.search(query, limit)

    for i, item in enumerate(output, start=1):
        print(f"{i}. {item['title']} (score: {item['score']:.4f})\n{item['description'][:100]}...\n")

# split text based on the chunk_size and overlap
def chunk(text, chunk_size, overlap) -> list[str]:
    # split text on whitespace
    text_split = text.split()

    # create chunks by chunk_size
    chunks = []
    for i in range(0, len(text_split), chunk_size - overlap):
        words = text_split[i:i + chunk_size]
        chunk_string = " ".join(words)
        chunks.append(chunk_string)

    return chunks

# split text based on a sentence boundaries to preserve meaning
def semantic_chunk(text, max_chunk_size, overlap) -> list[str]:
    # delete leading and trailing whitespace
    text = text.strip()
    if len(text) == 0:
        return []
    
    text_split = re.split(r"(?<=[.!?])\s+", text)

    # check if there is only 1 sentence and it ends with . ! ?
    if len(text_split) == 1 and not text.endswith((".", "!", "?")):
        text_split = [text]

    chunks = []
    for i in range(0, len(text_split), max_chunk_size - overlap):
        words = text_split[i:i + max_chunk_size]

        # Skip if this chunk is entirely within the previous chunk's overlap
        if i > 0 and i + max_chunk_size > len(text_split) and len(text_split) - i <= overlap:
            break
        chunk_string = " ".join(words)
        chunk_string = chunk_string.strip()
        if len(chunk_string) > 0:
            chunks.append(chunk_string)
    
    return chunks

def embed_chunks_command() -> np.ndarray:
    movies = search_utils.load_movies()

    css = ChunkedSemanticSearch()

    return css.load_or_create_chunk_embeddings(movies)

def search_chunked_command(query, limit) ->  dict:
    movies = search_utils.load_movies()

    css  = ChunkedSemanticSearch()

    css.load_or_create_chunk_embeddings(movies)

    results = css.search_chunks(query, limit)

    return {
        "query": query,
        "results": results
    }