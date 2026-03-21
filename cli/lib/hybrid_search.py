import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize(scores) -> list:
    if len(scores) == 0:
        return
    
    # find max and min score in a list
    max_score = max(scores)
    min_score = min(scores)

    if min_score == max_score:
        for i, item in enumerate(scores):
            scores[i] = 1.0
        
        return scores
    
    for i, item in enumerate(scores):
        scores[i] = (item - min_score) / (max_score - min_score)
    
    return scores