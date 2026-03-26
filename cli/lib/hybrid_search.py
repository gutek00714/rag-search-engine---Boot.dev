import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index_pkl"):
            self.idx.build()
            self.idx.save()
        else:
            self.idx.load()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        # get bm25 results
        bm25 = self._bm25_search(query, limit * 500)

        # get semantic results
        semantic = self.semantic_search.search_chunks(query, limit * 500)

        # combine all the results
        # -----------------------
        bm25_scores = normalize([score for _, _, score in bm25])
        semantic_scores = normalize([item["score"] for item in semantic])

        # map id to the document with its keyword and semantic search
        combined = {}

        for i, (doc_id, title, _) in enumerate(bm25):
            combined[doc_id] = {
                "doc": self.idx.docmap[doc_id],
                "keyword_score": bm25_scores[i],
                "semantic_score": 0.0,
            }
        
        for i, item in enumerate(semantic):
            doc_id = item["id"]
            if doc_id in combined:
                combined[doc_id]["semantic_score"] = semantic_scores[i]
            else:
                combined[doc_id] = {
                    "doc": self.documents[item["id"]] if isinstance(self.documents, dict) else next(d for d in self.documents if d["id"] == doc_id),
                    "keyword_score": 0.0,
                    "semantic_score": semantic_scores[i],
                }

        # add hybrid score to each document
        for doc_id in combined:
            combined[doc_id]["hybrid_score"] = (alpha * combined[doc_id]["keyword_score"] + (1 - alpha) * combined[doc_id]["semantic_score"])
        #--------------------

        sorted_results = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return sorted_results[:limit]


    def rrf_search(self, query, k, limit=10):
        # get BM25 results
        bm25 = self._bm25_search(query, limit * 500)

        # get semantic results
        semantic = self.semantic_search.search_chunks(query, limit * 500)

        # combine both results using Reciprocal Rank Fusion
        #--------------------
        combined = {}   # map document id to documents themselves and their bm25 and sematntic ranks
        for i, (doc_id, title, score) in enumerate(bm25, start=1):
            rrf = rrf_score(i, k)
            combined[doc_id] = {
                "doc": self.idx.docmap[doc_id],
                "bm25_rank": i,
                "semantic_rank": None,
                "rrf_score": rrf
            }

        for i, results in enumerate(semantic, start=1): #results = {id, title, document, score, metadata}
            doc_id = results["id"]
            rrf = rrf_score(i, k)
            if doc_id in combined:
                combined[doc_id]["semantic_rank"] = i
                combined[doc_id]["rrf_score"] += rrf
            else:
                combined[doc_id] = {
                    "doc": self.idx.docmap[doc_id],
                    "bm25_rank": None,
                    "semantic_rank": i,
                    "rrf_score": rrf
                }

        #--------------------
        sorted_results = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results[:limit]

    
def normalize(scores) -> list:
    if len(scores) == 0:
        return []
    
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

def rrf_score(rank, k=60):
    return 1 / (k + rank)