import json
import os
import time
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

# load api key from .env
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# use api key to create a new instance of gemini client
client = genai.Client(api_key=api_key)

# get a response (returns object)
model = "gemma-3-27b-it"

def llm_rerank_individual(query, documents, limit=5):
    scored_docs = []

    for doc in documents:
        movie = doc["doc"]
        title = doc.get("title", "")
        document = doc.get("document", "")
        
        response = client.models.generate_content(model=model, contents=f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {document}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:""")
        
        score_text = (response.text or "").strip().strip('"')
        score = float(score_text)
        scored_docs.append({**doc, "individual_score": score})
        time.sleep(3)

    sorted_results = sorted(scored_docs, key=lambda x: x["individual_score"], reverse=True)
    return sorted_results[:limit]

def llm_rerank_batch(query, documents, limit=5):
    lines=[]
    for doc in documents:
        movie = doc["doc"]
        lines.append(f"{movie['id']}: {movie['title']} - {movie.get('document', movie.get('description', ''))[:200]}")
    doc_list_str = "\n".join(lines)

    # JSON list
    response = client.models.generate_content(model=model, contents=f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:""")
    
    #parse the JSON to get new ranks
    ranking_text = (response.text or "").strip()
    new_ranks = json.loads(ranking_text)

    doc_map = {}
    for doc in documents:
        doc_map[doc["doc"]["id"]] = doc

    # add new ranking into documents
    reranked = []
    for i, doc_id in enumerate(new_ranks):
        if doc_id in doc_map:
            doc_map[doc_id]["batch_rank"] = i + 1
            reranked.append(doc_map[doc_id])

    #sort by rank
    sorted_reranked = sorted(reranked, key=lambda x: x["batch_rank"])
    return sorted_reranked[:limit]

def llm_rerank_cross_encoder(query, documents, limit=5):
    pairs = [] #list of "pair" lists
    
    for doc in documents:
        pairs.append([query, f"{doc["doc"].get('title', '')} - {doc["doc"].get('document', '')}"])

    # create an instance of crossencoder and compute scores for all the pairs
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    # `predict` returns a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)

    # attach scores back to document
    scored_docs = []
    for i, doc in enumerate(documents):
        scored_docs.append({**doc, "cross_encoder_score": float(scores[i])})

    sorted_scores = sorted(scored_docs, key=lambda x: x["cross_encoder_score"], reverse=True)

    return sorted_scores[:limit]


def rerank(query, documents, method=None, limit=5):
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    elif method == "batch":
        return llm_rerank_batch(query, documents, limit)
    elif method == "cross_encoder":
        return llm_rerank_cross_encoder(query, documents, limit)
    else:
        return documents[:limit]