import os
import time
from dotenv import load_dotenv
from google import genai

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
        title = movie.get("title", "")
        document = movie.get("document", "")
        
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

def rerank(query, documents, method=None, limit=5):
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    else:
        return documents[:limit]