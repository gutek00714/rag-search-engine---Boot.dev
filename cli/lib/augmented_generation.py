# load api key from .env
import os

from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# use api key to create a new instance of gemini client
client = genai.Client(api_key=api_key)

# get a response (returns object)
model = "gemma-3-27b-it"

def generate_answer(query, rrf: list[dict]) -> str:
    context = ""
    for result in rrf:
        context += f"{result['doc']['title']}: {result['doc'].get('description', '')}\n\n"


    response = client.models.generate_content(model=model, contents=f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{context}

Answer:""")
    
    return (response.text or "").strip()


def summarize_answer(query, rrf: list[dict]) -> str:
    context = ""
    for result in rrf:
        context += f"{result['doc']['title']}: {result['doc'].get('description', '')}\n\n"

    response = client.models.generate_content(model=model, contents=f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{context}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:""")
    
    return (response.text or "").strip()
