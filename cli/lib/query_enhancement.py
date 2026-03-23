import os
from typing import Optional
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

def spell_correct(query: str) -> str:
    response = client.models.generate_content(model=model, contents = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
""")
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query

def enhance_query(query: str, method: Optional[str]) -> str:
    if method == "spell":
        return spell_correct(query)
    else:
        return query
