import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"


def llm_judge_results(query: str, results: list[dict]) -> list[int]:
    if not api_key:
        return [0] * len(results)
    
    # 1. Format results as "1. Title", "2. Title", etc.
    formatted_results = []
    for i, title in enumerate(results, start=1):
        formatted_results.append(f"{i}. {title['doc']['title']}")


    # 2. Build the prompt and call the gemini API
    response = client.models.generate_content(model=model, contents=f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]""")

    # 3. Parse the JSON list from the response
    parsed_response = (response.text or "").strip().strip('"')
    scores = json.loads(parsed_response)

    # 5. Return list[int] of scores
    if len(scores) == len(results):
        return list(map(int, scores))

    raise ValueError(f"Expected {len(results)} scores, got {len(scores)}")
    