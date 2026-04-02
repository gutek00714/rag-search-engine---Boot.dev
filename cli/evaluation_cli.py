import argparse
import json
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # load golden_dataset
    with open('data/golden_dataset.json', 'r') as f:
        golden_dataset = json.load(f)

    movies = load_movies()
    hs = HybridSearch(movies)

    print(f"k={limit}")

    for case in golden_dataset["test_cases"]:
        data = hs.rrf_search(case["query"], k=60, limit=limit)

        # extract titles from data
        retrieved_titles = [result["doc"]["title"] for result in data]

        relevant_retrieved = sum(1 for title in retrieved_titles if title in case["relevant_docs"])

        # measures the quality of search results (how much of what you found is relevant)
        precision = relevant_retrieved / len(retrieved_titles)

        # measures completeness - the percentage of all relevant documents you actually retrieved (how much of what's relevant did you find)
        recall = relevant_retrieved / len(case["relevant_docs"])

        print(f" - Query: {case['query']}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Recall@{limit}: {recall:.4f}")
        print(f"    - Retrieved: {', '.join(retrieved_titles)}")
        print(f"    - Relevant: {', '.join(case['relevant_docs'])}")


if __name__ == "__main__":
    main()