import argparse

from lib.augmented_generation import generate_answer
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, RRF_K


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            movies = load_movies()
            hs = HybridSearch(movies)

            rrf = hs.rrf_search(query, k=RRF_K, limit=5)

            answer = generate_answer(query, rrf)

            print("Search Results:")
            for item in rrf:
                print(f"- {item['doc']['title']}")
            print()
            print("RAG Response:")
            print(f"{answer}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()