import argparse

from lib.augmented_generation import citations_answer, generate_answer, summarize_answer
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, RRF_K


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize the results")
    summarize_parser.add_argument("query", type=str, help="Query to summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Limit query")

    citations_parser = subparsers.add_parser("citations", help="Add citations to the result")
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="Limit query")

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
        case "summarize":
            query = args.query
            movies = load_movies()
            hs = HybridSearch(movies)

            rrf = hs.rrf_search(query, k=RRF_K, limit=args.limit)

            summ = summarize_answer(query, rrf)

            print("Search Results:")
            for item in rrf:
                print(f"    - {item['doc']['title']}")
            print()
            print("LLM Summary:")
            print(f"{summ}")
        case "citations":
            query = args.query
            movies = load_movies()
            hs = HybridSearch(movies)

            rrf = hs.rrf_search(query, k=RRF_K, limit=args.limit)

            citations = citations_answer(query, rrf)

            print("Search Results:")
            for item in rrf:
                print(f"    - {item['doc']['title']}")
            print()
            print("LLM Answer:")
            print(f"{citations}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()