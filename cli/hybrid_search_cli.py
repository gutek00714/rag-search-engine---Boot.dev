import argparse

from lib.hybrid_search import HybridSearch, normalize
from lib.search_utils import load_movies
from lib.query_enhancement import enhance_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalnize BM25 and Cosine score")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores")

    weighted_search = subparsers.add_parser("weighted-search", help="Combine keyword(BM25) and semantic scores")
    weighted_search.add_argument("query", type=str, help="Search query")
    weighted_search.add_argument("--alpha", type=float, default=0.5, help="Alpha constant")
    weighted_search.add_argument("--limit", type=int, default=5, help="Limit query")

    rrf_search = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion")
    rrf_search.add_argument("query", type=str, help="Search query")
    rrf_search.add_argument("-k", type=int, default=60, help="Parameter that controls how much more weight we give to a higher-ranked results vs. lower-ranked ones")
    rrf_search.add_argument("--limit", type=int, default=5, help="Limit query")
    rrf_search.add_argument("--enhance", type=str, choices=["spell"], help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize(args.scores)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            movies = load_movies()
            hs = HybridSearch(movies)
            search = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, item in enumerate(search, start=1):
                print(f"{i}. {item['doc']['title']}")
                print(f"Hybrid Score: {item['hybrid_score']:.3f}")
                print(f"BM25: {item['keyword_score']:.3f}, Semantic: {item['semantic_score']:.3f}")
                print(f"{item['doc']['description'][:100]}...")
        case "rrf-search":
            movies = load_movies()
            hs = HybridSearch(movies)
            enhanced = enhance_query(args.query, args.enhance)
            if enhanced != args.query:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced}'\n")
            rrf = hs.rrf_search(enhanced, args.k, args.limit)
            for i, item in enumerate(rrf, start=1):
                print(f"{i}. {item['doc']['title']}")
                print(f"RRF Score: {item['rrf_score']:.3f}")
                print(f"BM25 Rank: {item['bm25_rank']}, Semantic Rank: {item['semantic_rank']}")
                print(f"{item['doc']['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()