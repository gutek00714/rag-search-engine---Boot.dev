#!/usr/bin/env python3

import argparse
from lib.keyword_search import keyword_search, InvertedIndex



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            keyword_search(args.query)
        case "build":
            idx = InvertedIndex()
            idx.build()
            idx.save()
            docs = idx.get_documents("merida")
            print(f"First document for 'merida': {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()