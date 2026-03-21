#!/usr/bin/env python3

import argparse

from lib.semantic_search import SemanticSearch, embed_chunks_command, embed_query_text, semantic_chunk, verify_embeddings, verify_model, embed_text, semantic_search, chunk

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk in words")
    chunk_parser.add_argument("--overlap", type=int, help="Number of words to share between consecutive chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text on sentence boundaries")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of sentences to shatre between consecutive chunks")

    subparsers.add_parser("embed_chunks", help="embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunks = chunk(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters\n")
            for i, item in enumerate(chunks, start=1):
                print(f"{i}. {item}\n")
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f'Semantically chunking {len(args.text)} characters')
            for i, item in enumerate(chunks, start=1):
                print(f"{i}. {item}")
        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()