"""Semantic search over indexed documents."""

import argparse

from rag.embedder import Embedder
from rag.vector_store import VectorStore


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the vector store for chunks similar to the query."""
    embedder = Embedder()
    store = VectorStore()

    query_vec = embedder.embed_one(query)
    return store.query(query_vec, top_k=top_k)


def print_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print("=" * 70)
    for i, r in enumerate(results, start=1):
        # Cosine distance: 0 = identical, 2 = opposite. Lower is better.
        similarity = 1 - r["distance"]
        meta = r["metadata"]
        print(f"\n[{i}] similarity={similarity:.3f}  source={meta['source_name']}  page={meta['page_number']}")
        print("-" * 70)
        # Print with wrapping
        text = r["text"]
        print(text[:400] + ("..." if len(text) > 400 else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the knowledge base")
    parser.add_argument("query", help="the question to search for")
    parser.add_argument("--top-k", type=int, default=5, help="number of results")
    args = parser.parse_args()

    results = search(args.query, top_k=args.top_k)
    print_results(args.query, results)