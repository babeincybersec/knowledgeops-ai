"""Index a document: chunk -> embed -> store."""

import json
from pathlib import Path

from rag.chunker import chunk_document
from rag.embedder import Embedder
from rag.vector_store import VectorStore


def index_document(
    doc_path: Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 500,
    overlap: int = 100,
) -> int:
    """
    Load a processed document JSON, chunk it, embed chunks, store them.

    Returns the number of chunks indexed.
    """
    doc_dict = json.loads(doc_path.read_text(encoding="utf-8"))
    chunks = chunk_document(doc_dict, chunk_size=chunk_size, overlap=overlap)

    print(f"  Chunks: {len(chunks)}")

    # Embed all chunks in one batch (fast)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)

    # Store
    store.add_chunks(chunks, embeddings)
    return len(chunks)


if __name__ == "__main__":
    doc_path = Path("data/processed/employee_handbook.json")
    if not doc_path.exists():
        raise SystemExit(f"Not found: {doc_path}. Run: python -m ingestion.pipeline")

    print(f"Indexing: {doc_path}")
    embedder = Embedder()
    store = VectorStore()

    # Start fresh each time
    print("  Resetting vector store...")
    store.reset()

    n = index_document(doc_path, embedder, store)
    print(f"Indexed {n} chunks. Store now has {store.count()} chunks.")