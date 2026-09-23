"""ChromaDB wrapper for storing and retrieving chunk embeddings."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from rag.chunker import Chunk


DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION = "knowledgeops"


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.db_path = str(db_path)
        self.collection_name = collection_name

        # PersistentClient writes to disk so embeddings survive restarts
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # use cosine similarity
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with their precomputed embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )
        if not chunks:
            return

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source_name": c.source_name,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find the top_k chunks closest to the query embedding.

        Returns a list of dicts with keys: id, text, metadata, distance.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # ChromaDB returns parallel lists; flatten into a list of dicts
        out: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            out.append({
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": distances[i],
            })
        return out

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all chunks. Useful for re-indexing."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


if __name__ == "__main__":
    store = VectorStore()
    print(f"Collection: {store.collection_name}")
    print(f"Stored chunks: {store.count()}")