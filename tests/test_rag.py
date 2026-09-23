"""Tests for chunking, embedding, vector search."""

import json
from pathlib import Path
import pytest
from rag.chunker import Chunk, chunk_page, chunk_document
from rag.embedder import Embedder
from rag.vector_store import VectorStore

PROCESSED_DOC = Path("data/processed/employee_handbook.json")


def test_chunk_page_produces_chunks():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_page(text, source_name="test.pdf", page_number=1, chunk_size=30, overlap=5)
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_ids_are_unique():
    text = "A. B. C. D. E. F. G."
    chunks = chunk_page(text, source_name="test.pdf", page_number=1, chunk_size=10, overlap=2)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_respects_page_boundaries():
    text = "Content here. More content."
    chunks = chunk_page(text, source_name="test.pdf", page_number=2)
    assert all(c.page_number == 2 for c in chunks)


@pytest.mark.skipif(not PROCESSED_DOC.exists(), reason="Run ingestion first")
def test_chunk_document_uses_source_name():
    doc = json.loads(PROCESSED_DOC.read_text(encoding="utf-8"))
    chunks = chunk_document(doc)
    assert len(chunks) > 0


def test_embedder_dimension():
    e = Embedder()
    assert e.dimension == 384
    assert len(e.embed_one("hello")) == 384


def test_embedder_semantic_similarity():
    import numpy as np
    e = Embedder()
    a, b, c = e.embed(["vacation days off", "annual leave allowance", "password characters"])
    def cos(x, y):
        x, y = np.array(x), np.array(y)
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    assert cos(a, b) > cos(a, c)


def test_vector_store_add_and_query(tmp_path):
    store = VectorStore(db_path=tmp_path / "test_chroma", collection_name="test_col")
    assert store.count() == 0
    chunks = [
        Chunk(chunk_id="a", text="vacation", source_name="t.pdf", page_number=1, chunk_index=0, char_count=8),
        Chunk(chunk_id="b", text="password", source_name="t.pdf", page_number=2, chunk_index=0, char_count=8),
    ]
    embeddings = [[1.0, 0.0, 0.0] + [0.0] * 381, [0.0, 1.0, 0.0] + [0.0] * 381]
    store.add_chunks(chunks, embeddings)
    assert store.count() == 2
    results = store.query([1.0, 0.0, 0.0] + [0.0] * 381, top_k=1)
    assert results[0]["id"] == "a"


def test_vector_store_reset(tmp_path):
    store = VectorStore(db_path=tmp_path / "test_reset", collection_name="test_reset")
    store.add_chunks(
        [Chunk(chunk_id="x", text="hi", source_name="t.pdf", page_number=1, chunk_index=0, char_count=2)],
        [[0.1] * 384],
    )
    assert store.count() == 1
    store.reset()
    assert store.count() == 0
