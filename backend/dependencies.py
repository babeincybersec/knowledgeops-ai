"""Shared resources for the API.

These are loaded once at startup and reused across requests.
"""

import os
from functools import lru_cache

from rag.embedder import Embedder
from rag.llm import get_client
from rag.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Load the embedding model once."""
    return Embedder()


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """Open the persistent vector store once."""
    return VectorStore()


@lru_cache(maxsize=1)
def get_llm():
    """Create the LLM client once."""
    return get_client()


def get_provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "gemini")