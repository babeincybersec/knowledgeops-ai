"""Generate embeddings for text using sentence-transformers."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load the model once and cache it."""
    print(f"Loading embedding model: {model_name}")
    print("(First run downloads ~80MB. Subsequent runs use cache.)")
    return SentenceTransformer(model_name)


class Embedder:
    """Wrapper around SentenceTransformer for project-specific use."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = _load_model(model_name)
        # Get the dimension (384 for all-MiniLM-L6-v2)
        self.dimension = self.model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns a list of vectors."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text. Returns one vector."""
        return self.embed([text])[0]


if __name__ == "__main__":
    e = Embedder()
    print(f"Model: {e.model_name}")
    print(f"Embedding dimension: {e.dimension}")

    # Sanity check: semantically similar texts should be close
    import numpy as np

    texts = [
        "How many vacation days do I get?",
        "What is the annual leave allowance?",
        "Password must be 12 characters.",
    ]
    vectors = e.embed(texts)

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    print("\nCosine similarities:")
    print(f"  vacation vs annual leave: {cosine(vectors[0], vectors[1]):.3f}  (should be high)")
    print(f"  vacation vs password:     {cosine(vectors[0], vectors[2]):.3f}  (should be low)")