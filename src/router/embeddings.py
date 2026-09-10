"""OpenAI embedding adapter for a single incoming query."""

import os

import numpy as np
from openai import OpenAI

from router.config import EMBEDDING_MODEL


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Return L2-normalized vectors for cosine similarity/distance operations."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class QueryEmbedder:
    """Create a normalized embedding using the configured OpenAI model."""

    def __init__(self, client: OpenAI | None = None, model_name: str = EMBEDDING_MODEL) -> None:
        self.client = client or OpenAI()
        self.model_name = model_name

    def embed(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before routing a query.")
        response = self.client.embeddings.create(
            model=self.model_name, input=query, encoding_format="float"
        )
        vector = np.asarray(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        return normalize(vector)
