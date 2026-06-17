"""
Dense Embedding Generator — wraps embedding model for batch processing.
Used by the ingestion pipeline to embed chunks before indexing.
"""

from typing import List
from src.utils import get_embeddings


class Embedder:
    """Generates dense embeddings for document chunks."""

    def __init__(self):
        self.model = get_embeddings()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return self.model.embed_query(query)

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        from config.settings import settings
        return settings.embedding_dimension