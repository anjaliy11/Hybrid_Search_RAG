"""
Local Embedding Model — runs on CPU, zero API cost.
Singleton pattern avoids reloading the model on every call.
"""

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

_instance: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns singleton embedding model.

    Model: all-MiniLM-L6-v2
      - 384 dimensions
      - Normalized cosine similarity
      - Fast CPU inference
    """
    global _instance

    if _instance is None:
        logger.info(f"Loading embeddings: {settings.embedding_model}")
        _instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 64,
            },
        )
        logger.info("Embeddings loaded ✓")

    return _instance