"""
Cross-Encoder Reranker — improves precision at top-k.

Uses a cross-encoder that jointly encodes (query, document) pairs
for more accurate relevance scoring than bi-encoder similarity.

Model runs locally on CPU — no API needed.
"""

import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder
from src.retrieval.hybrid_retriever import RetrievedDocument

logger = logging.getLogger(__name__)

_instance: Optional["Reranker"] = None


def get_reranker() -> "Reranker":
    """Singleton reranker to avoid reloading model."""
    global _instance
    if _instance is None:
        _instance = Reranker()
    return _instance


class Reranker:
    """Cross-encoder reranker for precision improvement."""

    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        logger.info(f"Loading reranker: {self.MODEL}")
        self.model = CrossEncoder(self.MODEL)
        logger.info("Reranker loaded ✓")

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        """
        Rerank candidates using cross-encoder scoring.
        
        Cross-encoders are ~10x more accurate than bi-encoders
        but ~100x slower, so we only apply to pre-filtered candidates.
        """
        if not documents:
            return []

        pairs = [(query, doc.content) for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.score = float(score)

        reranked = sorted(documents, key=lambda d: d.score, reverse=True)

        for rank, doc in enumerate(reranked):
            doc.rank = rank + 1
            doc.source = "reranked"

        return reranked[:top_k]