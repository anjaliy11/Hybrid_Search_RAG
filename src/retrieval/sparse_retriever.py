"""
Sparse Retriever — BM25 keyword matching.
Separated for independent use and testing.
"""

import pickle
import logging
from pathlib import Path
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import settings
from src.retrieval.hybrid_retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class SparseRetriever:
    """BM25-based keyword retrieval."""

    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self._load()

    def _load(self):
        path = Path(settings.bm25_index_path)
        if not path.exists():
            logger.warning("BM25 index not found")
            return

        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def search(self, query: str, top_k: int = 20) -> List[RetrievedDocument]:
        """BM25 keyword search."""
        if self.bm25 is None:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedDocument(
                content=self.chunks[i].content,
                metadata=self.chunks[i].metadata,
                score=float(scores[i]),
                source="sparse",
            )
            for i in top_idx if scores[i] > 0
        ]