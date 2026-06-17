"""
Sparse Index Builder — creates and persists BM25 index.
Separated from pipeline for independent rebuild capability.
"""

import pickle
import logging
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi
from config.settings import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class SparseIndexer:
    """Builds and persists BM25 sparse retrieval index."""

    def __init__(self, index_path: str = None):
        self.index_path = Path(index_path or settings.bm25_index_path)

    def build(self, chunks: List[Chunk]):
        """Build BM25 index from chunks and save to disk."""
        corpus = [c.content.lower().split() for c in chunks]
        bm25 = BM25Okapi(corpus)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks}, f)

        logger.info(f"BM25 index: {len(chunks)} entries → {self.index_path}")

    def load(self) -> dict:
        """Load persisted index."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {self.index_path}")

        with open(self.index_path, "rb") as f:
            return pickle.load(f)
