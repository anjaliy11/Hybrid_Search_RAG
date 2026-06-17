"""
Hybrid Retriever — fuses dense semantic search with sparse keyword matching.

Architecture:
  Query → [Dense (ChromaDB)] → candidates
  Query → [Sparse (BM25)]   → candidates
  Candidates → Reciprocal Rank Fusion → ranked results

RRF is provably better than either method alone for diverse query types.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import chromadb

from config.settings import settings
from src.utils import get_embeddings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A retrieved document with score and provenance tracking."""
    content: str
    metadata: dict
    score: float
    source: str  # "dense", "sparse", "hybrid", "reranked"
    rank: int = 0


class HybridRetriever:
    """
    Production hybrid retriever combining:
      - Dense: ChromaDB cosine similarity (semantic)
      - Sparse: BM25 Okapi (lexical)
      - Fusion: Reciprocal Rank Fusion (RRF)
    """

    def __init__(self):
        self.embeddings = get_embeddings()

        # Dense store
        self.chroma = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Sparse store
        self.bm25 = None
        self.bm25_chunks = []
        self._load_sparse()

    def _load_sparse(self):
        """Load BM25 index from disk."""
        path = Path(settings.bm25_index_path)
        if not path.exists():
            logger.warning("BM25 index not found — sparse search disabled")
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.bm25_chunks = data["chunks"]
        logger.info(f"BM25 loaded: {len(self.bm25_chunks)} chunks")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
    ) -> List[RetrievedDocument]:
        """
        Execute hybrid retrieval with RRF fusion.
        
        Args:
            query: Search query string
            top_k: Final number of results
            alpha: Dense weight (0-1). None uses config default.
        """
        alpha = alpha if alpha is not None else settings.hybrid_alpha

        dense_results = self._search_dense(query, settings.top_k_dense)
        sparse_results = self._search_sparse(query, settings.top_k_sparse)

        fused = self._rrf_fuse(dense_results, sparse_results, alpha)
        return fused[:top_k]

    def _search_dense(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """ChromaDB semantic search."""
        count = self.collection.count()
        if count == 0:
            return []

        embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                docs.append(RetrievedDocument(
                    content=doc,
                    metadata=meta or {},
                    score=max(0.0, 1.0 - dist),
                    source="dense",
                ))
        return docs

    def _search_sparse(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """BM25 keyword search."""
        if self.bm25 is None:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedDocument(
                content=self.bm25_chunks[i].content,
                metadata=self.bm25_chunks[i].metadata,
                score=float(scores[i]),
                source="sparse",
            )
            for i in top_idx if scores[i] > 0
        ]

    def _rrf_fuse(
        self,
        dense: List[RetrievedDocument],
        sparse: List[RetrievedDocument],
        alpha: float,
        k: int = 60,
    ) -> List[RetrievedDocument]:
        """
        Reciprocal Rank Fusion.
        
        RRF(d) = α/(k + rank_dense(d)) + (1-α)/(k + rank_sparse(d))
        
        k=60 is standard (original paper by Cormack et al., 2009).
        """
        scores: dict = {}
        doc_map: dict = {}

        for rank, doc in enumerate(dense):
            key = doc.content[:128]
            scores[key] = scores.get(key, 0.0) + alpha / (k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(sparse):
            key = doc.content[:128]
            scores[key] = scores.get(key, 0.0) + (1 - alpha) / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (key, score) in enumerate(sorted_items):
            doc = doc_map[key]
            results.append(RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=score,
                source="hybrid",
                rank=rank + 1,
            ))

        return results