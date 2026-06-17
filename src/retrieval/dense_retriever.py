"""
Dense Retriever — ChromaDB vector similarity search.
Separated for independent use and testing.
"""

import logging
from typing import List

import chromadb
from config.settings import settings
from src.utils import get_embeddings
from src.retrieval.hybrid_retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class DenseRetriever:
    """ChromaDB-based semantic similarity retrieval."""

    def __init__(self):
        self.embeddings = get_embeddings()
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, top_k: int = 20) -> List[RetrievedDocument]:
        """Semantic search via cosine similarity."""
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