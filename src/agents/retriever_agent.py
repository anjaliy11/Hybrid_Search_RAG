"""
Retriever Agent — executes hybrid search + reranking for all sub-queries.
Deduplicates and produces the final context documents.
"""

import logging
from typing import List

from config.settings import settings
from src.agents.state import AgentState
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedDocument
from src.retrieval.reranker import get_reranker

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Orchestrates retrieval across sub-queries with dedup and reranking."""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = get_reranker()

    def run(self, state: AgentState) -> dict:
        """
        Retrieve for each sub-query → deduplicate → rerank against original.
        """
        sub_queries = state.get("sub_queries") or [state["current_query"]]
        attempts = state.get("retrieval_attempts", 0)

        logger.info(f"Retrieving for {len(sub_queries)} queries (attempt {attempts+1})")

        # Gather candidates from all sub-queries
        all_docs: List[RetrievedDocument] = []
        for q in sub_queries:
            docs = self.retriever.retrieve(q, top_k=10)
            all_docs.extend(docs)

        # Deduplicate by content prefix
        seen = set()
        unique = []
        for doc in all_docs:
            key = doc.content[:128]
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        # Rerank against original query
        if unique:
            reranked = self.reranker.rerank(
                query=state["original_query"],
                documents=unique,
                top_k=settings.top_k_rerank,
            )
        else:
            reranked = []

        # Serialize for state transport
        retrieved = [
            {
                "content": d.content,
                "metadata": d.metadata,
                "score": round(d.score, 4),
                "rank": d.rank,
            }
            for d in reranked
        ]

        logger.info(f"Retrieved {len(retrieved)} documents after reranking")

        return {
            "retrieved_documents": retrieved,
            "retrieval_attempts": attempts + 1,
            "next_agent": "synthesizer",
        }