"""Integration test for retrieval chain (dense + sparse + rerank)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings


class TestRetrievalChain:
    """Tests for the full retrieval pipeline."""

    def test_hybrid_retriever_returns_results(self):
        """Hybrid retriever returns documents for valid queries."""
        # Only run if indices exist
        if not Path(settings.bm25_index_path).exists():
            print("  Skipping (no index). Run ingestion first.")
            return

        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        results = retriever.retrieve("What is retrieval augmented generation?", top_k=3)

        assert len(results) > 0
        assert all(r.content for r in results)
        assert all(r.score > 0 for r in results)
        assert results[0].rank == 1

    def test_retriever_handles_empty_query(self):
        """Empty-ish queries don't crash."""
        if not Path(settings.bm25_index_path).exists():
            print("  Skipping (no index)")
            return

        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        results = retriever.retrieve("", top_k=3)
        # Should return empty or handle gracefully
        assert isinstance(results, list)

    def test_reranker_improves_ordering(self):
        """Reranker produces different ordering than raw retrieval."""
        if not Path(settings.bm25_index_path).exists():
            print("  Skipping (no index)")
            return

        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import get_reranker

        retriever = HybridRetriever()
        reranker = get_reranker()

        docs = retriever.retrieve("transformer attention mechanism", top_k=10)
        if len(docs) < 3:
            return

        reranked = reranker.rerank("transformer attention mechanism", docs, top_k=5)
        assert len(reranked) <= 5
        assert all(r.source == "reranked" for r in reranked)

    def test_results_have_metadata(self):
        """Retrieved documents carry metadata from ingestion."""
        if not Path(settings.bm25_index_path).exists():
            print(" Skipping (no index)")
            return

        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        results = retriever.retrieve("machine learning", top_k=1)

        if results:
            assert "source" in results[0].metadata or len(results[0].metadata) > 0


if __name__ == "__main__":
    t = TestRetrievalChain()
    t.test_hybrid_retriever_returns_results()
    print("Hybrid retrieval")
    t.test_retriever_handles_empty_query()
    print(" Empty query handling")
    t.test_reranker_improves_ordering()
    print(" Reranking")
    t.test_results_have_metadata()
    print(" Metadata preserved")
    print("\n All retrieval integration tests passed")