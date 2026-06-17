"""Unit tests for hybrid retrieval and RRF fusion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedDocument


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion logic."""

    def test_fusion_combines_both_sources(self):
        """RRF produces results from both dense and sparse."""
        retriever = HybridRetriever.__new__(HybridRetriever)

        dense = [
            RetrievedDocument(content="Dense doc 1", metadata={}, score=0.9, source="dense"),
            RetrievedDocument(content="Dense doc 2", metadata={}, score=0.8, source="dense"),
        ]
        sparse = [
            RetrievedDocument(content="Sparse doc 1", metadata={}, score=5.0, source="sparse"),
            RetrievedDocument(content="Dense doc 1", metadata={}, score=4.0, source="sparse"),  # overlap
        ]

        fused = retriever._rrf_fuse(dense, sparse, alpha=0.5)

        assert len(fused) >= 2
        # Overlapping doc should rank higher (boosted by both)
        contents = [d.content for d in fused]
        assert "Dense doc 1" in contents

    def test_fusion_alpha_weighting(self):
        """Alpha=1.0 should heavily favor dense results."""
        retriever = HybridRetriever.__new__(HybridRetriever)

        dense = [RetrievedDocument(content="Dense only", metadata={}, score=0.9, source="dense")]
        sparse = [RetrievedDocument(content="Sparse only", metadata={}, score=5.0, source="sparse")]

        # All dense weight
        fused_dense = retriever._rrf_fuse(dense, sparse, alpha=1.0)
        # All sparse weight
        fused_sparse = retriever._rrf_fuse(dense, sparse, alpha=0.0)

        assert fused_dense[0].content == "Dense only"
        assert fused_sparse[0].content == "Sparse only"

    def test_fusion_handles_empty_inputs(self):
        """Gracefully handles when one source returns nothing."""
        retriever = HybridRetriever.__new__(HybridRetriever)

        dense = [RetrievedDocument(content="Only dense", metadata={}, score=0.9, source="dense")]

        fused = retriever._rrf_fuse(dense, [], alpha=0.6)
        assert len(fused) == 1
        assert fused[0].content == "Only dense"

    def test_fusion_deduplicates(self):
        """Same content from both sources is merged, not duplicated."""
        retriever = HybridRetriever.__new__(HybridRetriever)

        same_content = "This is the same document appearing in both"
        dense = [RetrievedDocument(content=same_content, metadata={}, score=0.9, source="dense")]
        sparse = [RetrievedDocument(content=same_content, metadata={}, score=5.0, source="sparse")]

        fused = retriever._rrf_fuse(dense, sparse, alpha=0.5)
        assert len(fused) == 1
        assert fused[0].score > dense[0].score  # Boosted by appearing in both

    def test_fusion_assigns_ranks(self):
        """Output documents have sequential ranks starting at 1."""
        retriever = HybridRetriever.__new__(HybridRetriever)

        dense = [
            RetrievedDocument(content=f"Doc {i}", metadata={}, score=0.9-i*0.1, source="dense")
            for i in range(5)
        ]

        fused = retriever._rrf_fuse(dense, [], alpha=0.6)
        ranks = [d.rank for d in fused]
        assert ranks == list(range(1, len(fused) + 1))


if __name__ == "__main__":
    t = TestRRFFusion()
    t.test_fusion_combines_both_sources()
    t.test_fusion_alpha_weighting()
    t.test_fusion_handles_empty_inputs()
    t.test_fusion_deduplicates()
    t.test_fusion_assigns_ranks()
    print(" All RRF fusion tests passed")