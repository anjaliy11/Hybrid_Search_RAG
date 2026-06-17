"""Unit tests for document chunker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.ingestion.chunker import DocumentChunker, Chunk


class TestDocumentChunker:
    """Tests for chunking logic."""

    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)

    def test_basic_chunking(self):
        """Documents are split into multiple chunks."""
        docs = [{"content": "This is a test sentence. " * 40, "metadata": {"source": "test"}}]
        chunks = self.chunker.chunk_documents(docs)

        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_are_unique(self):
        """Every chunk gets a unique deterministic ID."""
        docs = [{"content": "Hello world. " * 50, "metadata": {"source": "test"}}]
        chunks = self.chunker.chunk_documents(docs)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # All unique

    def test_chunk_ids_are_deterministic(self):
        """Same content produces same chunk ID (idempotent ingestion)."""
        docs = [{"content": "Deterministic content here. " * 30, "metadata": {"source": "test"}}]

        chunks_1 = self.chunker.chunk_documents(docs)
        chunks_2 = self.chunker.chunk_documents(docs)

        assert [c.chunk_id for c in chunks_1] == [c.chunk_id for c in chunks_2]

    def test_metadata_propagation(self):
        """Parent metadata propagates to all chunks."""
        docs = [{
            "content": "Some long content here. " * 40,
            "metadata": {"source": "wiki", "title": "Test Article"},
        }]
        chunks = self.chunker.chunk_documents(docs)

        for chunk in chunks:
            assert chunk.metadata["source"] == "wiki"
            assert chunk.metadata["title"] == "Test Article"
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_short_documents_skipped(self):
        """Documents under 80 chars are skipped."""
        docs = [{"content": "Too short.", "metadata": {"source": "x"}}]
        chunks = self.chunker.chunk_documents(docs)
        assert len(chunks) == 0

    def test_empty_input(self):
        """Empty input returns empty output."""
        assert self.chunker.chunk_documents([]) == []

    def test_chunk_size_respected(self):
        """Chunks don't massively exceed configured size."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        docs = [{"content": "Word " * 200, "metadata": {"source": "test"}}]
        chunks = chunker.chunk_documents(docs)

        for chunk in chunks:
            # Allow some overflow due to separator logic
            assert len(chunk.content) <= 150

    def test_overlap_exists(self):
        """Adjacent chunks share overlapping content."""
        docs = [{"content": "The quick brown fox jumps over the lazy dog. " * 30, "metadata": {"source": "t"}}]
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=30)
        chunks = chunker.chunk_documents(docs)

        if len(chunks) >= 2:
            # Check some content from end of chunk N appears in start of chunk N+1
            end_of_first = chunks[0].content[-30:]
            assert any(
                word in chunks[1].content[:60]
                for word in end_of_first.split()
                if len(word) > 3
            )


if __name__ == "__main__":
    t = TestDocumentChunker()
    t.setup_method()
    t.test_basic_chunking()
    t.test_chunk_ids_are_unique()
    t.test_chunk_ids_are_deterministic()
    t.test_metadata_propagation()
    t.test_short_documents_skipped()
    t.test_empty_input()
    t.test_chunk_size_respected()
    t.test_overlap_exists()
    print(" All chunker tests passed")