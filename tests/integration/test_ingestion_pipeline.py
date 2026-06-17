"""Integration test for the full ingestion pipeline."""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.pipeline import IngestionPipeline


class TestIngestionPipeline:
    """Tests for end-to-end ingestion."""

    def setup_method(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        self._create_test_documents()

    def _create_test_documents(self):
        """Create sample documents for testing."""
        docs = [
            {
                "title": "Test Document 1",
                "content": "Retrieval augmented generation combines retrieval systems with generative models. "
                           "This approach reduces hallucinations by grounding outputs in factual documents. "
                           "RAG systems typically use vector databases for semantic search. " * 5,
                "metadata": {"source": "test_doc_1", "type": "wiki"},
            },
            {
                "title": "Test Document 2",
                "content": "Large language models are trained on vast corpora of text data. "
                           "They use transformer architectures with self-attention mechanisms. "
                           "Fine-tuning adapts pre-trained models to specific tasks. " * 5,
                "metadata": {"source": "test_doc_2", "type": "wiki"},
            },
            {
                "title": "Test Document 3",
                "content": "BM25 is a bag-of-words retrieval function that ranks documents based on term frequency. "
                           "It improves upon TF-IDF by incorporating document length normalization. "
                           "The algorithm is widely used in search engines. " * 5,
                "metadata": {"source": "test_doc_3", "type": "wiki"},
            },
        ]

        for i, doc in enumerate(docs):
            path = Path(self.temp_dir) / f"doc_{i}.json"
            with open(path, "w") as f:
                json.dump(doc, f)

    def test_pipeline_runs_without_error(self):
        """Pipeline completes successfully."""
        pipeline = IngestionPipeline()
        result = pipeline.run(self.temp_dir)

        assert result["documents_loaded"] == 3
        assert result["chunks_created"] > 0

    def test_chunks_are_indexed_in_chromadb(self):
        """Chunks are accessible in ChromaDB after ingestion."""
        pipeline = IngestionPipeline()
        result = pipeline.run(self.temp_dir)

        count = pipeline.collection.count()
        assert count == result["chunks_created"]

    def test_bm25_index_created(self):
        """BM25 index file is created on disk."""
        from config.settings import settings

        pipeline = IngestionPipeline()
        pipeline.run(self.temp_dir)

        assert Path(settings.bm25_index_path).exists()

    def test_empty_directory_handled(self):
        """Empty directory returns zeros without error."""
        empty_dir = tempfile.mkdtemp()
        pipeline = IngestionPipeline()
        result = pipeline.run(empty_dir)

        assert result["documents_loaded"] == 0
        assert result["chunks_created"] == 0


if __name__ == "__main__":
    t = TestIngestionPipeline()
    t.setup_method()
    t.test_pipeline_runs_without_error()
    print(" Pipeline runs")
    t.setup_method()
    t.test_chunks_are_indexed_in_chromadb()
    print(" ChromaDB indexed")
    t.setup_method()
    t.test_bm25_index_created()
    print(" BM25 created")
    t.test_empty_directory_handled()
    print(" Empty dir handled")
    print("\n All ingestion integration tests passed")