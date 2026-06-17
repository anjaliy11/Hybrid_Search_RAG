"""
Ingestion Pipeline — end-to-end: load → chunk → embed → index.

Creates two indices:
  1. Dense (ChromaDB) — semantic similarity via embeddings
  2. Sparse (BM25) — keyword matching via term frequency

Both are local and free.
"""

import pickle
import logging
from typing import List
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from config.settings import settings
from src.utils import get_embeddings
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker, Chunk

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates document ingestion into retrieval indices."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)
        self.embeddings = get_embeddings()

        # Initialize ChromaDB
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def run(self, source_dir: str) -> dict:
        """
        Execute full pipeline for a source directory.
        
        Returns:
            {"documents_loaded": int, "chunks_created": int}
        """
        # Load
        documents = self.loader.load_directory(source_dir)
        if not documents:
            return {"documents_loaded": 0, "chunks_created": 0}

        # Chunk
        chunks = self.chunker.chunk_documents(documents)
        logger.info(f"Chunked {len(documents)} docs → {len(chunks)} chunks")

        if not chunks:
            return {"documents_loaded": len(documents), "chunks_created": 0}

        # Index dense
        self._index_dense(chunks)

        # Index sparse
        self._index_sparse(chunks)

        return {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
        }

    def _index_dense(self, chunks: List[Chunk], batch_size: int = 32):
        """Embed and upsert chunks into ChromaDB."""
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]

            ids = [c.chunk_id for c in batch]
            texts = [c.content for c in batch]
            metas = [c.metadata for c in batch]

            embeddings = self.embeddings.embed_documents(texts)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metas,
            )

        logger.info(f"Dense index: {total} vectors in ChromaDB")

    def _index_sparse(self, chunks: List[Chunk]):
        """Build BM25 index and persist."""
        corpus = [c.content.lower().split() for c in chunks]
        bm25 = BM25Okapi(corpus)

        data = {"bm25": bm25, "chunks": chunks}

        path = Path(settings.bm25_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Sparse index: {len(chunks)} entries in BM25")