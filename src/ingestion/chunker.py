"""
Document Chunker — splits documents into retrieval-optimized chunks.

Uses RecursiveCharacterTextSplitter which respects natural boundaries
(paragraphs, sentences) while maintaining configurable chunk size.
"""

import hashlib
from typing import List
from dataclasses import dataclass, field

from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """Single text chunk with metadata and deterministic ID."""
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            # Deterministic ID from content — idempotent ingestion
            self.chunk_id = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()[:16]


class DocumentChunker:
    """
    Splits documents into overlapping chunks for retrieval.
    
    Design choices:
      - 512 chars ≈ 100 tokens — fits well in context windows
      - 64 char overlap — maintains sentence continuity across chunks
      - Separators respect paragraph and sentence boundaries
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: List[dict]) -> List[Chunk]:
        """
        Chunk all documents with metadata propagation.
        
        Each chunk inherits parent document metadata plus:
          - chunk_index: position within parent
          - total_chunks: total chunks from parent
        """
        chunks: List[Chunk] = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            if len(content.strip()) < 80:
                continue

            splits = self.splitter.split_text(content)

            for idx, text in enumerate(splits):
                text = text.strip()
                if len(text) < 30:
                    continue

                chunk_meta = {
                    **metadata,
                    "chunk_index": idx,
                    "total_chunks": len(splits),
                }

                chunks.append(Chunk(content=text, metadata=chunk_meta))

        return chunks