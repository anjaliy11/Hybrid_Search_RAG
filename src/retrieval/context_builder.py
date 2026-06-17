"""
Context Builder — assembles retrieved documents into a formatted
context window for the LLM, with token budget management.
"""

from typing import List
from src.retrieval.hybrid_retriever import RetrievedDocument
from config.settings import settings


class ContextBuilder:
    """Assembles and formats context from retrieved documents."""

    def __init__(self, max_chars: Optional[int] = None):
        # Rough approximation: 1 token ≈ 4 chars
        self.max_chars = max_chars or (settings.max_context_tokens * 4)

    def build(self, documents: List[RetrievedDocument]) -> str:
        """
        Build formatted context string within token budget.
        
        Format:
          [Doc 1] (source: xyz) — Title
          
          
          ---
          
          [Doc 2] ...
        """
        parts = []
        total_chars = 0

        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "")

            header = f"[Doc {i+1}] (source: {source})"
            if title and title != source:
                header += f" — {title}"

            entry = f"{header}\n{doc.content}"

            if total_chars + len(entry) > self.max_chars:
                # Truncate last document to fit
                remaining = self.max_chars - total_chars - len(header) - 10
                if remaining > 100:
                    entry = f"{header}\n{doc.content[:remaining]}..."
                    parts.append(entry)
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n\n---\n\n".join(parts)

    def build_from_dicts(self, documents: List[dict]) -> str:
        """Build context from serialized document dicts (from agent state)."""
        retrieved = [
            RetrievedDocument(
                content=d["content"],
                metadata=d.get("metadata", {}),
                score=d.get("score", 0.0),
                source="state",
                rank=d.get("rank", 0),
            )
            for d in documents
        ]
        return self.build(retrieved)


# Fix missing import
from typing import Optional