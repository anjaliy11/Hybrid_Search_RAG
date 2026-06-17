"""
Summary Buffer — compresses older conversation history into summaries.
Prevents context window overflow in long conversations.
"""

import logging
from typing import Optional
from src.utils import get_llm

logger = logging.getLogger(__name__)


class SummaryBuffer:
    """
    Maintains a running summary of conversation history.
    When the window fills, older turns are summarized and compressed.
    """

    def __init__(self, max_summary_length: int = 500):
        self.summary: str = ""
        self.max_length = max_summary_length
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_fast_llm()
        return self._llm

    def update(self, new_turns: str):
        """Add new conversation turns and update summary."""
        if not new_turns.strip():
            return

        combined = f"Previous summary: {self.summary}\n\nNew turns:\n{new_turns}" if self.summary else new_turns

        try:
            response = self.llm.invoke(
                f"Summarize this conversation in 2-3 sentences, "
                f"preserving key facts and entities:\n\n{combined}"
            )
            self.summary = response.content[:self.max_length]
        except Exception as e:
            logger.warning(f"Summary update failed: {e}")
            # Keep old summary
            pass

    def get_summary(self) -> str:
        return self.summary

    def clear(self):
        self.summary = ""