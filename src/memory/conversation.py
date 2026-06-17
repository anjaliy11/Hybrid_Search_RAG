"""
Conversation Memory — sliding window with context management.
Tracks multi-turn dialogue for coreference resolution and follow-ups.
"""

from typing import List
from dataclasses import dataclass
from collections import deque
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


@dataclass
class Turn:
    query: str
    response: str
    score: float = 0.0


class ConversationMemory:
    """
    Sliding window memory for multi-turn conversations.
    Keeps last N turns for context injection into agents.
    """

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self.turns: deque = deque(maxlen=window_size)

    def add_turn(self, query: str, response: str, score: float = 0.0):
        self.turns.append(Turn(query=query, response=response, score=score))

    def get_messages(self, last_n: int = 4) -> List[BaseMessage]:
        """Get recent turns as LangChain messages."""
        messages = []
        for turn in list(self.turns)[-last_n:]:
            messages.append(HumanMessage(content=turn.query))
            messages.append(AIMessage(content=turn.response[:500]))
        return messages

    def get_context(self, last_n: int = 3) -> str:
        """Get recent turns as formatted string."""
        if not self.turns:
            return ""
        lines = []
        for t in list(self.turns)[-last_n:]:
            lines.append(f"User: {t.query}")
            lines.append(f"Assistant: {t.response[:200]}")
        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def clear(self):
        self.turns.clear()