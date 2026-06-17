"""
Voice Agent — reformats answers for TTS/voice delivery.
Handles pronunciation, abbreviation expansion, and natural speech patterns.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_llm

logger = logging.getLogger(__name__)

VOICE_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """Reformat this answer for voice/spoken delivery:

Rules:
1. Spell out abbreviations first time: "RAG" → "R-A-G, Retrieval Augmented Generation"
2. Use short sentences (max 15-20 words each)
3. Replace citation brackets with natural phrases: "[Doc 1]" → "According to the source document"
4. Use transitions: "First...", "Additionally...", "In summary..."
5. Avoid parenthetical asides
6. Spell out numbers and special characters
7. Make it conversational but informative

Original: {answer}

Voice-optimized version:"""),
])


class VoiceAgent:
    """Formats answers for voice/TTS delivery."""

    def __init__(self):
        self.llm = get_fast_llm()
        self.chain = VOICE_PROMPT | self.llm

    def format_for_voice(self, answer: str) -> str:
        """Convert text answer to voice-optimized format."""
        try:
            response = self.chain.invoke({"answer": answer})
            return response.content
        except Exception as e:
            logger.warning(f"Voice formatting failed: {e}")
            return answer