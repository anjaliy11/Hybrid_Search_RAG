"""
LLM Provider Factory.

Supports free-tier providers:
  1. Google Gemini (15 RPM, 1M tokens/day)
  2. Groq (30 RPM, llama-3.1-70b)

Single responsibility: return a configured LLM instance.
All other modules import from here.
"""




import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from config.settings import settings

logger = logging.getLogger(__name__)


def get_llm(
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Get primary LLM from first available free provider.
    Priority: Gemini → Groq → RuntimeError
    """
    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens or settings.max_tokens

    # Google Gemini
    if settings.has_google:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model or settings.llm_model,
                temperature=temp,
                max_output_tokens=tokens,
                google_api_key=settings.google_api_key,
            )
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")

    # Groq
    if settings.has_groq:
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=model or settings.groq_model,
                temperature=temp,
                max_tokens=tokens,
                groq_api_key=settings.groq_api_key,
            )
        except Exception as e:
            logger.warning(f"Groq failed: {e}")

    raise RuntimeError(
        "No LLM configured. Set GOOGLE_API_KEY or GROQ_API_KEY in .env\n"
        "  Google: https://aistudio.google.com/apikey\n"
        "  Groq:   https://console.groq.com/keys"
    )


def get_fast_llm() -> BaseChatModel:
    """
    Lighter/faster model for sub-tasks (evaluation, routing).
    Uses Groq 8B (very fast) or falls back to primary.
    """
    if settings.has_groq:
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=settings.fast_model,
                temperature=0.0,
                max_tokens=1024,
                groq_api_key=settings.groq_api_key,
            )
        except Exception:
            pass

    return get_llm(temperature=0.0)