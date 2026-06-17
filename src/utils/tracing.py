"""
Observability — LangSmith tracing integration.
Enables when LANGSMITH_API_KEY is set.
"""

import os
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


def setup_tracing():
    """Configure LangSmith tracing if API key available."""
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info(f"LangSmith tracing enabled: {settings.langsmith_project}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"