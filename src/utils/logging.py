"""Structured logging configuration."""

import logging
import sys


def setup_logging(level: str = "INFO"):
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress noisy loggers
    for name in ("httpx", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)