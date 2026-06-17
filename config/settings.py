"""
Centralized configuration — single source of truth for all settings.
Uses Pydantic v2 Settings with .env file loading and validation.
"""

from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Project ──
    project_name: str = "Agentic RAG"
    version: str = "1.0.0"
    debug: bool = False

    # ── LLM Providers (all free-tier) ──
    google_api_key: str = ""
    groq_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    groq_model: str = "llama-3.1-70b-versatile"
    fast_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.1
    max_tokens: int = 2048

    # ── Embeddings (local — no API key) ──
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Vector Store (ChromaDB local) ──
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "agentic_rag"

    # ── Sparse Index ──
    bm25_index_path: str = "./data/processed/bm25_index.pkl"

    # ── Retrieval ──
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    hybrid_alpha: float = 0.6  # 1.0 = all dense, 0.0 = all sparse

    # ── Agent ──
    max_agent_iterations: int = 5
    confidence_threshold: float = 0.65
    max_context_tokens: int = 6000

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # ── Observability ──
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "agentic-rag"
    log_level: str = "INFO"

    # ── Paths ──
    data_dir: Path = Path("./data")
    raw_dir: Path = Path("./data/raw")
    eval_dir: Path = Path("./data/eval_sets")

    @property
    def has_google(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()