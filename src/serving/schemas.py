"""Pydantic schemas for API request/response validation."""

from typing import Optional, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")
    stream: bool = Field(default=False)
    voice_mode: bool = Field(default=False, description="Format output for TTS")


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: float = 0.0
    hallucination_detected: bool = False
    iterations: int = 0
    session_id: str = ""


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    indices: dict = {}


class IngestRequest(BaseModel):
    source_dir: str


class IngestResponse(BaseModel):
    documents_loaded: int = 0
    chunks_created: int = 0
    status: str = "success"


class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str] = []
    ground_truth: str = ""


class EvalResponse(BaseModel):
    scores: dict = {}
    hallucination: dict = {}