"""
FastAPI Application — production-ready Agentic RAG service.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.serving.middleware import LatencyMiddleware
from src.serving.routes import health, chat, ingest, evaluate

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 {settings.project_name} v{settings.version} starting")
    yield
    logger.info("👋 Shutting down")


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="Multi-turn agentic RAG with hybrid retrieval, hallucination detection, and self-evaluation.",
    lifespan=lifespan,
)

app.add_middleware(LatencyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(evaluate.router, prefix="/api/v1", tags=["Evaluation"])