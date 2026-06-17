"""Health and readiness endpoint."""

from fastapi import APIRouter
from pathlib import Path
import chromadb

from config.settings import settings
from src.serving.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Service health with index status."""
    indices = {}

    try:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        col = client.get_or_create_collection(name=settings.collection_name)
        indices["chromadb"] = {"status": "ok", "vectors": col.count()}
    except Exception as e:
        indices["chromadb"] = {"status": "error", "detail": str(e)}

    indices["bm25"] = {
        "status": "ok" if Path(settings.bm25_index_path).exists() else "missing"
    }

    return HealthResponse(status="healthy", version=settings.version, indices=indices)