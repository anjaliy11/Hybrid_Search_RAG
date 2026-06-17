"""Document ingestion endpoint."""

import logging
from fastapi import APIRouter, HTTPException

from src.ingestion.pipeline import IngestionPipeline
from src.serving.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest documents from a directory."""
    try:
        pipeline = IngestionPipeline()
        result = pipeline.run(request.source_dir)
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))