from fastapi import APIRouter, HTTPException
from fastapi.logger import logger

from app.api.api_models.request import IngestRequest
from app.api.api_models.response import IngestResponse
from app.services.ingestion_service import ingest_paper_to_kg

router = APIRouter(
    prefix="/graph",
    tags=["Knowledge Graph"],
)

@router.post("/ingest", response_model=IngestResponse)
async def ingest_to_kg(request: IngestRequest):
    """
    Ingest and embed a PDF document into the Knowledge Graph.

    This will:
    1. Split into chunks
    2. Extract entities and relationships
    3. Store in Neo4j Knowledge Graph
    4. Create SAME_AS links across papers

    Returns:
        Ingestion statistics
    """

    result = await ingest_paper_to_kg(
        paper_id=request.paper_id,
        paper_name=request.paper_name,
        parsed_text=request.parsed_text,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {result.error}")

    return IngestResponse(
        paper_id=request.paper_id,
        status="success",
        message=result.message,
    )