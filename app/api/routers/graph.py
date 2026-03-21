import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.logger import logger

from app.api.api_models.request import ChatRequest, IngestRequest
from app.api.api_models.response import ChatResponse, IngestResponse
from app.core.dependencies import get_embed_llm, get_extract_llm, get_chat_llm
from app.core.dependencies import get_graph_store

from app.agents.ingest.prompts import KG_TRIPLET_EXTRACT_TMPL
from app.domain.models import PaperInfo
from app.helpers.utils import parse_fn

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

    try:
        from llama_index.core import PropertyGraphIndex
        from llama_index.core.schema import BaseNode, TextNode

        from app.agents.ingest.extractor import GraphRAGExtractor

        paper_info = PaperInfo(
            paper_id=request.paper_id,
            paper_name=request.paper_name,
        )
        
        # Use extract LLM for entity/relationship extraction
        extract_llm = get_extract_llm()
        graph_store = get_graph_store()
        embed_model = get_embed_llm()

        # parsed_text is a JSON string produced by HybridChunker in the parse step
        parsed = json.loads(request.parsed_text)
        nodes: list[BaseNode] = [
            TextNode(
                text=chunk["text"],
                metadata={
                    "headings": chunk.get("headings") or [],
                    "captions": chunk.get("captions") or [],
                },
            )
            for chunk in parsed["chunks"]
            if chunk.get("text", "").strip()
        ]

        extractor = GraphRAGExtractor(
            llm=extract_llm,
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
            parse_fn=parse_fn,
            paper_info=paper_info,
        )

        nodes_with_kg = extractor(nodes, show_progress=True)
        index = await asyncio.to_thread(
            PropertyGraphIndex,
            nodes=nodes_with_kg,
            property_graph_store=graph_store,
            embed_model=embed_model,
            show_progress=True,
            llm=extract_llm,
            kg_extractors=[],
        )

        graph_store.create_same_as_links(request.paper_id)

        return IngestResponse(
            paper_id=request.paper_id,
            status="success",
            message="Document successfully ingested into Knowledge Graph",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")