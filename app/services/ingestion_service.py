"""Core ingestion logic shared by the HTTP endpoint and the RabbitMQ consumer.

Extracting this into a standalone service avoids duplicating the heavy
KG-ingestion pipeline in two places.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Outcome of a single ingestion attempt."""

    success: bool
    message: str
    error: Optional[str] = None


async def ingest_paper_to_kg(
    paper_id: str,
    paper_name: str,
    parsed_text: str,
) -> IngestionResult:
    """Run the full knowledge-graph ingestion pipeline for one paper.

    Parameters
    ----------
    paper_id:
        Unique identifier for the paper (comes from the .NET PaperBank).
    paper_name:
        Human-readable paper title / filename.
    parsed_text:
        JSON string produced by ``HybridChunker`` during the parse step.
        Expected shape: ``{"chunks": [{"text": "...", "headings": [...], "captions": [...]}]}``.

    Returns
    -------
    IngestionResult
        Contains *success* flag and a human-readable *message* (or *error*).
    """
    try:
        from llama_index.core import PropertyGraphIndex
        from llama_index.core.schema import BaseNode, TextNode

        from app.agents.ingest.extractor import GraphRAGExtractor
        from app.agents.ingest.prompts import KG_TRIPLET_EXTRACT_TMPL
        from app.core.dependencies import get_embed_llm, get_extract_llm, get_graph_store
        from app.domain.models import PaperInfo
        from app.helpers.utils import parse_fn

        paper_info = PaperInfo(
            paper_id=paper_id,
            paper_name=paper_name,
        )

        extract_llm = get_extract_llm()
        graph_store = get_graph_store()
        embed_model = get_embed_llm()
        unescaped_text = parsed_text.replace('\\"', '"').replace('\\\\','\\')
        
        parsed = json.loads(unescaped_text)
            
        nodes: list[BaseNode] = [
            TextNode(
                text=chunk["text"],
                metadata={
                    "headings": chunk.get("headings") or [],
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

        graph_store.create_same_as_links(paper_id)

        logger.info("Ingestion succeeded for paper %s (%s)", paper_id, paper_name)
        return IngestionResult(
            success=True,
            message="Document successfully ingested into Knowledge Graph",
        )

    except Exception as exc:
        logger.exception("Ingestion failed for paper %s (%s)", paper_id, paper_name)
        return IngestionResult(
            success=False,
            message="Ingestion failed",
            error=str(exc),
        )
