import json
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from app.api.api_models.request import PaperAutoTagRequest
from app.api.api_models.response import PaperParseResponse, PaperAutoTagResponse
from app.core.config import settings
from app.core.dependencies import get_summary_llm
from app.agents.tagger.auto_tagger import AutoTagger


router = APIRouter(
    prefix="/papers",
    tags=["Papers"],
)


@router.post("/parse", response_model=PaperParseResponse)
async def parse_paper(file: UploadFile = File(...)):
    content = await file.read()
    if not content.startswith(b"%PDF-"):
        raise HTTPException(400, "Invalid PDF file")

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        from app.agents.ingest.document_parser import parse_document_per_batch
        import asyncio
        text = await asyncio.to_thread(parse_document_per_batch, tmp_path)

        return PaperParseResponse(parsed_text=json.dumps(text))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse document: {str(e)}",
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/auto-tag", response_model=PaperAutoTagResponse)
async def auto_tag_paper(request: PaperAutoTagRequest):
    existing_tags = request.existing_tags or []
    summary_llm = get_summary_llm()

    try:
        from llama_index.core.schema import TextNode

        # parsedText is a JSON string: {"chunks": [{"text": ..., "headings": ..., "captions": ...}]}
        parsed = json.loads(request.parsed_text)

        nodes = []
        for chunk in parsed.get("chunks", []):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            # Optionally prepend headings/captions to give the tagger more context
            headings = chunk.get("headings") or []
            captions = chunk.get("captions") or []

            enriched_parts = []
            if headings:
                enriched_parts.append(" > ".join(headings))
            enriched_parts.append(text)
            if captions:
                enriched_parts.extend(captions)

            nodes.append(TextNode(text="\n".join(enriched_parts)))

        if not nodes:
            return PaperAutoTagResponse(tags=[])

        tagger = AutoTagger(llm=summary_llm, existing_tags=existing_tags)
        nodes_with_tags = tagger(nodes, show_progress=True)

        tags = nodes_with_tags[0].metadata.get("tags", []) if nodes_with_tags else []

        return PaperAutoTagResponse(tags=tags)

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"parsedText is not valid JSON: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to auto-tag document: {str(e)}",
        )
