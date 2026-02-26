import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.api_models.request import PaperAutoTagRequest
from app.api.api_models.response import PaperParseResponse, PaperAutoTagResponse
from app.core.config import settings
from app.core.dependencies import get_llm
from app.services.auto_tagger import AutoTagger


router = APIRouter(
    prefix="/papers",
    tags=["Papers"],
)


@router.post("/parse", response_model=PaperParseResponse)
async def parse_paper(file: UploadFile = File(...)):
    content = await file.read()
    if not content.startswith(b"%PDF-"):
        raise HTTPException(400, "Invalid PDF file")

    try:
        tmp_path = None
       
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        from app.services.document_parser import parse_document

        text = parse_document(
            tmp_path,
        )
      
        return PaperParseResponse(
            parsedText=text,
        )

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
    text = request.parsedText
    existing_tags = request.existingTags or []
    llm = get_llm()

    try:
        from llama_index.core import Document
        from llama_index.core.node_parser import SentenceSplitter

        text = text.strip()
        splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        documents = [Document(text=text)]
        nodes = splitter.get_nodes_from_documents(documents)

        tagger = AutoTagger(llm=llm, existing_tags=existing_tags)
        nodes_with_tags = tagger(nodes, show_progress=True)

        tags = nodes_with_tags[0].metadata.get("tags", []) if nodes_with_tags else []

        return PaperAutoTagResponse(
            tags=tags,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to auto-tag document: {str(e)}",
        )
