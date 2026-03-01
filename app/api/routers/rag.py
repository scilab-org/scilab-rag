import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.api_models.request import IngestRequest, ChatRequest
from app.api.api_models.response import PdfUploadResponse, IngestResponse, ChatResponse
from app.core.config import settings
from app.core.dependencies import get_graph_store, get_llm
from app.core.prompts import KG_TRIPLET_EXTRACT_TMPL
from app.services.utils import parse_fn


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Graph RAG"])

# In-memory storage for uploaded PDFs (maps document_id to temp file path)
_uploaded_documents: dict[str, str] = {}


def _compute_document_id(content: bytes) -> str:
    """Compute a unique document ID based on content hash."""
    return hashlib.sha256(content).hexdigest()[:16]


@router.post("/pdf", response_model=PdfUploadResponse)
async def upsert_pdf(
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
):
    """
    Upsert a PDF file.

    Upload a PDF document for processing. If document_id is provided,
    it will replace an existing document with that ID.

    The file is stored in a temporary location and will be cleaned up
    after processing or on service restart.

    Args:
        file: PDF file to upload
        document_id: Optional ID to upsert (replace existing)

    Returns:
        Document ID and status
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()

    if document_id is None:
        document_id = _compute_document_id(content)

    if document_id in _uploaded_documents:
        old_path = Path(_uploaded_documents[document_id])
        if old_path.exists():
            old_path.unlink()

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
        prefix=f"scilab_{document_id}_",
    )
    temp_file.write(content)
    temp_file.close()

    _uploaded_documents[document_id] = temp_file.name

    return PdfUploadResponse(
        documentId=document_id,
        filename=file.filename,
        sizeBytes=len(content),
        status="uploaded",
        message="PDF uploaded successfully. Call /ingest to process into Knowledge Graph.",
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_to_kg(request: IngestRequest):
    """
    Ingest and embed a PDF document into the Knowledge Graph.

    This will:
    1. Parse the PDF using Docling
    2. Split into chunks
    3. Extract entities and relationships
    4. Store in Neo4j Knowledge Graph
    5. Detect communities
    6. Generate community summaries

    Args:
        document_id: ID of the uploaded PDF
        do_picture_description: Enable AI image descriptions
        do_formula_enrichment: Enable formula extraction

    Returns:
        Ingestion statistics
    """
    document_id = request.documentId

    if document_id not in _uploaded_documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found. Upload it first via /pdf endpoint.",
        )

    file_path = _uploaded_documents[document_id]

    if not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Document file not found. Please re-upload.",
        )

    try:
        from llama_index.core import Document, PropertyGraphIndex
        from llama_index.core.node_parser import SentenceSplitter

        from app.services.document_parser import parse_document
        from app.services.extractor import GraphRAGExtractor

        llm = get_llm()
        graph_store = get_graph_store()

        parsed_text = parse_document(
            file_path,
        )

        splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        
        documents = [Document(text=parsed_text)]
        nodes = splitter.get_nodes_from_documents(documents)

        extractor = GraphRAGExtractor(
            llm=llm,
            max_path_per_chunks=request.maxTripletsPerChunk or settings.MAX_TRIPLETS_PER_CHUNK,
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
            parse_fn=parse_fn,
        )

        nodes_with_kg = extractor(nodes, show_progress=True)
        logger.info("Extracted %d nodes with KG info", len(nodes_with_kg))
        index = PropertyGraphIndex(
            nodes=nodes_with_kg,
            property_graph_store=graph_store,
            embed_model=None,
            show_progress=True,
            kg_extractors=[],
        )

        await index.property_graph_store.build_communities()

        triplets = graph_store.get_triplets()
        community_count = len(graph_store.community_summary)

        Path(file_path).unlink(missing_ok=True)
        del _uploaded_documents[document_id]

        return IngestResponse(
            documentId=document_id,
            status="success",
            chunkCount=len(nodes),
            tripletCount=len(triplets),
            communityCount=community_count,
            message="Document successfully ingested into Knowledge Graph",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the Knowledge Graph.
    """
    logger.info("Received chat request")

    try:
        from llama_index.core import PropertyGraphIndex
        from app.services.query_engine import GraphRAGQueryEngine

        logger.debug("Getting dependencies for chat request")
        graph_store = get_graph_store()
        llm = get_llm()

        logger.debug("Checking for existing community summaries")
        if not graph_store.community_summary:
            logger.info("No community summaries found, building communities")
            await graph_store.build_communities()

        logger.debug(
            "Community summary count=%d, entity info count=%d",
            len(graph_store.community_summary),
            len(graph_store.entity_info),
        )

        logger.debug("Loading property graph index")
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            embed_model=None,
        )

        logger.debug("Creating GraphRAGQueryEngine")
        query_engine = GraphRAGQueryEngine(
            graph_store=graph_store,
            index=index,
            llm=llm,
            similarity_top_k=request.similarityTopK or settings.SIMILARITY_TOP_K,
        )
        logger.debug("Calling acustom_query on query engine")
        answer = await query_engine.acustom_query(request.message)
        logger.debug("Got answer (truncated to 100 chars): %s", answer[:100])

        response = ChatResponse(
            message=request.message,
            answer=answer,
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.get("/status")
async def get_status():
    """
    Get the current status of the Knowledge Graph.

    Returns:
        Statistics about the knowledge graph
    """
    try:
        graph_store = get_graph_store()
        triplets = graph_store.get_triplets()

        return {
            "status": "ready",
            "pending_documents": len(_uploaded_documents),
            "triplet_count": len(triplets),
            "community_count": len(graph_store.community_summary),
            "has_data": len(triplets) > 0,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
