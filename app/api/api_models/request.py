"""
API request schemas.
"""

from typing import Optional

from pydantic import BaseModel, Field


class PaperAutoTagRequest(BaseModel):   
    parsedText: str = Field(
        ...,
        min_length=100,
        description="The extracted text from the PDF"
    )
    
    existingTags: list[str] | None = Field(
        description="Existing tags for the paper (optional)",
        default=None,
    ) 

class IngestRequest(BaseModel):
    """Request to ingest a document into the Knowledge Graph."""
    
    documentId: str = Field(..., description="ID of the uploaded PDF document")
    doPictureDescription: bool = Field(
        default=False,
        description="Enable AI-generated descriptions for images in the PDF",
    )
    doFormulaEnrichment: bool = Field(
        default=False,
        description="Enable formula enrichment for mathematical expressions",
    )
    maxTripletsPerChunk: Optional[int] = Field(
        default=None,
        description="Maximum entity-relation triplets to extract per chunk",
    )


class ChatRequest(BaseModel):
    """Request to chat with the Knowledge Graph."""

    message: str = Field(..., description="The question to ask", min_length=1)
    similarityTopK: Optional[int] = Field(
        default=None,
        description="Number of similar entities to retrieve",
    )


class SystemInfoRequest(BaseModel):
    """Request to create or update system info."""

    key: str = Field(..., description="Info key", min_length=1, max_length=255)
    value: Optional[str] = Field(None, description="Info value")
