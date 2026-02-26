"""
API response schemas.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PaperParseResponse(BaseModel):
    parsedText: str = Field(..., description="The extracted text from the PDF")
    
class PaperAutoTagResponse(BaseModel): 
    tags: List[str] = Field(default_factory=list, description="Auto-generated tags for the paper")    
    
class PdfUploadResponse(BaseModel):
    """Response after uploading a PDF."""
    
    documentId: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    sizeBytes: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")


class IngestResponse(BaseModel):
    """Response after ingesting a document into the Knowledge Graph."""
    
    documentId: str = Field(..., description="Document identifier")
    status: str = Field(..., description="Ingestion status")
    chunkCount: int = Field(..., description="Number of text chunks processed")
    tripletCount: int = Field(..., description="Number of entity-relation triplets extracted")
    communityCount: int = Field(..., description="Number of communities detected")
    message: str = Field(..., description="Status message")


class ChatResponse(BaseModel):
    """Response for a chat query."""

    message: str = Field(..., description="The original question")
    answer: str = Field(..., description="The AI-generated answer")


class DbStatusResponse(BaseModel):
    """Response for database connectivity check."""

    status: str = Field(..., description="Database status")


class SystemInfoResponse(BaseModel):
    """Response for system info entry."""

    key: str = Field(..., description="Info key")
    value: Optional[str] = Field(None, description="Info value")
    createdAt: datetime = Field(..., description="Creation timestamp")
    updatedAt: datetime = Field(..., description="Last update timestamp")
