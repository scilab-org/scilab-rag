"""
API request schemas.
"""


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
    
    paperId: str = Field(..., description="ID of the uploaded PDF document")
    paperName: str = Field(..., description="Original filename of the PDF")
    parsedText: str = Field(..., description="JSON string of parsed chunks from the PDF (output of /papers/parse)")


class ChatRequest(BaseModel):
    """Request to chat with the Knowledge Graph."""

    message: str = Field(..., description="The question to ask", min_length=1)
    projectId: str = Field(..., description="Project ID to scope the query to")


