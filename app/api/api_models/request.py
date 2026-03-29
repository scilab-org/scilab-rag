"""
API request schemas.

All fields are snake_case internally; the alias_generator on CamelCaseModel
automatically exposes them as camelCase in the JSON wire format.
"""
import uuid
from typing import Optional

from pydantic import Field

from app.api.api_models import CamelCaseModel

class PaperAutoTagRequest(CamelCaseModel):
    parsed_text: str = Field(
        ...,
        min_length=100,
        description="The extracted text from the PDF",
    )
    existing_tags: list[str] | None = Field(
        default=None,
        description="Existing tags for the paper (optional)",
    )

class IngestRequest(CamelCaseModel):
    """Request to ingest a document into the Knowledge Graph."""

    paper_id: str = Field(..., description="ID of the uploaded PDF document")
    paper_name: str = Field(..., description="Original filename of the PDF")
    parsed_text: str = Field(
        ...,
        description="JSON string of parsed chunks from the PDF (output of /papers/parse)",
    )

class ChatRequest(CamelCaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[uuid.UUID] = None
    project_id: Optional[str] = None
    paper_ids: list[str] = Field(default_factory=list, description="Paper IDs to scope the query. Empty list queries without paper filtering.")

class SessionRenameRequest(CamelCaseModel):
    title: str = Field(..., min_length=1, max_length=255)
