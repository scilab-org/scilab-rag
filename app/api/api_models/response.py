"""
API response schemas.

All fields are snake_case internally; the alias_generator on CamelCaseModel
automatically serialises them as camelCase in the JSON wire format.

ORM models with snake_case column names are mapped transparently because
CamelCaseModel sets from_attributes=True and populate_by_name=True, so
model_validate() accepts both snake_case (ORM) and camelCase (JSON) keys.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from pydantic import Field

from app.api.api_models import CamelCaseModel

class PaperParseResponse(CamelCaseModel):
    parsed_text: str = Field(..., description="JSON string of parsed chunks from the PDF")

class PaperAutoTagResponse(CamelCaseModel):
    tags: List[str] = Field(default_factory=list, description="Auto-generated tags for the paper")

class PdfUploadResponse(CamelCaseModel):
    """Response after uploading a PDF."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")

class IngestResponse(CamelCaseModel):
    """Response after ingesting a document into the Knowledge Graph."""

    paper_id: str = Field(..., description="Document identifier")
    status: str = Field(..., description="Ingestion status")
    message: str = Field(..., description="Status message")

class ChatResponse(CamelCaseModel):
    """Response for a chat query."""

    message: str = Field(..., description="The original question")
    answer: str = Field(..., description="The AI-generated answer")
    
class MessageResponse(CamelCaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    msg_metadata: dict
    created_at: datetime

class SessionResponse(CamelCaseModel):
    id: uuid.UUID
    project_id: Optional[str] = None
    section_id: Optional[str] = None
    section_target: Optional[str] = None
    title: str
    context: dict
    created_at: datetime
    updated_at: datetime

class ChatMessageResponse(CamelCaseModel):
    session_id: uuid.UUID
    user_message: MessageResponse
    assistant_message: MessageResponse

class SessionListResponse(CamelCaseModel):
    sessions: list[SessionResponse]
    total: int

class MessageListResponse(CamelCaseModel):
    messages: list[MessageResponse]
    total: int

class SessionRenameResponse(CamelCaseModel):
    id: uuid.UUID
    title: str


# ---------------------------------------------------------------------------
# Writing-feature response models
# ---------------------------------------------------------------------------

class WritingOutput(CamelCaseModel):
    """Full LaTeX output for the target section — lives in metadata.writing_output."""
    section_target: str = Field(..., description="e.g. 'methodology', 'results'")
    content: str = Field(..., description="Complete LaTeX content of the section")


class ValidationSummary(CamelCaseModel):
    """Summary of the structural LaTeX validation step."""
    issues_found: int = Field(..., ge=0)
    issues_fixed: int = Field(..., ge=0)


class QuestionOption(CamelCaseModel):
    """A single option inside a select-type planning question."""
    label: str
    value: str


class PlanningQuestion(CamelCaseModel):
    """One structured question from the planning agent."""
    type: str = Field(..., description="single_select | multi_select | text")
    prompt: str = Field(..., description="The question text")
    options: Optional[List[QuestionOption]] = Field(default=None, description="Choices for select-type questions")
    allow_custom: bool = Field(default=True, description="Whether the user can type a custom answer")
