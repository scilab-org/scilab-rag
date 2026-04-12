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
    mode: str = Field(default="chat", pattern=r"^(chat|write)$", description="'chat' for Q&A, 'write' for paper writing.")
    section_id: Optional[str] = Field(default=None, description="UUID from .NET backend — scopes session to a section (both chat and write modes).")
    section_target: Optional[str] = Field(default=None, description="Target section type, e.g. 'methodology', 'results' (both chat and write modes).")
    writing: Optional["WritingPayload"] = Field(default=None, description="Required when mode='write'.")


class ReferencedSection(CamelCaseModel):
    """A section the user attached for cross-reference context (like Copilot @file)."""
    section_type: str = Field(..., description="e.g. 'introduction', 'methodology', 'results'")
    content: str = Field(..., min_length=1, description="LaTeX content of the section")


class WritingPayload(CamelCaseModel):
    """Payload sent alongside a write-mode request."""
    current_section: Optional[str] = Field(default=None, description="LaTeX content of the section being worked on (can be non-null even for write_new).")
    referenced_sections: Optional[list[ReferencedSection]] = Field(default=None, description="Other sections user attached for cross-reference.")
    ruleset: Optional[str] = Field(default=None, description="Style/formatting rules as markdown")

class SessionRenameRequest(CamelCaseModel):
    title: str = Field(..., min_length=1, max_length=255)


# Resolve forward reference: ChatRequest.writing -> WritingPayload
ChatRequest.model_rebuild()
