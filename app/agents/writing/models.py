"""
Shared domain types for the writing-feature agent pipeline.

Every agent (orchestrator, planning, writing, validation) imports from
here so there is a single source of truth for enums, decision objects,
and planning-state shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────

class WritingMode(str, Enum):
    """What the writing agent should do with the section."""
    WRITE_NEW = "write_new"
    EXTEND = "extend"
    REWRITE = "rewrite"
    FIX_CONTENT = "fix_content"
    FIX_LATEX = "fix_latex"


class ValidationScope(str, Enum):
    """How deeply the validation agent should inspect the output."""
    FULL = "full"
    CONTENT_ONLY = "content_only"
    SYNTAX_ONLY = "syntax_only"
    STYLE_ONLY = "style_only"


class PlanningStatus(str, Enum):
    """Status of the planning phase stored in session.context JSONB."""
    IDLE = "idle"
    ASKING = "asking"          # questions sent, awaiting answers
    COMPLETE = "complete"      # instructions built, ready to write


# ── Orchestrator decision ────────────────────────────────────────────────

@dataclass
class OrchestratorDecision:
    """
    The output of the orchestrator's LLM classification.
    Tells downstream agents exactly what to do for this request.
    """
    writing_mode: WritingMode
    validation_scope: ValidationScope
    invoke_planning: bool
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "writing_mode": self.writing_mode.value,
            "validation_scope": self.validation_scope.value,
            "invoke_planning": self.invoke_planning,
            "reasoning": self.reasoning,
        }


# ── Planning state (persisted in session.context JSONB) ──────────────────

@dataclass
class PlanningQuestion:
    """A single structured question from the planning agent."""
    type: str                             # single_select | multi_select | text
    prompt: str
    options: list[dict] = field(default_factory=list)  # [{label, value}, ...]
    allow_custom: bool = True


@dataclass
class PlanningState:
    """
    In-memory representation of session.context["planning_state"] JSONB.
    Serialised to/from dict for persistence.

    qa_rounds accumulates the full Q&A transcript across planning rounds.
    Each element is one round's complete exchange as sent by the FE —
    questions and answers together (e.g. "Q: ...\nA: ...\n\nQ: ...\nA: ...").
    """
    status: PlanningStatus = PlanningStatus.IDLE
    qa_rounds: list[str] = field(default_factory=list)
    accumulated_context: str = ""             # RAG context, grows each round
    instructions: Optional[str] = None        # final markdown instructions for writing agent

    # ── Serialisation helpers ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "qa_rounds": self.qa_rounds,
            "accumulated_context": self.accumulated_context,
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanningState:
        if not data:
            return cls()
        return cls(
            status=PlanningStatus(data.get("status", "idle")),
            qa_rounds=data.get("qa_rounds", []),
            accumulated_context=data.get("accumulated_context", ""),
            instructions=data.get("instructions"),
        )


# ── Writing context bundle (passed through the pipeline) ─────────────────

@dataclass
class WritingContext:
    """
    Everything a downstream agent needs to do its job.
    Built by the orchestrator and enriched by the planning agent.
    """
    # From the request
    user_message: str
    section_target: Optional[str] = None
    current_section: Optional[str] = None
    referenced_sections: list[dict] = field(default_factory=list)  # [{section_type, content}, ...]
    ruleset: Optional[str] = None
    paper_ids: list[str] = field(default_factory=list)

    # Set by the orchestrator
    decision: Optional[OrchestratorDecision] = None

    # Set by the planning agent
    planning_instructions: Optional[str] = None  # final markdown instructions for writing agent
