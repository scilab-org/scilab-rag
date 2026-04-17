"""
Shared domain types for the writing-feature agent pipeline.

Every agent (orchestrator, planning, writing, validation) imports from
here so there is a single source of truth for enums, decision objects,
and planning-state shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ── Enums ────────────────────────────────────────────────────────────────

class PlanningStatus(str, Enum):
    """Status of the planning phase stored in session.context JSONB."""
    IDLE = "idle"
    ASKING = "asking"          # questions sent, awaiting answers
    COMPLETE = "complete"      # instructions built, ready to write


# ── Orchestrator decision ────────────────────────────────────────────────

@dataclass
class OrchestratorDecision:
    """
    Binary decision from the orchestrator LLM.
    Determines only whether the planning phase (RAG + QnA) should run.
    """
    invoke_planning: bool
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
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
    initial_context: str = ""                 # RAG context from start_planning (round 1)
    answer_context: str = ""                  # RAG context from process_answers (round 2)
    instructions: Optional[str] = None        # final markdown instructions for writing agent

    # ── Serialisation helpers ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "qa_rounds": self.qa_rounds,
            "initial_context": self.initial_context,
            "answer_context": self.answer_context,
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanningState:
        if not data:
            return cls()
        return cls(
            status=PlanningStatus(data.get("status", "idle")),
            qa_rounds=data.get("qa_rounds", []),
            initial_context=data.get("initial_context", ""),
            answer_context=data.get("answer_context", ""),
            instructions=data.get("instructions"),
        )


# ── Writing context bundle (passed through the pipeline) ─────────────────

@dataclass
class WritingContext:
    """
    Everything a downstream agent needs to do its job.
    Built fresh each request from the HTTP body + session context.
    """
    # From the request
    user_message: str
    section_target: Optional[str] = None
    current_section: Optional[str] = None
    referenced_sections: list[dict] = field(default_factory=list)  # [{section_type, content}, ...]
    ruleset: Optional[str] = None
    section_context: Optional[str] = None
    paper_ids: list[str] = field(default_factory=list)

    # Set by the orchestrator
    decision: Optional[OrchestratorDecision] = None

    # Set by the planning agent
    planning_instructions: Optional[str] = None  # final markdown instructions for writing agent

    # Populated in chat.py before the writing agent runs (paper_id → cite_key)
    cite_key_map: Dict[str, str] = field(default_factory=dict)

    # Loaded from session.context["latest_output"] — last written content
    previous_attempt: Optional[str] = None

    # Last N written outputs from DB messages — passed to build_instructions only
    conversation_history: List[str] = field(default_factory=list)
