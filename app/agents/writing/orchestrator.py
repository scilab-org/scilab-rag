"""
Writing Orchestrator — LLM-based intent classifier that routes each
write-mode request to the appropriate agent pipeline.

Stateless per request, except for checking planning_state to short-circuit
back to the planning agent when questions are pending.
"""

import json
import logging
from typing import TYPE_CHECKING, Optional

from llama_index.core.llms import ChatMessage, LLM

from app.agents.writing.models import (
    OrchestratorDecision,
    PlanningState,
    PlanningStatus,
    ValidationScope,
    WritingContext,
    WritingMode,
)
from app.agents.writing.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_USER_PROMPT,
)

if TYPE_CHECKING:
    from app.agents.writing.debug import WritePipelineDebugger

logger = logging.getLogger(__name__)

_PHASE = "orchestrator"


class WritingOrchestrator:
    """
    Classifies every incoming write-mode request into an OrchestratorDecision.

    Two code paths:
    1. **Planning short-circuit** — if planning_state.status is "asking",
       the user is answering planning questions.  Route directly to the
       planning agent without calling the LLM classifier.
    2. **LLM classification** — for all other requests, call the LLM to
       determine writing_mode, validation_scope, invoke_planning.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def classify(
        self,
        ctx: WritingContext,
        planning_state: Optional[PlanningState] = None,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> OrchestratorDecision:
        """
        Determine what to do with this request.

        Args:
            ctx: The full writing context built from the incoming request.
            planning_state: Current planning state from DB (may be None).
            dbg: Optional debug tracer.

        Returns:
            OrchestratorDecision describing the route for this request.
        """

        # ── Short-circuit: planning questions pending ────────────────────
        if planning_state and planning_state.status == PlanningStatus.ASKING:
            logger.info(
                "Planning short-circuit: status=%s, routing to planning agent",
                planning_state.status.value,
            )
            decision = OrchestratorDecision(
                writing_mode=WritingMode.WRITE_NEW,   # placeholder — planning agent decides
                validation_scope=ValidationScope.FULL,
                invoke_planning=True,
                reasoning=f"User is answering planning questions (status={planning_state.status.value})",
            )
            if dbg:
                dbg.log_step(_PHASE, "short_circuit", {
                    "activated": True,
                    "planning_status": planning_state.status.value,
                    "decision": decision.to_dict(),
                })
            return decision

        # ── LLM classification ──────────────────────────────────────────
        if dbg:
            dbg.log_step(_PHASE, "short_circuit", {"activated": False})
        return await self._classify_with_llm(ctx, dbg=dbg)

    async def _classify_with_llm(
        self,
        ctx: WritingContext,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> OrchestratorDecision:
        """Call the LLM to classify the user's writing intent."""

        # Format referenced sections for the prompt
        if ctx.referenced_sections:
            ref_text = "\n\n".join(
                f"### {s['section_type']}\n{s['content'][:500]}..."
                if len(s.get("content", "")) > 500
                else f"### {s['section_type']}\n{s.get('content', '')}"
                for s in ctx.referenced_sections
            )
        else:
            ref_text = "(none)"

        user_prompt = ORCHESTRATOR_USER_PROMPT.format(
            user_message=ctx.user_message,
            section_target=ctx.section_target or "(not specified)",
            current_section=_truncate(ctx.current_section, 1000) or "(empty — new section)",
            referenced_sections=ref_text,
        )

        if dbg:
            dbg.log_step(_PHASE, "user_prompt", user_prompt)

        messages = [
            ChatMessage(role="system", content=ORCHESTRATOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        logger.debug("Orchestrator LLM call — user_message: %s", ctx.user_message[:80])

        async with (dbg.llm_timer("orchestrator", "classify") if dbg else _noop_ctx()) as _t:
            response = await self._llm.achat(messages)
        raw = (response.message.content or "").strip()

        if dbg:
            dbg.log_step(_PHASE, "llm_raw_response", raw)

        # Strip markdown fences if the LLM wraps them
        raw = _strip_json_fences(raw)

        if dbg:
            dbg.log_step(_PHASE, "json_stripped", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Orchestrator LLM returned invalid JSON, falling back: %s", raw[:200])
            if dbg:
                dbg.log_step(_PHASE, "parse_result", {"success": False, "error": "JSONDecodeError"})
            decision = _fallback_decision(ctx)
            if dbg:
                dbg.log_step(_PHASE, "fallback_used", True)
                dbg.log_step(_PHASE, "final_decision", decision.to_dict())
            return decision

        if dbg:
            dbg.log_step(_PHASE, "parse_result", {"success": True, "data": data})

        decision = _parse_decision(data, ctx)

        if dbg:
            # Check if fallback was used (reasoning starts with "Fallback:")
            is_fallback = decision.reasoning.startswith("Fallback:")
            dbg.log_step(_PHASE, "fallback_used", is_fallback)
            dbg.log_step(_PHASE, "final_decision", decision.to_dict())

        return decision


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_decision(data: dict, ctx: WritingContext) -> OrchestratorDecision:
    """Parse the LLM's JSON into a validated OrchestratorDecision."""
    try:
        return OrchestratorDecision(
            writing_mode=WritingMode(data["writing_mode"]),
            validation_scope=ValidationScope(data["validation_scope"]),
            invoke_planning=bool(data.get("invoke_planning", False)),
            reasoning=data.get("reasoning", ""),
        )
    except (KeyError, ValueError) as exc:
        logger.warning("Failed to parse orchestrator decision (%s), falling back", exc)
        return _fallback_decision(ctx)


def _fallback_decision(ctx: WritingContext) -> OrchestratorDecision:
    """
    Conservative fallback when LLM classification fails.

    Heuristic:
    - No current_section → write_new with planning
    - Has current_section → extend without planning
    """
    if not ctx.current_section:
        return OrchestratorDecision(
            writing_mode=WritingMode.WRITE_NEW,
            validation_scope=ValidationScope.FULL,
            invoke_planning=True,
            reasoning="Fallback: no current section → write_new with planning",
        )
    return OrchestratorDecision(
        writing_mode=WritingMode.EXTEND,
        validation_scope=ValidationScope.FULL,
        invoke_planning=False,
        reasoning="Fallback: has current section → extend without planning",
    )


def _truncate(text: Optional[str], max_len: int) -> Optional[str]:
    """Truncate text for prompt inclusion."""
    if not text:
        return text
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... (truncated, {len(text)} chars total)"


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that LLMs sometimes add."""
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```)
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class _NoopCtx:
    """Sync/async context manager that does nothing (used when dbg is None)."""
    async def __aenter__(self):
        return self
    async def __aexit__(self, *_exc):
        pass


def _noop_ctx() -> _NoopCtx:
    return _NoopCtx()
