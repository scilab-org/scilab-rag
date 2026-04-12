"""
Writing Agent — produces LaTeX content for a paper section.

Uses a single unified template.  The LLM infers the action (write new,
extend, rewrite, fix) from the user's message and the available context.

The agent always returns the COMPLETE section (not a patch), plus a
structured explanation of what was done and why (replaces the old diff
summary).
"""

import logging
from typing import TYPE_CHECKING, Optional

from llama_index.core.llms import ChatMessage, LLM

from app.agents.writing.models import WritingContext
from app.agents.writing.prompts import (
    WRITING_EXPLAIN_PROMPT,
    WRITING_SYSTEM_PROMPT,
    WRITING_USER_PROMPT,
    WRITING_USER_PROMPT_WITH_RULESET_ISSUES,
)

if TYPE_CHECKING:
    from app.agents.writing.debug import WritePipelineDebugger

logger = logging.getLogger(__name__)

_PHASE = "writing"


class WritingAgent:
    """Generates LaTeX content and a structured explanation for a single section."""

    def __init__(self, llm: LLM, explain_llm: Optional[LLM] = None) -> None:
        self._llm = llm
        # explain_llm can be a cheaper/faster model for the explanation step
        self._explain_llm = explain_llm or llm

    async def write(
        self,
        ctx: WritingContext,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> dict:
        """
        Produce a section output.

        Returns:
            {
                "section_target": str,
                "content": str,          # full LaTeX
            }
        """
        section_target = ctx.section_target or "unnamed"

        logger.info("Writing agent: section=%s", section_target)

        if dbg:
            dbg.log_step(_PHASE, "section_target", section_target)

        # ── Build the unified user prompt ────────────────────────────────
        user_prompt = WRITING_USER_PROMPT.format(
            section_target=section_target,
            user_message=ctx.user_message,
            current_section=ctx.current_section or "(empty — new section)",
            previous_attempt=ctx.previous_attempt or "(none — first attempt)",
            planning_instructions=ctx.planning_instructions or "(no planning context — use user's request and current section directly)",
            referenced_sections=_format_referenced_sections(ctx.referenced_sections),
            ruleset=_format_ruleset(ctx.ruleset),
            available_citations=_format_available_citations(ctx.cite_key_map),
        )

        if dbg:
            dbg.log_step(_PHASE, "writing_prompt", user_prompt)

        latex_content = await self._call_writing_llm(user_prompt, dbg=dbg)

        return {
            "section_target": section_target,
            "content": latex_content,
        }

    async def rewrite_with_ruleset_issues(
        self,
        ctx: WritingContext,
        draft_with_issues: str,
        ruleset_issues: str,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> dict:
        """
        Re-run the writing agent with ruleset issues appended.

        Called when ruleset validation finds style violations.
        Only called once (single retry).

        Returns same structure as write().
        """
        section_target = ctx.section_target or "unnamed"

        logger.info("Writing agent (ruleset retry): section=%s", section_target)

        if dbg:
            dbg.log_step(_PHASE, "ruleset_retry", True)
            dbg.log_step(_PHASE, "ruleset_issues", ruleset_issues)

        user_prompt = WRITING_USER_PROMPT_WITH_RULESET_ISSUES.format(
            section_target=section_target,
            user_message=ctx.user_message,
            current_section=ctx.current_section or "(empty — new section)",
            previous_attempt=ctx.previous_attempt or "(none — first attempt)",
            planning_instructions=ctx.planning_instructions or "(no planning context)",
            referenced_sections=_format_referenced_sections(ctx.referenced_sections),
            ruleset=_format_ruleset(ctx.ruleset),
            available_citations=_format_available_citations(ctx.cite_key_map),
            draft_with_issues=draft_with_issues,
            ruleset_issues=ruleset_issues,
        )

        if dbg:
            dbg.log_step(_PHASE, "ruleset_retry_prompt", user_prompt)

        latex_content = await self._call_writing_llm(user_prompt, dbg=dbg)

        return {
            "section_target": section_target,
            "content": latex_content,
        }

    async def explain_output(
        self,
        ctx: WritingContext,
        final_content: str,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> str:
        """
        Generate a structured explanation of what was written and why.

        This replaces the old diff summary and is displayed in the chat
        timeline so the user can decide whether to accept or reject.

        Uses the explain_llm (can be a cheaper model).
        """
        section_target = ctx.section_target or "unnamed"

        prompt = WRITING_EXPLAIN_PROMPT.format(
            section_target=section_target,
            user_message=ctx.user_message,
            current_section=ctx.current_section or "(empty — new section)",
            previous_attempt=ctx.previous_attempt or "(none — first attempt)",
            final_content=_truncate(final_content, 3000),
            planning_instructions=ctx.planning_instructions or "(no planning context)",
            ruleset=ctx.ruleset or "(no ruleset)",
        )

        if dbg:
            dbg.log_step(_PHASE, "explain_prompt", prompt)

        messages = [
            ChatMessage(role="user", content=prompt),
        ]

        async with (dbg.llm_timer("writing", "explain") if dbg else _noop_ctx()) as _t:
            response = await self._explain_llm.achat(messages)
        explanation = (response.message.content or "").strip()

        if dbg:
            dbg.log_step(_PHASE, "explain_response", explanation)

        return explanation

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _call_writing_llm(
        self,
        user_prompt: str,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> str:
        """Call the writing LLM and clean the response."""
        messages = [
            ChatMessage(role="system", content=WRITING_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        async with (dbg.llm_timer("writing", "write") if dbg else _noop_ctx()) as _t:
            response = await self._llm.achat(messages)
        latex_content = (response.message.content or "").strip()

        if dbg:
            dbg.log_step(_PHASE, "llm_raw_response", latex_content)

        # Strip any markdown fences the LLM might add around the LaTeX
        latex_content = _strip_latex_fences(latex_content)

        if dbg:
            dbg.log_step(_PHASE, "cleaned_latex", latex_content)

        return latex_content


# ── Formatting helpers ───────────────────────────────────────────────────

def _format_referenced_sections(sections: list[dict]) -> str:
    """Format referenced sections for prompt inclusion."""
    if not sections:
        return "(none)"
    parts = []
    for s in sections:
        section_type = s.get("section_type", "unknown")
        content = s.get("content", "")
        parts.append(f"### {section_type}\n{content}")
    return "\n\n".join(parts)


def _format_available_citations(cite_key_map: dict) -> str:
    """Format {paper_id: cite_key} as a list of citation entries for the prompt.

    Shows both \\autocite (parenthetical) and \\textcite (narrative) forms so
    the LLM knows exactly which commands to use.
    """
    if not cite_key_map:
        return "(no citation keys available — do not use citation commands)"
    keys = sorted(k for k in cite_key_map.values() if k)
    if not keys:
        return "(no citation keys available — do not use citation commands)"
    lines = [
        f"- \\autocite{{{key}}} (parenthetical)  or  \\textcite{{{key}}} (narrative)"
        for key in keys
    ]
    return "\n".join(lines)


def _format_ruleset(ruleset: Optional[str]) -> str:
    """Format ruleset for prompt inclusion."""
    if not ruleset:
        return "(no specific style rules)"
    return ruleset


def _truncate(text: str, max_len: int) -> str:
    """Truncate for prompt inclusion."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... (truncated, {len(text)} chars total)"


def _strip_latex_fences(text: str) -> str:
    """Remove ```latex ... ``` or ```tex ... ``` wrappers."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class _NoopCtx:
    """Async context manager that does nothing (used when dbg is None)."""
    async def __aenter__(self):
        return self
    async def __aexit__(self, *_exc):
        pass


def _noop_ctx() -> _NoopCtx:
    return _NoopCtx()
