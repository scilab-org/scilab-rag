"""
Writing Agent — produces LaTeX content for a paper section.

Supports five writing modes: write_new, extend, rewrite, fix_content, fix_latex.
Each mode uses a mode-specific prompt template.  The agent always returns the
COMPLETE section (not a patch), plus a human-readable diff summary for the
chat timeline.
"""

import logging
from typing import TYPE_CHECKING, Optional

from llama_index.core.llms import ChatMessage, LLM

from app.agents.writing.models import WritingContext, WritingMode
from app.agents.writing.prompts import (
    WRITING_DIFF_SUMMARY_PROMPT,
    WRITING_MODE_PROMPTS,
    WRITING_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from app.agents.writing.debug import WritePipelineDebugger

logger = logging.getLogger(__name__)

_PHASE = "writing"


class WritingAgent:
    """Generates LaTeX content and a diff summary for a single section."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

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
                "diff_summary": str,     # human-readable changes for chat
            }
        """
        if ctx.decision is None:
            raise ValueError("WritingAgent requires ctx.decision to be set")
        mode = ctx.decision.writing_mode
        section_target = ctx.section_target or "unnamed"

        logger.info("Writing agent: mode=%s, section=%s", mode.value, section_target)

        if dbg:
            dbg.log_step(_PHASE, "mode", mode.value)
            dbg.log_step(_PHASE, "section_target", section_target)

        # ── Build the mode-specific user prompt ──────────────────────────
        template = WRITING_MODE_PROMPTS.get(mode.value)
        if template is None:
            raise ValueError(f"No prompt template for writing mode: {mode.value}")

        user_prompt = template.format(
            section_target=section_target,
            user_message=ctx.user_message,
            current_section=ctx.current_section or "(empty)",
            planning_instructions=ctx.planning_instructions or "(no planning context)",
            referenced_sections=_format_referenced_sections(ctx.referenced_sections),
            ruleset=_format_ruleset(ctx.ruleset),
        )

        if dbg:
            dbg.log_step(_PHASE, "selected_template", mode.value)
            dbg.log_step(_PHASE, "writing_prompt", user_prompt)

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

        # ── Generate diff summary ────────────────────────────────────────
        diff_summary = await self._generate_diff_summary(
            mode=mode,
            section_target=section_target,
            original=ctx.current_section,
            updated=latex_content,
            dbg=dbg,
        )

        return {
            "section_target": section_target,
            "content": latex_content,
            "diff_summary": diff_summary,
        }

    async def _generate_diff_summary(
        self,
        mode: WritingMode,
        section_target: str,
        original: Optional[str],
        updated: str,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> str:
        """Generate a human-readable summary of changes for the chat timeline."""

        prompt = WRITING_DIFF_SUMMARY_PROMPT.format(
            section_target=section_target,
            writing_mode=mode.value,
            original_content=original or "(no previous content — new section)",
            updated_content=_truncate(updated, 3000),
        )

        if dbg:
            dbg.log_step(_PHASE, "diff_summary_prompt", prompt)

        messages = [
            ChatMessage(role="user", content=prompt),
        ]

        async with (dbg.llm_timer("writing", "diff_summary") if dbg else _noop_ctx()) as _t:
            response = await self._llm.achat(messages)
        summary = (response.message.content or "").strip()

        if dbg:
            dbg.log_step(_PHASE, "diff_summary_response", summary)

        return summary


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
