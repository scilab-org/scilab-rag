"""
Validation Agent — structural LaTeX validation and fixing.

Stripped down to ONLY handle structural LaTeX issues:
  1. Programmatic checks (syntax via pylatexenc, ref/label integrity)
  2. LLM fixer ONLY if programmatic checks find issues — fixes structure only

Does NOT check or modify:
  - Citations (no _check_citations — was causing blind add/remove)
  - Content quality or factual accuracy
  - Style or ruleset compliance (handled by RulesetValidator separately)
"""

import json
import logging
from typing import TYPE_CHECKING, Optional

from llama_index.core.llms import ChatMessage, LLM

from app.agents.writing.models import WritingContext
from app.agents.writing.prompts import LATEX_VALIDATION_PROMPT, VALIDATION_SYSTEM_PROMPT
from app.services.latex_validator import (
    check_ref_integrity,
    validate_latex_syntax,
)

if TYPE_CHECKING:
    from app.agents.writing.debug import WritePipelineDebugger

logger = logging.getLogger(__name__)

_PHASE = "validation"


class ValidationAgent:
    """
    Validates and auto-fixes structural LaTeX issues only.

    Combines programmatic checks (fast, deterministic) with an LLM fixer
    that is invoked ONLY when programmatic checks find issues.

    Does NOT touch content, citations, or style.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def validate(
        self,
        content: str,
        ctx: WritingContext,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> dict:
        """
        Run structural LaTeX validation.

        Returns:
            {
                "content": str,           # final (possibly fixed) LaTeX
                "validation_summary": {
                    "issues_found": int,
                    "issues_fixed": int,
                }
            }
        """
        if dbg:
            dbg.log_step(_PHASE, "scope", "structural_only")

        # ── Programmatic checks ──────────────────────────────────────────
        programmatic_issues = self._run_programmatic_checks(content)

        if dbg:
            dbg.log_step(_PHASE, "programmatic_issues", programmatic_issues)

        issues_found = len(programmatic_issues)

        if not programmatic_issues:
            # No structural issues — pass through unchanged
            logger.info("LaTeX validation passed — no structural issues found")
            if dbg:
                dbg.log_step(_PHASE, "result", "passed_no_issues")

            return {
                "content": content,
                "validation_summary": {
                    "issues_found": 0,
                    "issues_fixed": 0,
                },
            }

        # ── LLM structural fix (only invoked when issues exist) ──────────
        logger.info("LaTeX validation found %d structural issues, invoking LLM fixer", issues_found)

        fixed_content = await self._fix_structural_issues(
            content, programmatic_issues, dbg=dbg,
        )

        issues_fixed = issues_found if fixed_content != content else 0

        if dbg:
            dbg.log_step(_PHASE, "content_changed", fixed_content != content)

        summary = {
            "issues_found": issues_found,
            "issues_fixed": issues_fixed,
        }

        if dbg:
            dbg.log_step(_PHASE, "summary", summary)

        return {
            "content": fixed_content,
            "validation_summary": summary,
        }

    # ── Programmatic checks ──────────────────────────────────────────────

    def _run_programmatic_checks(self, content: str) -> list[dict]:
        """Run fast, deterministic structural LaTeX checks."""
        issues: list[dict] = []

        # Syntax checks (braces, environments, commands)
        syntax_result = validate_latex_syntax(content)
        for issue in syntax_result.issues:
            issues.append({
                "type": issue.type,
                "description": issue.description,
                "severity": issue.severity,
            })

        # Ref/label integrity
        ref_issues = check_ref_integrity(content)
        for issue in ref_issues:
            issues.append({
                "type": issue.type,
                "description": issue.description,
                "severity": issue.severity,
            })

        return issues

    # ── LLM structural fixer ─────────────────────────────────────────────

    async def _fix_structural_issues(
        self,
        content: str,
        programmatic_issues: list[dict],
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> str:
        """Call the LLM to fix structural LaTeX issues only."""

        issues_text = "\n".join(
            f"- [{i['severity']}] {i['description']}" for i in programmatic_issues
        )

        user_prompt = LATEX_VALIDATION_PROMPT.format(
            content=content,
            programmatic_issues=issues_text,
        )

        if dbg:
            dbg.log_step(_PHASE, "llm_fix_prompt", user_prompt)

        messages = [
            ChatMessage(role="system", content=VALIDATION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        async with (dbg.llm_timer("validation", "fix_structural") if dbg else _noop_ctx()) as _t:
            response = await self._llm.achat(messages)
        raw = _strip_json_fences((response.message.content or "").strip())

        if dbg:
            dbg.log_step(_PHASE, "llm_fix_raw_response", raw)

        try:
            result = json.loads(raw)
            fixed = result.get("fixed_content", content)
            if dbg:
                dbg.log_step(_PHASE, "llm_fix_parse", {
                    "success": True,
                    "has_issues": result.get("has_issues", False),
                    "issue_count": len(result.get("issues", [])),
                })
            return fixed
        except json.JSONDecodeError:
            logger.warning("Validation LLM returned invalid JSON, keeping original: %s", raw[:200])
            if dbg:
                dbg.log_step(_PHASE, "llm_fix_parse", {
                    "success": False,
                    "error": "JSONDecodeError",
                })
            return content


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers."""
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
