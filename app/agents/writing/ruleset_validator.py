"""
Ruleset Validator — checks writing output against user-provided style rules.

Sits between the writing agent and LaTeX validation in the pipeline.
If issues are found, the writing agent re-runs ONCE with the issues
appended to the original prompt.

Skipped entirely if no ruleset is configured.
"""

import json
import logging
from typing import TYPE_CHECKING, Optional

from llama_index.core.llms import ChatMessage, LLM

from app.agents.writing.prompts import RULESET_VALIDATION_PROMPT

if TYPE_CHECKING:
    from app.agents.writing.debug import WritePipelineDebugger

logger = logging.getLogger(__name__)

_PHASE = "ruleset_validation"


class RulesetValidator:
    """
    Checks written LaTeX output against a user-provided ruleset.

    Returns a list of style issues. The caller (chat.py) decides whether
    to trigger a writing agent re-run.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def validate(
        self,
        content: str,
        ruleset: str,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> dict:
        """
        Check content against ruleset.

        Args:
            content: The LaTeX section to check.
            ruleset: The user-provided style rules (markdown).
            dbg: Optional debug tracer.

        Returns:
            {
                "has_issues": bool,
                "issues": list[dict],       # [{rule, description, location}, ...]
                "issues_text": str,          # formatted string for writing agent re-run
            }
        """
        if dbg:
            dbg.log_step(_PHASE, "ruleset", ruleset[:500] if ruleset else "(none)")

        prompt = RULESET_VALIDATION_PROMPT.format(
            ruleset=ruleset,
            content=content,
        )

        if dbg:
            dbg.log_step(_PHASE, "prompt", prompt)

        messages = [
            ChatMessage(role="user", content=prompt),
        ]

        async with (dbg.llm_timer("ruleset_validation", "validate") if dbg else _noop_ctx()) as _t:
            response = await self._llm.achat(messages)
        raw = _strip_json_fences((response.message.content or "").strip())

        if dbg:
            dbg.log_step(_PHASE, "raw_response", raw)

        try:
            result = json.loads(raw)
            has_issues = result.get("has_issues", False)
            issues = result.get("issues", [])

            if dbg:
                dbg.log_step(_PHASE, "parse_result", {
                    "success": True,
                    "has_issues": has_issues,
                    "issue_count": len(issues),
                })

            # Build a formatted string for the writing agent re-run prompt
            if issues:
                issues_text = "\n".join(
                    f"- [{i.get('rule', 'unknown rule')}] {i.get('description', '')} "
                    f"(at: {i.get('location', 'unspecified')})"
                    for i in issues
                )
            else:
                issues_text = ""

            return {
                "has_issues": has_issues,
                "issues": issues,
                "issues_text": issues_text,
            }

        except json.JSONDecodeError:
            logger.warning("Ruleset validation LLM returned invalid JSON: %s", raw[:200])
            if dbg:
                dbg.log_step(_PHASE, "parse_result", {
                    "success": False,
                    "error": "JSONDecodeError",
                })
            # On parse failure, assume no issues — don't block the pipeline
            return {
                "has_issues": False,
                "issues": [],
                "issues_text": "",
            }


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
