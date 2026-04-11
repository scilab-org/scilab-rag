"""
Validation Agent — auto-fix loop that checks LaTeX output quality.

Runs up to MAX_ITERATIONS (3) cycles of:
  1. Programmatic checks (syntax via pylatexenc, citation keys, ref integrity)
  2. LLM-based checks (content accuracy, style compliance)
  3. If issues found → LLM fixes them → repeat

Four scopes (set by the orchestrator):
  - full:          syntax + content + style
  - content_only:  citation/reference check + factual consistency
  - syntax_only:   LaTeX syntax only
  - style_only:    ruleset compliance only
"""

import json
import logging
from typing import TYPE_CHECKING, Optional

from llama_index.core.llms import ChatMessage, LLM

from app.agents.writing.models import ValidationScope, WritingContext
from app.agents.writing.prompts import VALIDATION_SCOPE_PROMPTS, VALIDATION_SYSTEM_PROMPT
from app.services.latex_validator import (
    check_ref_integrity,
    extract_citations,
    validate_latex_syntax,
)
from app.services.store import GraphRAGStore

if TYPE_CHECKING:
    from app.agents.writing.debug import WritePipelineDebugger

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

_PHASE = "validation"


class ValidationAgent:
    """
    Validates and auto-fixes LaTeX section output.

    Combines programmatic checks (fast, deterministic) with LLM-based checks
    (content accuracy, style compliance).  Runs an auto-fix loop up to
    MAX_ITERATIONS times.
    """

    def __init__(
        self,
        llm: LLM,
        graph_store: Optional[GraphRAGStore] = None,
    ) -> None:
        self._llm = llm
        self._graph_store = graph_store

    async def validate(
        self,
        content: str,
        ctx: WritingContext,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> dict:
        """
        Run the validation loop.

        Returns:
            {
                "content": str,           # final (possibly fixed) LaTeX
                "validation_summary": {
                    "iterations": int,
                    "issues_found": int,
                    "issues_fixed": int,
                    "scope": str,
                }
            }
        """
        if ctx.decision is None:
            raise ValueError("ValidationAgent requires ctx.decision to be set")
        scope = ctx.decision.validation_scope
        total_found = 0
        total_fixed = 0
        current = content

        if dbg:
            dbg.log_step(_PHASE, "scope", scope.value)

        iteration = 0
        for iteration in range(1, MAX_ITERATIONS + 1):
            logger.info("Validation iteration %d/%d, scope=%s", iteration, MAX_ITERATIONS, scope.value)

            # ── Programmatic checks ──────────────────────────────────────
            programmatic_issues = self._run_programmatic_checks(current, scope, ctx)

            if dbg:
                dbg.log_step(_PHASE, f"iter_{iteration}_programmatic_issues", programmatic_issues)

            # ── LLM-based checks + fix ───────────────────────────────────
            llm_result = await self._run_llm_validation(
                current, scope, ctx, programmatic_issues, iteration=iteration, dbg=dbg,
            )

            has_issues = llm_result.get("has_issues", False)
            issues = llm_result.get("issues", [])
            fixed = llm_result.get("fixed_content", current)

            issues_this_round = len(issues) + len(programmatic_issues)
            total_found += issues_this_round

            if not has_issues and not programmatic_issues:
                logger.info("Validation passed on iteration %d", iteration)
                if dbg:
                    dbg.log_step(_PHASE, f"iter_{iteration}_exit_reason", "no_issues")
                break

            # Count fixes (content changed)
            if fixed != current:
                total_fixed += issues_this_round
                current = fixed
                if dbg:
                    dbg.log_step(_PHASE, f"iter_{iteration}_content_changed", True)
            else:
                # LLM found issues but didn't fix → stop to avoid infinite loop
                logger.warning("Validation found issues but produced no fix, stopping")
                if dbg:
                    dbg.log_step(_PHASE, f"iter_{iteration}_content_changed", False)
                    dbg.log_step(_PHASE, f"iter_{iteration}_exit_reason", "no_fix_produced")
                break

        final_iteration = iteration  # always bound (initialised to 0 before loop)
        summary = {
            "iterations": final_iteration,
            "issues_found": total_found,
            "issues_fixed": total_fixed,
            "scope": scope.value,
        }

        if dbg:
            dbg.log_step(_PHASE, "summary", summary)

        return {
            "content": current,
            "validation_summary": summary,
        }

    # ── Programmatic checks ──────────────────────────────────────────────

    def _run_programmatic_checks(
        self,
        content: str,
        scope: ValidationScope,
        ctx: WritingContext,
    ) -> list[dict]:
        """Run fast, deterministic checks appropriate for the scope."""
        issues: list[dict] = []

        # Syntax checks (for full and syntax_only)
        if scope in (ValidationScope.FULL, ValidationScope.SYNTAX_ONLY):
            syntax_result = validate_latex_syntax(content)
            for issue in syntax_result.issues:
                issues.append({
                    "type": issue.type,
                    "description": issue.description,
                    "severity": issue.severity,
                })

            # Ref integrity
            ref_issues = check_ref_integrity(content)
            for issue in ref_issues:
                issues.append({
                    "type": issue.type,
                    "description": issue.description,
                    "severity": issue.severity,
                })

        # Citation check (for full and content_only)
        if scope in (ValidationScope.FULL, ValidationScope.CONTENT_ONLY):
            cite_issues = self._check_citations(content, ctx.paper_ids)
            issues.extend(cite_issues)

        return issues

    def _check_citations(self, content: str, paper_ids: list[str]) -> list[dict]:
        """Check that citation commands reference known keys, and that
        citations actually appear when papers are available."""
        if not self._graph_store or not paper_ids:
            return []

        # Resolve known cite keys from the graph store (paper_id → cite_key)
        known_cite_keys = self._graph_store.resolve_cite_keys(paper_ids)
        valid_keys = set(known_cite_keys.values())

        cited_keys = extract_citations(content)

        issues: list[dict] = []

        # Flag sections that have available papers but zero citations
        if not cited_keys and valid_keys:
            issues.append({
                "type": "citation",
                "description": (
                    "No citation commands found in the output, but papers are "
                    "available. Add \\autocite{key} or \\textcite{key} commands "
                    "to cite relevant sources."
                ),
                "severity": "error",
            })
            return issues

        # Flag unknown citation keys
        for key in cited_keys:
            if key not in valid_keys:
                issues.append({
                    "type": "citation",
                    "description": (
                        f"Citation key '{key}' does not match any known "
                        f"citation key in the project library"
                    ),
                    "severity": "warning",
                })

        return issues

    # ── LLM-based validation ─────────────────────────────────────────────

    async def _run_llm_validation(
        self,
        content: str,
        scope: ValidationScope,
        ctx: WritingContext,
        programmatic_issues: list[dict],
        iteration: int = 1,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> dict:
        """Call the LLM to check content quality and produce fixes."""

        # Build scope-specific prompt
        template = VALIDATION_SCOPE_PROMPTS.get(scope.value)
        if template is None:
            logger.warning("No validation template for scope %s", scope.value)
            return {"has_issues": False, "issues": [], "fixed_content": content}

        # Gather known citation keys for the prompt
        known_cite_keys: dict = {}
        if self._graph_store and ctx.paper_ids:
            known_cite_keys = self._graph_store.resolve_cite_keys(ctx.paper_ids)
        known_keys_str = ", ".join(sorted(known_cite_keys.values())) if known_cite_keys else "(none available)"

        # Ruleset description
        ruleset_desc = "(no specific rules)"
        if ctx.ruleset:
            ruleset_desc = ctx.ruleset

        user_prompt = template.format(
            content=content,
            known_citation_keys=known_keys_str,
            ruleset_description=ruleset_desc,
        )

        # Prepend programmatic issues if any
        if programmatic_issues:
            issues_text = "\n".join(
                f"- [{i['severity']}] {i['description']}" for i in programmatic_issues
            )
            user_prompt = (
                f"## Programmatic issues already detected\n{issues_text}\n\n"
                f"Fix these AND check for additional issues.\n\n{user_prompt}"
            )

        if dbg:
            dbg.log_step(_PHASE, f"iter_{iteration}_llm_prompt", user_prompt)

        messages = [
            ChatMessage(role="system", content=VALIDATION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        async with (dbg.llm_timer("validation", f"validate_iter_{iteration}") if dbg else _noop_ctx()) as _t:
            response = await self._llm.achat(messages)
        raw = _strip_json_fences((response.message.content or "").strip())

        if dbg:
            dbg.log_step(_PHASE, f"iter_{iteration}_llm_raw_response", raw)

        try:
            result = json.loads(raw)
            if dbg:
                dbg.log_step(_PHASE, f"iter_{iteration}_llm_parse", {
                    "success": True,
                    "has_issues": result.get("has_issues", False),
                    "issue_count": len(result.get("issues", [])),
                })
            return result
        except json.JSONDecodeError:
            logger.warning("Validation LLM returned invalid JSON: %s", raw[:200])
            if dbg:
                dbg.log_step(_PHASE, f"iter_{iteration}_llm_parse", {
                    "success": False,
                    "error": "JSONDecodeError",
                })
            return {"has_issues": False, "issues": [], "fixed_content": content}


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
