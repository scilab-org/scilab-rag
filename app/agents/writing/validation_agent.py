"""
Validation Agent — 3-stage pipeline.

  Stage 1:  Grammar & diction check (LLM)
  Stage 2a: Checklist rules check (LLM) — only if checklist_items provided
  Stage 2b: Journal style audit (LLM)   — only if journal_style provided
  Stage 3:  Citation check
    3a. Programmatic — invalid \cite{} keys + raw @article blocks in body
    3b. Fact-check — per-claim retrieval (LLM, full sentence preserved)

All stages emit a unified ValidationIssue dict. Output is a well-formatted
markdown summary (persisted as message.content) plus the flat issues list
(persisted in metadata for Fix-All).
"""

import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llama_index.core.llms import ChatMessage, LLM

from app.agents.writing.models import WritingContext
from app.agents.writing.planning_agent import _build_attribution
from app.agents.writing.prompts import (
    CHECKLIST_VALIDATION_PROMPT,
    CITATION_FACT_CHECK_PROMPT,
    GRAMMAR_VALIDATION_PROMPT,
    JOURNAL_STYLE_AUDIT_PROMPT,
    LATEX_VALIDATION_PROMPT,
    RULESET_VALIDATION_PROMPT,
    VALIDATION_SYSTEM_PROMPT,
)
from app.services.latex_validator import (
    check_ref_integrity,
    extract_citations,
    validate_latex_syntax,
)

if TYPE_CHECKING:
    from app.agents.writing.debug import WritePipelineDebugger

logger = logging.getLogger(__name__)


def _format_citation_chunk(c: dict) -> str:
    """Format a retrieved chunk with a source attribution header."""
    attribution = _build_attribution(
        authors=c.get("authors", ""),
        pub_year_str=c.get("publication_month_year", ""),
        paper_name=c.get("paper_name", ""),
    )
    header = f"[Source: {attribution}]" if attribution else "[Source: unknown]"
    return f"{header}\n{c['text'].strip()}"


# ── Sentence truncation ──────────────────────────────────────────────────

_SENTENCE_MAX = 120


def _truncate(s: str, max_len: int = _SENTENCE_MAX) -> str:
    """Mid-truncate a string to max_len characters."""
    if not s or len(s) <= max_len:
        return s
    half = (max_len - 3) // 2
    return s[:half] + "..." + s[-(max_len - half - 3):]


def _cap(s: str, max_len: int = 200) -> str:
    """Hard-cap a string (tail truncate) — safety net for LLM-generated detail."""
    if not s or len(s) <= max_len:
        return s
    return s[:max_len - 1] + "…"


# ── Issue builder ────────────────────────────────────────────────────────

def _make_issue(
    stage: str,
    severity: str,
    rule: str,
    sentence: str,
    detail: str,
    cite_key: Optional[str] = None,
    preserve_sentence: bool = False,
    preserve_detail: bool = False,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "stage": stage,
        "severity": severity,
        "rule": rule,
        "sentence": sentence if preserve_sentence else _truncate(sentence),
        "detail": detail if preserve_detail else detail,
        **({"citeKey": cite_key} if cite_key else {}),
    }


# ── Strip JSON fences ────────────────────────────────────────────────────

def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class ValidationAgent:
    """
    Combines grammar checking, checklist/journal-style compliance,
    programmatic citation key + @article-block detection, and
    citation fact-checking.
    """

    def __init__(
        self,
        llm: LLM,
        fact_check_llm: Optional[LLM] = None,
        graph_store: Optional[Any] = None,
        embed_model: Optional[Any] = None,
    ) -> None:
        self._llm = llm
        self._fact_check_llm = fact_check_llm or llm
        self._graph_store = graph_store
        self._embed_model = embed_model

    # ════════════════════════════════════════════════════════════════════
    # PUBLIC: user-triggered full validation
    # ════════════════════════════════════════════════════════════════════

    async def validate_content(self, ctx: WritingContext) -> dict:
        """
        Run all 3 validation stages and return:
        {
            "markdown_summary": str,   # persisted as message.content
            "issues": list[dict],      # flat unified list for metadata
            "has_issues": bool,
        }
        """
        if not ctx.current_section:
            return {
                "markdown_summary": "## Validation Results\n\n✓ No content to validate.",
                "issues": [],
                "has_issues": False,
            }

        content = ctx.current_section
        all_issues: List[dict] = []

        # ── Stage 1: Grammar ─────────────────────────────────────────────
        try:
            grammar_issues = await self._validate_grammar(content)
            all_issues.extend(grammar_issues)
        except Exception:
            logger.exception("Grammar check failed, skipping")

        # ── Stage 2a: Checklist ───────────────────────────────────────────
        if ctx.checklist_items:
            try:
                checklist_issues = await self._validate_checklist(
                    content, ctx.checklist_items
                )
                all_issues.extend(checklist_issues)
            except Exception:
                logger.exception("Checklist check failed, skipping")

        # ── Stage 2b: Journal style audit ────────────────────────────────
        if ctx.journal_style:
            try:
                style_issues = await self._validate_journal_style(
                    content, ctx.journal_style, ctx.section_target or ""
                )
                all_issues.extend(style_issues)
            except Exception:
                logger.exception("Journal style check failed, skipping")

        # ── Stage 3a: Programmatic citation checks ───────────────────────
        programmatic = self._check_citation_keys(content, ctx.cite_key_map)
        programmatic += self._check_article_blocks(content)
        all_issues.extend(programmatic)

        # ── Stage 3b: Citation fact-check (LLM + retrieval) ─────────────
        # if self._graph_store and self._embed_model and ctx.paper_ids:
        #     try:
        #         fact_issues = await self._validate_citation_facts(content, ctx)
        #         all_issues.extend(fact_issues)
        #     except Exception:
        #         logger.exception("Citation fact-check failed, skipping")

        has_issues = bool(all_issues)
        markdown_summary = self._format_summary(all_issues)

        return {
            "markdown_summary": markdown_summary,
            "issues": all_issues,
            "has_issues": has_issues,
        }

    # ════════════════════════════════════════════════════════════════════
    # PUBLIC: inline ruleset pass (called during write pipeline)
    # ════════════════════════════════════════════════════════════════════

    async def validate_ruleset(self, content: str, ruleset: str) -> dict:
        """
        Check written content against the user-provided ruleset only.

        Called between the write step and LaTeX structural validation in the
        write pipeline.  Skipped entirely when ruleset is empty.

        Returns:
            {
                "has_issues": bool,
                "issues_text": str,  # formatted bullet string for rewrite prompt
            }
        """
        prompt = RULESET_VALIDATION_PROMPT.format(ruleset=ruleset, content=content)
        messages = [ChatMessage(role="user", content=prompt)]

        try:
            response = await self._llm.achat(messages)
            raw = _strip_json((response.message.content or "").strip())
            result = json.loads(raw)
            has_issues = result.get("has_issues", False)
            issues = result.get("issues", [])

            issues_text = "\n".join(
                f"- [{i.get('rule', 'Style Rule')}] {i.get('description', '')} "
                f"(at: {i.get('location', 'unspecified')})"
                for i in issues
            ) if issues else ""

            return {"has_issues": has_issues, "issues_text": issues_text}

        except (json.JSONDecodeError, Exception):
            logger.warning("Ruleset validation failed, skipping rewrite", exc_info=True)
            return {"has_issues": False, "issues_text": ""}

    # ════════════════════════════════════════════════════════════════════
    # STAGE 1: Grammar
    # ════════════════════════════════════════════════════════════════════

    async def _validate_grammar(self, content: str) -> List[dict]:
        prompt = GRAMMAR_VALIDATION_PROMPT.format(content=content)
        messages = [ChatMessage(role="user", content=prompt)]

        response = await self._fact_check_llm.achat(messages)
        raw = _strip_json((response.message.content or "").strip())

        try:
            result = json.loads(raw)
            issues = []
            for item in result.get("issues", []):
                issues.append(_make_issue(
                    stage="grammar",
                    severity="warning",
                    rule=item.get("rule", "Grammar"),
                    sentence=item.get("sentence", ""),
                    detail=item.get("detail", ""),
                ))
            return issues
        except json.JSONDecodeError:
            logger.warning("Grammar validation returned invalid JSON: %s", raw[:200])
            return []

    # ════════════════════════════════════════════════════════════════════
    # STAGE 2: Checklist + journal style
    # ════════════════════════════════════════════════════════════════════

    async def _validate_checklist(
        self,
        content: str,
        checklist_items: List[dict],
    ) -> List[dict]:
        # Format checklist items as a numbered list for the prompt
        items_text = "\n".join(
            f"{i + 1}. [{item.get('id', '')}] **{item.get('name', '')}** "
            f"(weight: {item.get('weight', 1)}): {item.get('rule', '')}"
            for i, item in enumerate(checklist_items)
        )

        prompt = CHECKLIST_VALIDATION_PROMPT.format(
            checklist_items=items_text,
            content=content,
        )
        messages = [ChatMessage(role="user", content=prompt)]

        response = await self._fact_check_llm.achat(messages)
        raw = _strip_json((response.message.content or "").strip())

        try:
            result = json.loads(raw)
            issues = []
            # LLM returns ALL items; filter FAILs here in Python
            for item in result.get("results", []):
                if item.get("status", "PASS") != "FAIL":
                    continue
                sentence = item.get("sentence", "")
                detail = item.get("detail", "")
                # Skip empty FAIL entries — LLM returned FAIL with no evidence
                if not sentence and not detail:
                    continue
                issues.append(_make_issue(
                    stage="semantic",
                    severity="error",
                    rule=item.get("rule", "Checklist rule"),
                    sentence=sentence,
                    detail=detail,
                ))
            return issues
        except json.JSONDecodeError:
            logger.warning("Checklist validation returned invalid JSON: %s", raw[:200])
            return []

    # ════════════════════════════════════════════════════════════════════
    # STAGE 2b: Journal style audit
    # ════════════════════════════════════════════════════════════════════

    async def _validate_journal_style(
        self,
        content: str,
        journal_style: str,
        section_target: str,
    ) -> List[dict]:
        prompt = JOURNAL_STYLE_AUDIT_PROMPT.format(
            journal_style=journal_style,
            section_target=section_target or "this section",
            content=content,
        )
        messages = [ChatMessage(role="user", content=prompt)]

        response = await self._fact_check_llm.achat(messages)
        raw = _strip_json((response.message.content or "").strip())

        try:
            result = json.loads(raw)
            issues = []
            for item in result.get("violations", []):
                sentence = item.get("sentence", "")
                detail = item.get("detail", "")
                rule = item.get("rule", "Style Rule")
                if not detail:
                    continue
                issues.append(_make_issue(
                    stage="style",
                    severity="error",
                    rule=rule,
                    sentence=sentence,
                    detail=detail,
                ))
            return issues
        except json.JSONDecodeError:
            logger.warning("Journal style validation returned invalid JSON: %s", raw[:200])
            return []

    # ════════════════════════════════════════════════════════════════════
    # STAGE 3a: Programmatic citation checks
    # ════════════════════════════════════════════════════════════════════

    def _check_citation_keys(
        self, content: str, cite_key_map: Dict[str, str]
    ) -> List[dict]:
        """Flag \cite{key} commands whose key is not in the library."""
        issues = []
        # Build reverse map: cite_key → paper_id
        known_keys = set(cite_key_map.values())
        used_keys = extract_citations(content)

        for key in used_keys:
            if key not in known_keys:
                # Find the sentence containing this citation
                sentence = self._find_sentence(content, rf"\\[A-Za-z]*cite[A-Za-z]*\*?(?:\[[^\]]*\])*\{{[^}}]*\b{re.escape(key)}\b[^}}]*\}}")
                issues.append(_make_issue(
                    stage="citation",
                    severity="error",
                    rule="Invalid Citation Key",
                    sentence=sentence or rf"\cite{{{key}}}",
                    detail=f"Citation key '{key}' does not exist in the reference library.",
                    cite_key=key,
                ))
        return issues

    def _check_article_blocks(self, content: str) -> List[dict]:
        """
        Flag raw BibTeX @article/@inproceedings/etc. blocks embedded
        directly in the LaTeX section body — these are major errors.
        """
        issues = []
        # Match @type{ or @type ( entry blocks
        pattern = re.compile(
            r'(@(?:article|inproceedings|book|misc|techreport|phdthesis|mastersthesis|'
            r'incollection|conference|proceedings)\s*\{[^}]{0,120})',
            re.IGNORECASE,
        )
        for match in pattern.finditer(content):
            snippet = match.group(1).strip()
            issues.append(_make_issue(
                stage="citation",
                severity="error",
                rule="Embedded @article Block",
                sentence=snippet,
                detail=(
                    "Raw BibTeX entry found in section body. "
                    "Remove this entire block from the section content completely."
                ),
            ))
        return issues

    # ════════════════════════════════════════════════════════════════════
    # STAGE 3b: Citation fact-check (LLM + retrieval)
    # ════════════════════════════════════════════════════════════════════

    async def _validate_citation_facts(
        self, content: str, ctx: WritingContext
    ) -> List[dict]:
        inverse_map: Dict[str, str] = {v: k for k, v in ctx.cite_key_map.items()}
        cite_keys = extract_citations(content)
        claims = self._extract_claim_sentences(content, cite_keys)

        staged = []
        for cite_key, sentence in claims:
            paper_id = inverse_map.get(cite_key)
            if not paper_id:
                continue
            embedding = self._embed_model.get_text_embedding(sentence)
            chunks = self._graph_store.retrieve_chunks(embedding, [paper_id], top_k=3)
            chunks_text = "\n---\n".join(
                _format_citation_chunk(c) for c in chunks if c.get("text")
            )
            if chunks_text:
                staged.append((cite_key, sentence, chunks_text))

        if not staged:
            return []

        results = await asyncio.gather(
            *[self._check_one_citation(ck, s, ct) for ck, s, ct in staged],
            return_exceptions=True,
        )

        issues = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Citation fact-check error: %s", r)
            elif isinstance(r, dict):
                issues.append(r)
        return issues

    async def _check_one_citation(
        self, cite_key: str, sentence: str, chunks_text: str
    ) -> Optional[dict]:
        prompt = CITATION_FACT_CHECK_PROMPT.format(
            cite_key=cite_key,
            claim=sentence,
            retrieved_context=chunks_text,
        )
        messages = [ChatMessage(role="user", content=prompt)]

        try:
            response = await self._fact_check_llm.achat(messages)
            raw = _strip_json((response.message.content or "").strip())
            result = json.loads(raw)

            if result.get("supported", True):
                return None

            return _make_issue(
                stage="citation",
                severity="warning",
                rule="Fact Check",
                sentence=sentence,
                detail=result.get("issue", "Claim not supported by cited paper."),
                cite_key=cite_key,
                preserve_sentence=True,
                preserve_detail=True,
            )
        except Exception:
            logger.warning("Citation fact-check LLM error for [%s], skipping", cite_key)
            return None

    # ════════════════════════════════════════════════════════════════════
    # LaTeX structural validation (write-mode only, unchanged)
    # ════════════════════════════════════════════════════════════════════

    async def validate_latex(
        self,
        content: str,
        ctx: WritingContext,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> dict:
        _phase = "validation"
        if dbg:
            dbg.log_step(_phase, "scope", "structural_only")

        programmatic_issues = self._run_programmatic_checks(content)

        if dbg:
            dbg.log_step(_phase, "programmatic_issues", programmatic_issues)

        issues_found = len(programmatic_issues)

        if not programmatic_issues:
            if dbg:
                dbg.log_step(_phase, "result", "passed_no_issues")
            return {
                "content": content,
                "validation_summary": {"issues_found": 0, "issues_fixed": 0},
            }

        fixed_content = await self._fix_structural_issues(
            content, programmatic_issues, dbg=dbg
        )
        issues_fixed = issues_found if fixed_content != content else 0
        summary = {"issues_found": issues_found, "issues_fixed": issues_fixed}

        if dbg:
            dbg.log_step(_phase, "summary", summary)

        return {"content": fixed_content, "validation_summary": summary}

    # ════════════════════════════════════════════════════════════════════
    # Markdown summary formatter
    # ════════════════════════════════════════════════════════════════════

    def _format_summary(self, issues: List[dict]) -> str:
        # Group by stage
        by_stage: Dict[str, List[dict]] = {
            "grammar": [],
            "semantic": [],
            "style": [],
            "citation": [],
        }
        for issue in issues:
            by_stage.setdefault(issue["stage"], []).append(issue)

        severity_icon = {"error": "✕", "warning": "⚠"}
        lines = ["# Validation Results", ""]

        def _render_issues(stage_issues: List[dict]) -> None:
            for issue in stage_issues:
                icon = severity_icon.get(issue["severity"], "⚠")
                safe_sentence = " ".join(issue.get("sentence", "").split())
                line = f"- {icon} **[{issue['rule']}]**"
                if safe_sentence:
                    line += f" `{safe_sentence}`"
                if issue.get("detail"):
                    line += f" — {issue['detail']}"
                lines.append(line)

        # ── 1. Grammar ───────────────────────────────────────────────────
        grammar_issues = by_stage.get("grammar", [])
        if grammar_issues:
            error_count = sum(1 for i in grammar_issues if i["severity"] == "error")
            warn_count = sum(1 for i in grammar_issues if i["severity"] == "warning")
            counts = []
            if error_count:
                counts.append(f"{error_count} error{'s' if error_count > 1 else ''}")
            if warn_count:
                counts.append(f"{warn_count} warning{'s' if warn_count > 1 else ''}")
            lines.append(f"## 1. Lexical & Grammar  ·  {' · '.join(counts)}")
            lines.append("")
            _render_issues(grammar_issues)
        else:
            lines.append("## 1. Lexical & Grammar")
            lines.append("")
            lines.append("✓ No grammar errors found.")
        lines.append("")

        # ── 2. Style & Checklist ─────────────────────────────────────────
        checklist_issues = by_stage.get("semantic", [])
        style_issues = by_stage.get("style", [])
        total_sc = len(checklist_issues) + len(style_issues)
        if total_sc:
            lines.append(f"## 2. Style & Checklist  ·  {total_sc} issue{'s' if total_sc != 1 else ''}")
        else:
            lines.append("## 2. Style & Checklist")
        lines.append("")

        lines.append("### Checklist")
        if checklist_issues:
            _render_issues(checklist_issues)
        else:
            lines.append("✓ All checklist rules passed.")
        lines.append("")

        lines.append("### Journal Style")
        if style_issues:
            _render_issues(style_issues)
        else:
            lines.append("✓ All journal style rules passed.")
        lines.append("")

        # ── 3. Citations ─────────────────────────────────────────────────
        citation_issues = by_stage.get("citation", [])
        if citation_issues:
            error_count = sum(1 for i in citation_issues if i["severity"] == "error")
            warn_count = sum(1 for i in citation_issues if i["severity"] == "warning")
            counts = []
            if error_count:
                counts.append(f"{error_count} error{'s' if error_count > 1 else ''}")
            if warn_count:
                counts.append(f"{warn_count} warning{'s' if warn_count > 1 else ''}")
            lines.append(f"## 3. Citations  ·  {' · '.join(counts)}")
            lines.append("")
            _render_issues(citation_issues)
        else:
            lines.append("## 3. Citations")
            lines.append("")
            lines.append("✓ All citations are valid.")
        lines.append("")

        total = len(issues)
        lines.append("---")
        lines.append(f"{total} issue{'s' if total != 1 else ''} found.")

        return "\n".join(lines)

    # ════════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════════

    def _find_sentence(self, content: str, pattern: str) -> Optional[str]:
        """Return the full sentence containing the first regex match."""
        sentence_re = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_re.split(content)
        key_re = re.compile(pattern)
        for sentence in sentences:
            if key_re.search(sentence):
                return sentence.strip()
        return None

    def _extract_claim_sentences(
        self, content: str, cite_keys: List[str]
    ) -> List[tuple]:
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(content)
        seen: set = set()
        result = []
        for cite_key in cite_keys:
            if cite_key in seen:
                continue
            key_pattern = re.compile(
                r'\\[A-Za-z]*cite[A-Za-z]*\*?(?:\[[^\]]*\])*\{[^}]*\b'
                + re.escape(cite_key)
                + r'\b[^}]*\}'
            )
            for sentence in sentences:
                if key_pattern.search(sentence):
                    result.append((cite_key, sentence.strip()))
                    seen.add(cite_key)
                    break
        return result

    def _run_programmatic_checks(self, content: str) -> list[dict]:
        issues: list[dict] = []
        syntax_result = validate_latex_syntax(content)
        for issue in syntax_result.issues:
            issues.append({
                "type": issue.type,
                "description": issue.description,
                "severity": issue.severity,
            })
        ref_issues = check_ref_integrity(content)
        for issue in ref_issues:
            issues.append({
                "type": issue.type,
                "description": issue.description,
                "severity": issue.severity,
            })
        return issues

    async def _fix_structural_issues(
        self,
        content: str,
        programmatic_issues: list[dict],
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> str:
        _phase = "validation"
        issues_text = "\n".join(
            f"- [{i['severity']}] {i['description']}" for i in programmatic_issues
        )
        user_prompt = LATEX_VALIDATION_PROMPT.format(
            content=content,
            programmatic_issues=issues_text,
        )
        if dbg:
            dbg.log_step(_phase, "llm_fix_prompt", user_prompt)

        messages = [
            ChatMessage(role="system", content=VALIDATION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]
        async with (
            dbg.llm_timer("validation", "fix_structural") if dbg else _noop_ctx()
        ):
            response = await self._llm.achat(messages)
        fixed = (response.message.content or "").strip()

        if dbg:
            dbg.log_step(_phase, "llm_fix_response", fixed)

        return fixed if fixed else content


class _NoopCtx:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *_exc):
        pass


def _noop_ctx() -> _NoopCtx:
    return _NoopCtx()
