"""
Planning Agent — gathers context before the writing agent runs.

Operates in a single-round planning flow:

  start_planning (Round 1):
    Query Refiner → targeted search queries
    RAG query (refined queries) → initial_context
    LLM(initial_context, user_message, current_section, previous_attempt) → questions or []
    Save {initial_context, questions} → return questions to user
    (If LLM returns [] → build_instructions immediately)

  process_answers (always terminal):
    Query Refiner(user's answer) → targeted search queries
    RAG query (refined queries) → answer_context
    build_instructions(initial_context, answer_context, qa_history)
    → writing phase (never asks follow-up questions)

RAG context is stored as two separate labeled fields:
  - initial_context: from the original user request (round 1)
  - answer_context: from the user's Q&A answers (round 2)
"""

import json
import logging
from typing import TYPE_CHECKING, Optional

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.embeddings import BaseEmbedding

from app.agents.writing.models import (
    PlanningQuestion,
    PlanningState,
    PlanningStatus,
    WritingContext,
)
from app.agents.writing.prompts import (
    PLANNING_BUILD_INSTRUCTIONS_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    PLANNING_USER_PROMPT,
    QUERY_REFINER_SYSTEM_PROMPT,
    QUERY_REFINER_USER_PROMPT,
)
from app.services.store import GraphRAGStore

if TYPE_CHECKING:
    from app.agents.writing.debug import WritePipelineDebugger

logger = logging.getLogger(__name__)

_PHASE = "planning"


def _build_attribution(authors: str, pub_year_str: str, paper_name: str = "") -> str:
    """Build a short attribution string like 'LeCun et al., 2015' or paper name."""
    year = ""
    if pub_year_str:
        for part in reversed(pub_year_str.strip().split()):
            if part.isdigit() and len(part) == 4:
                year = part
                break

    author_short = ""
    if authors:
        author_list = [a.strip() for a in authors.replace(";", ",").split(",") if a.strip()]
        if author_list:
            first_parts = author_list[0].strip().split()
            last_name = first_parts[0] if first_parts else author_list[0]
            author_short = f"{last_name} et al." if len(author_list) > 1 else last_name

    suffix = f" — {paper_name}" if paper_name else ""
    if author_short and year:
        return f"{author_short}, {year}{suffix}"
    if author_short:
        return f"{author_short}{suffix}"
    if year:
        return f"{year}{suffix}"
    return paper_name  # fall back to paper name if no author/year info


class PlanningAgent:
    """
    Gathers information needed before writing, via a unified planning
    loop that combines RAG retrieval and structured Q&A.
    """

    def __init__(
        self,
        llm: LLM,
        graph_store: Optional[GraphRAGStore] = None,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: int = 10,
    ) -> None:
        self._llm = llm
        self._graph_store = graph_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k

    # ── Public entry points ──────────────────────────────────────────────

    async def start_planning(
        self,
        ctx: WritingContext,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> dict:
        """
        Begin the planning phase (Round 1).

        Uses Query Refiner to produce targeted RAG queries, then retrieves
        context and decides whether to ask the user questions.

        Returns:
            If questions needed:
                {"action": "planning_questions", "planning_state": ..., "questions": [...]}
            If no questions needed (LLM returned []):
                {"action": "planning_complete", "planning_state": ..., "instructions": "..."}
        """
        if dbg:
            dbg.log_step(_PHASE, "mode", "start_planning_round_1")

        planning_state = PlanningState(status=PlanningStatus.ASKING)

        # ── 1. Query Refiner on user message ─────────────────────────────
        refined_queries = await self._refine_query(
            ctx.user_message, ctx, planning_state, dbg=dbg,
        )

        if dbg:
            dbg.log_step(_PHASE, "refined_queries", refined_queries)

        # ── 2. RAG retrieval → initial_context ───────────────────────────
        if refined_queries:
            seen: dict[str, None] = {}
            for query in refined_queries:
                lines = await self._retrieve_rag_context(query, ctx.paper_ids, dbg=dbg)
                for line in lines:
                    seen[line] = None
            planning_state.initial_context = "\n".join(seen.keys())
        else:
            if dbg:
                dbg.log_step(_PHASE, "rag_skipped", "query refiner returned empty — no RAG needed")

        if dbg:
            dbg.log_step(_PHASE, "initial_context_length", len(planning_state.initial_context))

        # ── 3. LLM: ask questions or signal readiness ────────────────────
        questions = await self._ask_or_ready(ctx, planning_state, qa_history="", dbg=dbg)

        if questions:
            planning_state.status = PlanningStatus.ASKING

            if dbg:
                dbg.log_step(_PHASE, "round_result", {
                    "needs_more": True,
                    "question_count": len(questions),
                })

            return {
                "action": "planning_questions",
                "planning_state": planning_state,
                "questions": [
                    {
                        "type": q.type,
                        "prompt": q.prompt,
                        "options": q.options,
                        "allowCustom": q.allow_custom,
                    }
                    for q in questions
                ],
            }

        # ── 4. LLM returned [] — build instructions immediately ─────────
        if dbg:
            dbg.log_step(_PHASE, "round_result", {"needs_more": False})

        qa_history = ""
        instructions = await self._build_instructions(ctx, planning_state, qa_history, dbg=dbg)

        planning_state.status = PlanningStatus.COMPLETE
        planning_state.instructions = instructions

        return {
            "action": "planning_complete",
            "planning_state": planning_state,
            "instructions": instructions,
        }

    async def process_answers(
        self,
        ctx: WritingContext,
        planning_state: PlanningState,
        user_answer: str,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> dict:
        """
        Process user's Q&A answers — always terminal.

        Appends the Q&A transcript to qa_rounds, runs Query Refiner + RAG
        on the answer (stored as answer_context), then goes straight to
        _build_instructions. Never asks follow-up questions.

        Always returns:
            {"action": "planning_complete", "planning_state": ..., "instructions": "..."}
        """
        if dbg:
            dbg.log_step(_PHASE, "mode", "process_answers_terminal")
            dbg.log_step(_PHASE, "user_answer", user_answer)

        # Record this Q&A round
        planning_state.qa_rounds.append(user_answer)

        # ── 1. Query Refiner on user answer ──────────────────────────────
        refined_queries = await self._refine_query(
            user_answer, ctx, planning_state, dbg=dbg,
        )

        if dbg:
            dbg.log_step(_PHASE, "refined_queries", refined_queries)

        # ── 2. RAG retrieval → answer_context ────────────────────────────
        if refined_queries:
            seen: dict[str, None] = {}
            for query in refined_queries:
                lines = await self._retrieve_rag_context(query, ctx.paper_ids, dbg=dbg)
                for line in lines:
                    seen[line] = None
            planning_state.answer_context = "\n".join(seen.keys())
        else:
            if dbg:
                dbg.log_step(_PHASE, "rag_skipped", "query refiner returned empty — no RAG needed")

        if dbg:
            dbg.log_step(_PHASE, "answer_context_length", len(planning_state.answer_context))

        # ── 3. Build instructions (always — no more questions) ───────────
        qa_history = self._format_qa_history(planning_state)
        instructions = await self._build_instructions(ctx, planning_state, qa_history, dbg=dbg)

        planning_state.status = PlanningStatus.COMPLETE
        planning_state.instructions = instructions

        return {
            "action": "planning_complete",
            "planning_state": planning_state,
            "instructions": instructions,
        }

    # ── Query Refiner ────────────────────────────────────────────────────

    async def _refine_query(
        self,
        raw_input: str,
        ctx: WritingContext,
        planning_state: PlanningState,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> list[str]:
        """
        Call the Query Refiner LLM to produce targeted search queries.

        Returns a list of query strings, or empty list if no RAG needed.
        """
        user_prompt = QUERY_REFINER_USER_PROMPT.format(
            user_message=raw_input,
            section_target=ctx.section_target or "(not specified)",
            initial_context=planning_state.initial_context[:2000] if planning_state.initial_context else "(none — first retrieval round)",
            previous_attempt=ctx.previous_attempt[:2000] if ctx.previous_attempt else "(none)",
        )

        if dbg:
            dbg.log_step(_PHASE, "query_refiner_prompt", user_prompt)

        messages = [
            ChatMessage(role="system", content=QUERY_REFINER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        async with (dbg.llm_timer("planning", "query_refiner") if dbg else _noop_ctx()) as _t:
            response = await self._llm.achat(messages)
        raw = _strip_json_fences((response.message.content or "").strip())

        if dbg:
            dbg.log_step(_PHASE, "query_refiner_raw_response", raw)

        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array")
            # Filter to strings only
            queries = [q for q in data if isinstance(q, str) and q.strip()]
            return queries
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Query Refiner returned invalid JSON (%s), using fallback query", exc)
            if dbg:
                dbg.log_step(_PHASE, "query_refiner_parse_error", str(exc))
            # Fallback: use raw input + section_target as a single query
            return [f"{raw_input} {ctx.section_target or ''}".strip()]

    # ── RAG retrieval ────────────────────────────────────────────────────

    async def _retrieve_rag_context(
        self,
        query_text: str,
        paper_ids: list[str],
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> list[str]:
        """Retrieve chunk-only context for the query. Returns deduplicated list of formatted lines."""
        if not self._graph_store or not self._embed_model or not paper_ids:
            if dbg:
                dbg.log_step(_PHASE, "rag_skipped", "missing graph_store, embed_model, or paper_ids")
            return []

        try:
            if dbg:
                dbg.log_step(_PHASE, "rag_query_text_actual", query_text)

            query_embedding = await self._embed_model.aget_query_embedding(query_text)

            chunks = self._graph_store.retrieve_chunks(
                query_embedding=query_embedding,
                paper_ids=paper_ids,
                top_k=5,
            )

            if dbg:
                dbg.log_step(_PHASE, "rag_raw_results", {"chunk_count": len(chunks)})

            parts: list[str] = []
            for chunk in chunks:
                text = chunk.get("text", "")
                paper = chunk.get("paper_name", "")
                authors = chunk.get("authors", "")
                pub_year = chunk.get("publication_month_year", "")
                cite_key = chunk.get("cite_key", "")
                if text:
                    attribution = _build_attribution(authors, pub_year, paper)
                    ck_suffix = f" @{cite_key}" if cite_key else ""
                    header = f"[{attribution}{ck_suffix}]" if (attribution or cite_key) else f"[{paper}]" if paper else ""
                    parts.append(f"{header} {text}" if header else text)

            deduped = list(dict.fromkeys(parts))

            if dbg:
                dbg.log_step(_PHASE, "rag_formatted_context", "\n".join(deduped))

            return deduped

        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            if dbg:
                dbg.log_step(_PHASE, "rag_error", str(exc))
            return []

    # ── LLM: ask questions or signal readiness ───────────────────────────

    async def _ask_or_ready(
        self,
        ctx: WritingContext,
        planning_state: PlanningState,
        qa_history: str,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> list[PlanningQuestion]:
        """
        Call the planning LLM.
        Returns a list of questions if more info is needed, or [] if ready.
        """
        user_prompt = PLANNING_USER_PROMPT.format(
            section_target=ctx.section_target or "unnamed",
            user_message=ctx.user_message,
            section_context=ctx.section_context or "(none)",
            initial_context=planning_state.initial_context or "(no RAG context available)",
            referenced_sections=_format_referenced_sections(ctx.referenced_sections),
            current_section=ctx.current_section or "(empty)",
            previous_attempt=ctx.previous_attempt or "(none — first attempt)",
            qa_history=qa_history or "(first round — no Q&A yet)",
        )

        if dbg:
            dbg.log_step(_PHASE, "ask_or_ready_prompt", user_prompt)

        messages = [
            ChatMessage(role="system", content=PLANNING_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        async with (dbg.llm_timer("planning", "ask_or_ready") if dbg else _noop_ctx()) as _t:
            response = await self._llm.achat(messages)
        raw = _strip_json_fences((response.message.content or "").strip())

        if dbg:
            dbg.log_step(_PHASE, "ask_or_ready_raw_response", raw)

        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Planning LLM returned invalid JSON (%s), using fallback questions", exc)
            if dbg:
                dbg.log_step(_PHASE, "ask_or_ready_parse", {"success": False, "error": str(exc)})
            # On parse failure in round 1, fall back to generic questions.
            # On parse failure in round 2+, assume ready (return []).
            if len(planning_state.qa_rounds) == 0:
                return _fallback_questions(ctx.section_target or "unnamed")
            return []

        # Empty array → LLM is satisfied
        if len(data) == 0:
            if dbg:
                dbg.log_step(_PHASE, "ask_or_ready_parse", {"success": True, "ready": True})
            return []

        questions = [
            PlanningQuestion(
                type=q.get("type", "text"),
                prompt=q.get("prompt", ""),
                options=q.get("options", []),
                allow_custom=q.get("allow_custom", True),
            )
            for q in data
            if q.get("prompt")
        ]

        if dbg:
            dbg.log_step(_PHASE, "ask_or_ready_parse", {
                "success": True,
                "ready": False,
                "count": len(questions),
                "questions": [q.prompt for q in questions],
            })

        return questions

    # ── Build final instructions ─────────────────────────────────────────

    async def _build_instructions(
        self,
        ctx: WritingContext,
        planning_state: PlanningState,
        qa_history: str,
        dbg: Optional["WritePipelineDebugger"] = None,
    ) -> str:
        """Synthesise all gathered info into markdown instructions for the writer."""

        # Format conversation history
        if ctx.conversation_history:
            conv_text = "\n\n---\n\n".join(
                f"### Output {i+1}\n{output[:500]}..."
                if len(output) > 500
                else f"### Output {i+1}\n{output}"
                for i, output in enumerate(ctx.conversation_history)
            )
        else:
            conv_text = "(no previous outputs in this session)"

        prompt = PLANNING_BUILD_INSTRUCTIONS_PROMPT.format(
            section_target=ctx.section_target or "unnamed",
            user_message=ctx.user_message,
            section_context=ctx.section_context or "(none)",
            qa_history=qa_history or "(no Q&A — planning completed immediately)",
            initial_context=planning_state.initial_context or "(none)",
            answer_context=planning_state.answer_context or "(none — no Q&A round)",
            referenced_sections=_format_referenced_sections(ctx.referenced_sections),
            current_section=ctx.current_section or "(empty — new section)",
            previous_attempt=ctx.previous_attempt or "(none — first attempt)",
            conversation_history=conv_text,
        )

        if dbg:
            dbg.log_step(_PHASE, "build_instructions_prompt", prompt)

        messages = [
            ChatMessage(role="user", content=prompt),
        ]

        async with (dbg.llm_timer("planning", "build_instructions") if dbg else _noop_ctx()) as _t:
            response = await self._llm.achat(messages)
        instructions = (response.message.content or "").strip()

        if dbg:
            dbg.log_step(_PHASE, "build_instructions_response", instructions)

        return instructions

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _format_qa_history(planning_state: PlanningState) -> str:
        """
        Join all Q&A rounds into a single history string.

        Each element of qa_rounds is the full Q&A transcript for that round
        (questions + answers) as sent by the frontend in body.message.
        """
        if not planning_state.qa_rounds:
            return ""
        return "\n\n---\n\n".join(planning_state.qa_rounds)


# ── Module-level helpers ─────────────────────────────────────────────────

def _format_referenced_sections(sections: list[dict]) -> str:
    if not sections:
        return "(none)"
    parts = []
    for s in sections:
        parts.append(f"### {s.get('section_type', 'unknown')}\n{s.get('content', '')}")
    return "\n\n".join(parts)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _fallback_questions(section_target: str) -> list[PlanningQuestion]:
    """Generic questions when the LLM fails to generate good ones."""
    return [
        PlanningQuestion(
            type="text",
            prompt=f"What is the main focus or argument of your {section_target} section?",
        ),
        PlanningQuestion(
            type="text",
            prompt="What key points or results should be covered?",
        ),
        PlanningQuestion(
            type="text",
            prompt="Are there any specific methodologies, frameworks, or references to include?",
        ),
    ]


class _NoopCtx:
    """Async context manager that does nothing (used when dbg is None)."""
    async def __aenter__(self):
        return self
    async def __aexit__(self, *_exc):
        pass


def _noop_ctx() -> _NoopCtx:
    return _NoopCtx()
