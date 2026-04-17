"""
Router: /chat
Handles sending messages with auto session creation.
Supports two modes: "chat" (Q&A) and "write" (paper writing).
"""

import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.api_models.request import ChatRequest
from app.api.api_models.response import ChatMessageResponse, MessageResponse
from app.auth import CurrentUser
from app.core.dependencies import get_chat_llm, get_embed_llm, get_graph_store
from app.core.config import settings
from app.db.database import get_db, AsyncSessionLocal
from app.db.repo.message_repo import ChatMessageRepository
from app.db.repo.session_repo import ChatSessionRepository
from app.agents.chat.query_engine import GraphRAGQueryEngine
from app.domain.models import ChatQuery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

DB = Annotated[AsyncSession, Depends(get_db)]

# GET /chat

@router.post("", response_model=ChatMessageResponse)
async def send_message(
    body: ChatRequest,
    user: CurrentUser,
    db: DB,
):
    session_repo = ChatSessionRepository(db)
    msg_repo = ChatMessageRepository(db)

    try:
        # 1. Resolve or create session
        if body.session_id is None:
            if not body.project_id:
                raise HTTPException(
                    status_code=422,
                    detail="projectId is required when creating a new session",
                )
            session = await session_repo.create(
                user_id=user.user_id,
                project_id=body.project_id,
                section_id=body.section_id,
                section_target=body.section_target,
            )
        else:
            session = await session_repo.get_by_id(body.session_id, user.user_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

        history = await msg_repo.get_last_n(session.id, n=settings.HISTORY_LIMIT)
        
        # 2. Persist user message
        user_msg = await msg_repo.create(
            session_id=session.id,
            role="user",
            content=body.message,
        )

        # 3. Auto-title
        if session.title == "New chat":
            from app.helpers.utils import generate_chat_title
            from app.core.dependencies import get_summary_llm
            llm = get_summary_llm()
            title = await generate_chat_title(llm, body.message)
            await session_repo.update_title(session.id, user.user_id, title)
            session.title = title

        # 4. Commit now so the user message
        await db.commit()

        # 5. Resolve paper scope
        paper_ids = body.paper_ids

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Session/message setup failed: {exc}",
        )

    # 6. Dispatch based on mode
    if body.mode == "write":
        return await _handle_write_mode(body, session, user_msg, paper_ids, db)

    # 6. LLM query (chat mode — isolated)
    try:
        query_engine = GraphRAGQueryEngine(
            graph_store=get_graph_store(),
            embed_model=get_embed_llm(),
            llm=get_chat_llm(),
        )

        query_request = ChatQuery(
            query_str=body.message,
            paper_ids=paper_ids,
            history=history,
            summary=session.context.get("summary"),
        )

        answer, paper_names = await query_engine.acustom_query(query_request)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM query failed: {exc}",
        )

    # 7. Persist assistant reply
    
    try:
        assistant_msg = await msg_repo.create(
            session_id=session.id,
            role="assistant",
            content=answer,
            msg_metadata={
                "model": settings.OPENROUTER_CHAT_MODEL,
                "sources": paper_ids,
                "paperNames": paper_names,
            },
        )

        await session_repo.touch(session.id)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save assistant message: {exc}",
        )

    try:
        return ChatMessageResponse(
            session_id=session.id,
            user_message=MessageResponse.model_validate(user_msg),
            assistant_message=MessageResponse.model_validate(assistant_msg),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Response serialization failed: {exc}",
        )


# ── Write-mode handler ──────────────────────────────────────────────────

async def _handle_write_mode(
    body: ChatRequest,
    session,
    user_msg,
    paper_ids: list[str],
    db: AsyncSession,
) -> ChatMessageResponse:
    """
    Full write-mode pipeline (v2):
        orchestrator → planning (if needed) → writing → ruleset validation
        → LaTeX validation → explain step.

    Planning state is persisted in session.context["planning_state"] and is
    NOT reset between requests — corrections and follow-ups retain context.

    Returns a ChatMessageResponse where:
    - content = structured explanation of what was written and why
    - metadata = { writing_action, writing_output, validation_summary, ... }
    """
    from app.core.dependencies import (
        get_writing_orchestrator,
        get_planning_agent,
        get_writing_agent,
        get_validation_agent,
        get_ruleset_validator,
        get_graph_store,
    )
    from app.agents.writing.models import (
        PlanningState,
        PlanningStatus,
        WritingContext,
    )
    from app.agents.writing.debug import WritePipelineDebugger
    from app.services.latex_validator import extract_citations
    from sqlalchemy import select as sa_select
    from app.db.models.chat import ChatMessage as ChatMessageModel

    msg_repo = ChatMessageRepository(db)
    session_repo = ChatSessionRepository(db)

    # ── Debug tracer (no-op when WRITING_DEBUG=false) ────────────────────
    dbg = WritePipelineDebugger.from_settings()
    dbg.set_request_info(
        session_id=session.id,
        user_id=session.user_id,
        project_id=session.project_id,
        section_id=session.section_id,
        section_target=session.section_target,
        mode="write",
        user_message=body.message,
    )

    try:
        # Validate write-mode payload
        if not body.writing:
            raise HTTPException(
                status_code=422,
                detail="writing payload is required when mode='write'",
            )

        # Build WritingContext from request
        referenced = []
        if body.writing.referenced_sections:
            referenced = [
                {"section_type": s.section_type, "content": s.content}
                for s in body.writing.referenced_sections
            ]

        # section_target now comes from ChatRequest level, not WritingPayload
        section_target = body.section_target or session.section_target

        # ── Load previous_attempt from session.context["latest_output"] ──
        context = dict(session.context) if session.context else {}
        previous_attempt = context.get("latest_output")

        dbg.log_step("ingestion", "previous_attempt_loaded", bool(previous_attempt))

        # ── Load conversation_history from last N assistant messages ──────
        # Query assistant messages that have writingOutput.content in metadata
        conversation_history: list[str] = []
        try:
            _CONV_HISTORY_LIMIT = 5
            result = await db.execute(
                sa_select(ChatMessageModel)
                .where(
                    ChatMessageModel.session_id == session.id,
                    ChatMessageModel.role == "assistant",
                )
                .order_by(ChatMessageModel.created_at.desc())
                .limit(_CONV_HISTORY_LIMIT * 2)  # fetch extra, filter in Python
            )
            recent_msgs = list(result.scalars().all())
            for m in reversed(recent_msgs):  # oldest first
                meta = m.msg_metadata or {}
                wo = meta.get("writingOutput")
                if wo and wo.get("content"):
                    conversation_history.append(wo["content"])
                    if len(conversation_history) >= _CONV_HISTORY_LIMIT:
                        break
        except Exception as exc:
            logger.warning("Failed to load conversation_history: %s", exc)

        dbg.log_step("ingestion", "conversation_history_count", len(conversation_history))

        ctx = WritingContext(
            user_message=body.message,
            section_target=section_target,
            current_section=body.writing.current_section,
            referenced_sections=referenced,
            ruleset=body.writing.ruleset,
            section_context=body.writing.section_context,
            paper_ids=paper_ids,
            previous_attempt=previous_attempt,
            conversation_history=conversation_history,
        )

        dbg.log_step("ingestion", "writing_payload", {
            "current_section": body.writing.current_section,
            "referenced_sections_count": len(referenced),
            "ruleset": body.writing.ruleset,
        })
        dbg.log_step("ingestion", "resolved_section_target", section_target)
        dbg.log_step("ingestion", "paper_ids", paper_ids)

        # Load planning state from session context
        planning_dict = context.get("planning_state")
        if planning_dict:
            planning_state = PlanningState.from_dict(planning_dict)
        else:
            planning_state = PlanningState(status=PlanningStatus.IDLE)

        dbg.log_step("ingestion", "planning_state_loaded", planning_state.to_dict())

        # ── Helper: persist planning state + latest_output to session ─────
        async def _save_context(**updates) -> None:
            for key, val in updates.items():
                context[key] = val
            await session_repo.update_context(session.id, context)

        async def _save_planning_state(state: PlanningState) -> None:
            await _save_context(planning_state=state.to_dict())

        # ── 1. Orchestrator classification ───────────────────────────────
        orchestrator = get_writing_orchestrator()
        decision = await orchestrator.classify(ctx, planning_state, dbg=dbg)
        ctx.decision = decision

        logger.info(
            "Orchestrator decision: invoke_planning=%s, reasoning=%s",
            decision.invoke_planning,
            decision.reasoning[:80],
        )

        # ── 2. Planning phase (if needed) ────────────────────────────────
        planning_agent = get_planning_agent()

        if planning_state.status == PlanningStatus.ASKING:
            # ── Branch A: user is answering planning questions (always terminal) ──
            dbg.log_step("planning", "branch", "A_answering_questions")

            plan_result = await planning_agent.process_answers(
                ctx, planning_state, body.message, dbg=dbg,
            )

            dbg.log_step("planning", "process_answers_result_action", plan_result["action"])

            # process_answers is always terminal — instructions ready
            ctx.planning_instructions = plan_result["instructions"]
            planning_state = plan_result["planning_state"]
            await _save_planning_state(planning_state)

            dbg.log_step("planning", "instructions", plan_result["instructions"])

        elif decision.invoke_planning:
            # ── Branch B: start new planning phase ───────────────────────
            dbg.log_step("planning", "branch", "B_new_planning")

            plan_result = await planning_agent.start_planning(ctx, dbg=dbg)

            dbg.log_step("planning", "start_planning_result_action", plan_result["action"])

            if plan_result["action"] == "planning_questions":
                new_state = plan_result["planning_state"]
                await _save_planning_state(new_state)

                dbg.log_step("planning", "initial_questions", plan_result["questions"])
                dbg.log_step("planning", "early_return", True)
                dbg.finalize()

                content = "Before I write this section, I need some information from you."
                metadata = {
                    "model": settings.OPENROUTER_CHAT_MODEL,
                    "writingAction": "planning_questions",
                    "questionSchema": plan_result["questions"],
                }

                assistant_msg = await msg_repo.create(
                    session_id=session.id,
                    role="assistant",
                    content=content,
                    msg_metadata=metadata,
                )
                await session_repo.touch(session.id)

                return ChatMessageResponse(
                    session_id=session.id,
                    user_message=MessageResponse.model_validate(user_msg),
                    assistant_message=MessageResponse.model_validate(assistant_msg),
                )

            # Planning completed immediately (LLM returned [] on round 1)
            ctx.planning_instructions = plan_result.get("instructions")
            planning_state = plan_result["planning_state"]
            await _save_planning_state(planning_state)

            dbg.log_step("planning", "instructions", plan_result.get("instructions"))

        else:
            # ── Branch C: planning skipped ───────────────────────────────
            dbg.log_step("planning", "branch", "C_skipped")

            # When invoke_planning=false, do NOT reuse persisted instructions.
            # planning_instructions stays None — the writing agent works from
            # the user message + current_section + previous_attempt directly.
            ctx.planning_instructions = None
            dbg.log_step("planning", "planning_instructions", None)

        # ── 3. Writing phase ─────────────────────────────────────────────
        # Populate cite_key_map so writing agent uses real BibTeX keys
        if paper_ids:
            graph_store = get_graph_store()
            ctx.cite_key_map = graph_store.resolve_cite_keys(paper_ids)
            dbg.log_step("writing", "cite_key_map", ctx.cite_key_map)

        writer = get_writing_agent()
        write_result = await writer.write(ctx, dbg=dbg)

        dbg.log_step("writing", "section_target", write_result["section_target"])
        dbg.log_step("writing", "content", write_result["content"])

        draft_content = write_result["content"]

        # ── 4. Ruleset validation (if ruleset provided) ──────────────────
        if ctx.ruleset:
            ruleset_validator = get_ruleset_validator()
            ruleset_result = await ruleset_validator.validate(
                draft_content, ctx.ruleset, dbg=dbg,
            )

            dbg.log_step("ruleset_validation", "result", {
                "has_issues": ruleset_result["has_issues"],
                "issue_count": len(ruleset_result["issues"]),
            })

            if ruleset_result["has_issues"] and ruleset_result["issues_text"]:
                # Re-run writing agent ONCE with issues appended
                logger.info("Ruleset issues found, re-running writing agent")
                rewrite_result = await writer.rewrite_with_ruleset_issues(
                    ctx, draft_content, ruleset_result["issues_text"], dbg=dbg,
                )
                draft_content = rewrite_result["content"]
                dbg.log_step("ruleset_validation", "rewrite_content", draft_content)
        else:
            dbg.log_step("ruleset_validation", "skipped", "no ruleset configured")

        # ── 5. LaTeX validation phase ────────────────────────────────────
        validator = get_validation_agent()
        val_result = await validator.validate(draft_content, ctx, dbg=dbg)

        final_content = val_result["content"]
        validation_summary = val_result["validation_summary"]

        dbg.log_step("validation", "final_summary", validation_summary)

        # ── 5b. Extract referenced paper IDs from LaTeX ──────────────────
        cited_keys = extract_citations(final_content)
        cite_key_to_paper_id: dict[str, str] = {
            v: k for k, v in ctx.cite_key_map.items()
        }
        referenced_paper_ids: list[str] = list({
            cite_key_to_paper_id[key]
            for key in cited_keys
            if key in cite_key_to_paper_id
        })
        dbg.log_step("writing", "referenced_paper_ids", referenced_paper_ids)

        # ── 6. Explain step ──────────────────────────────────────────────
        explanation = await writer.explain_output(ctx, final_content, dbg=dbg)
        dbg.log_step("explain", "explanation", explanation)

        # ── 7. Persist latest_output in session context ──────────────────
        # Do NOT reset planning state — it persists across the session
        await _save_context(latest_output=final_content)

        # ── 8. Build response ────────────────────────────────────────────
        content = explanation
        metadata = {
            "model": settings.OPENROUTER_CHAT_MODEL,
            "writingAction": "section_output",
            "writingOutput": {
                "sectionTarget": write_result["section_target"],
                "content": final_content,
                "referencedPaperIds": referenced_paper_ids,
            },
            "validationSummary": validation_summary,
            "sources": paper_ids,
        }

        assistant_msg = await msg_repo.create(
            session_id=session.id,
            role="assistant",
            content=content,
            msg_metadata=metadata,
        )
        await session_repo.touch(session.id)

        dbg.log_step("response", "metadata", metadata)
        dbg.finalize()

        return ChatMessageResponse(
            session_id=session.id,
            user_message=MessageResponse.model_validate(user_msg),
            assistant_message=MessageResponse.model_validate(assistant_msg),
        )

    except HTTPException:
        dbg.log_step("error", "type", "HTTPException")
        dbg.finalize()
        raise
    except Exception as exc:
        logger.exception("Write-mode pipeline failed")
        dbg.log_step("error", "type", type(exc).__name__)
        dbg.log_step("error", "detail", str(exc))
        dbg.finalize()
        raise HTTPException(
            status_code=500,
            detail=f"Write-mode pipeline failed: {exc}",
        )


# ── SSE streaming endpoint ──────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


@router.post("/stream")
async def stream_message(
    body: ChatRequest,
    user: CurrentUser,
    db: DB,
):
    """Send a message and stream the assistant's response as SSE events.

    Event types:
      - session:  {"type":"session","sessionId":"...","userMessageId":"..."}
      - delta:    {"type":"delta","content":"token text"}
      - done:     {"type":"done","assistantMessageId":"..."}
      - error:    {"type":"error","detail":"..."}
    """
    session_repo = ChatSessionRepository(db)
    msg_repo = ChatMessageRepository(db)

    # ── Pre-stream setup (session, user message, history) ────────────────
    try:
        if body.session_id is None:
            if not body.project_id:
                raise HTTPException(
                    status_code=422,
                    detail="projectId is required when creating a new session",
                )
            session = await session_repo.create(
                user_id=user.user_id,
                project_id=body.project_id,
                section_id=body.section_id,
                section_target=body.section_target,
            )
        else:
            session = await session_repo.get_by_id(body.session_id, user.user_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

        history = await msg_repo.get_last_n(session.id, n=settings.HISTORY_LIMIT)

        user_msg = await msg_repo.create(
            session_id=session.id,
            role="user",
            content=body.message,
        )

        if session.title == "New chat":
            from app.helpers.utils import generate_chat_title
            from app.core.dependencies import get_chat_llm
            llm = get_chat_llm()
            title = await generate_chat_title(llm, body.message)
            await session_repo.update_title(session.id, user.user_id, title)
            session.title = title

        # Commit so the user message gets its own timestamp before the
        # assistant message is inserted in the post-stream transaction.
        await db.commit()

        paper_ids = body.paper_ids

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Session/message setup failed: {exc}",
        )

    # ── Capture values for the generator closure ─────────────────────────
    session_id = session.id
    user_message_id = user_msg.id
    summary = session.context.get("summary")

    query_request = ChatQuery(
        query_str=body.message,
        paper_ids=paper_ids,
        history=history,
        summary=summary,
    )

    async def event_generator():
        """SSE generator — streams tokens, then persists the full answer."""
        # Emit session info immediately
        yield _sse_event({
            "type": "session",
            "sessionId": str(session_id),
            "userMessageId": str(user_message_id),
        })

        collected_tokens: list[str] = []
        paper_names: dict[str, str] = {}

        try:
            query_engine = GraphRAGQueryEngine(
                graph_store=get_graph_store(),
                embed_model=get_embed_llm(),
                llm=get_chat_llm(),
            )

            token_generator, paper_names = await query_engine.astream_query(query_request)
            async for token in token_generator:
                collected_tokens.append(token)
                yield _sse_event({"type": "delta", "content": token})

        except Exception as exc:
            logger.exception("LLM streaming failed")
            yield _sse_event({"type": "error", "detail": str(exc)})
            return

        # Persist the full answer after stream completes
        full_answer = "".join(collected_tokens).strip()

        try:
            async with AsyncSessionLocal() as db_post:
                post_msg_repo = ChatMessageRepository(db_post)
                post_session_repo = ChatSessionRepository(db_post)

                assistant_msg = await post_msg_repo.create(
                    session_id=session_id,
                    role="assistant",
                    content=full_answer,
                    msg_metadata={
                        "model": settings.OPENROUTER_CHAT_MODEL,
                        "sources": paper_ids,
                        "paperNames": paper_names,
                    },
                )
                await post_session_repo.touch(session_id)
                await db_post.commit()

            yield _sse_event({
                "type": "done",
                "assistantMessageId": str(assistant_msg.id),
                "paperNames": paper_names,
            })

        except Exception as exc:
            logger.exception("Failed to persist streamed answer")
            yield _sse_event({"type": "error", "detail": f"Save failed: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
