"""
Router: /chat
Handles sending messages with auto session creation.
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

    # 6. LLM query (isolated)
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