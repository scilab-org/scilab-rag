"""
Router: /chat
Handles sending messages with auto session creation.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.api_models.request import ChatRequest
from app.api.api_models.response import ChatMessageResponse, MessageResponse
from app.auth import CurrentUser
from app.core.dependencies import get_chat_llm, get_embed_llm, get_graph_store
from app.core.config import settings
from app.db.database import get_db
from app.db.repo.message_repo import ChatMessageRepository
from app.db.repo.session_repo import ChatSessionRepository
from app.agents.chat.query_engine import GraphRAGQueryEngine
from app.domain.models import ChatQuery

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
            title = body.message[:60].strip() + ("..." if len(body.message) > 60 else "")
            await session_repo.update_title(session.id, user.user_id, title)
            session.title = title

        # 5. Resolve paper scope
        from app.helpers.mock import get_paper_ids_by_project
        paper_ids = get_paper_ids_by_project(session.project_id)
        if not paper_ids:
            raise HTTPException(
                status_code=422,
                detail="No papers found for this project.",
            )

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

        answer = await query_engine.acustom_query(query_request)

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