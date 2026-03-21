"""
Router: /chat/sessions
Handles session CRUD and message history.
All endpoints require a valid Keycloak JWT.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.api_models.request import SessionRenameRequest

from app.api.api_models.response import (
    MessageListResponse,
    MessageResponse,
    SessionListResponse,
    SessionRenameResponse,
    SessionResponse,
)
from app.auth import CurrentUser
from app.db.database import get_db
from app.db.repo.message_repo import ChatMessageRepository
from app.db.repo.session_repo import ChatSessionRepository

router = APIRouter(prefix="/sessions", tags=["Chat Sessions"])

DB = Annotated[AsyncSession, Depends(get_db)]


# ── GET /sessions ────────────────────────────────────────────────────────

@router.get("", response_model=SessionListResponse)
async def list_sessions(
    user: CurrentUser,
    db: DB,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all sessions for the authenticated user, most recent first."""
    repo = ChatSessionRepository(db)
    sessions = await repo.list_by_user(
        user_id=user.user_id,
        limit=limit,
        offset=offset,
    )
    return SessionListResponse(
        sessions=[SessionResponse.model_validate(s) for s in sessions],
        total=len(sessions),
    )

# ── PATCH /sessions/{id} ─────────────────────────────────────────────────

@router.patch("/{session_id}", response_model=SessionRenameResponse)
async def rename_session(
    session_id: uuid.UUID,
    body: SessionRenameRequest,
    user: CurrentUser,
    db: DB,
):
    """Rename a session."""
    repo = ChatSessionRepository(db)
    session = await repo.update_title(session_id, user.user_id, body.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionRenameResponse.model_validate(session)


# ── DELETE /sessions/{id} ────────────────────────────────────────────────

@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    user: CurrentUser,
    db: DB,
):
    """Delete a session and all its messages (CASCADE)."""
    repo = ChatSessionRepository(db)
    deleted = await repo.delete(session_id, user.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")


# ── GET /sessions/{id}/messages ─────────────────────────────────────────

@router.get("/{session_id}/messages", response_model=MessageListResponse)
async def list_messages(
    session_id: uuid.UUID,
    user: CurrentUser,
    db: DB,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return paginated message history for a session."""
    # Verify ownership first
    session_repo = ChatSessionRepository(db)
    session = await session_repo.get_by_id(session_id, user.user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    msg_repo = ChatMessageRepository(db)
    messages = await msg_repo.list_by_session(
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
    total = await msg_repo.count_by_session(session_id)
    return MessageListResponse(
        messages=[MessageResponse.model_validate(m) for m in messages],
        total=total,
    )
    
