"""
Repository for ChatMessage — all DB operations, no business logic.
"""

import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat import ChatMessage


class ChatMessageRepository:

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        session_id: uuid.UUID,
        role: str,
        content: str,
        msg_metadata: Optional[dict] = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            msg_metadata=msg_metadata or {},
        )
        self._db.add(message)
        await self._db.flush()
        await self._db.refresh(message)
        return message

    async def list_by_session(
        self,
        session_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ChatMessage]:
        """Return messages in chronological order (oldest first)."""
        result = await self._db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_last_n(
        self,
        session_id: uuid.UUID,
        n: int,
    ) -> list[ChatMessage]:
        """
        Fetch the last N messages for history injection.
        Returns them in chronological order (oldest first).
        """
        subq = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(n)
            .subquery()
        )
        from sqlalchemy.orm import aliased
        sub_alias = aliased(ChatMessage, subq)
        result = await self._db.execute(
            select(sub_alias).order_by(sub_alias.created_at.asc())
        )
        return list(result.scalars().all())

    async def count_by_session(self, session_id: uuid.UUID) -> int:
        """Total message count for a session — used to trigger summary refresh."""
        result = await self._db.execute(
            select(func.count())
            .select_from(ChatMessage)
            .where(ChatMessage.session_id == session_id)
        )
        return result.scalar_one()