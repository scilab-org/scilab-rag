"""
Repository for ChatSession — all DB operations, no business logic.
"""

import uuid
from typing import Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat import ChatSession


class ChatSessionRepository:

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        user_id: str,
        title: str = "New chat",
        project_id: Optional[str] = None,
    ) -> ChatSession:
        session = ChatSession(
            user_id=user_id,
            title=title,
            project_id=project_id,
        )
        self._db.add(session)
        await self._db.flush()
        await self._db.refresh(session)
        return session

    async def get_by_id(
        self,
        session_id: uuid.UUID,
        user_id: str,
    ) -> Optional[ChatSession]:
        """Fetch a session, enforcing user ownership."""
        result = await self._db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChatSession]:
        """Return sessions ordered by most recently updated."""
        result = await self._db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update_title(
        self,
        session_id: uuid.UUID,
        user_id: str,
        title: str,
    ) -> Optional[ChatSession]:
        await self._db.execute(
            update(ChatSession)
            .where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
            .values(title=title)
        )
        return await self.get_by_id(session_id, user_id)

    async def update_context(
        self,
        session_id: uuid.UUID,
        context: dict,
    ) -> None:
        """Update the JSONB context field (summary, summary_at)."""
        await self._db.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(context=context)
        )

    async def touch(self, session_id: uuid.UUID) -> None:
        """Bump updated_at after a new message — keeps list ordering fresh."""
        await self._db.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(updated_at=func.now())
        )

    async def delete(
        self,
        session_id: uuid.UUID,
        user_id: str,
    ) -> bool:
        """Returns True if a row was deleted, False if not found."""
        result = await self._db.execute(
            delete(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
        )
        return result.rowcount > 0