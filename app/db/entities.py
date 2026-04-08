"""
SQLAlchemy ORM models.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class ProcessedMessage(Base):
    """Idempotency guard for ``PaperIngestionEvent`` messages.

    A row is inserted (with ``paper_id`` as the primary key) **before**
    ingestion begins.  Any subsequent message carrying the same ``paper_id``
    will trigger a primary-key violation, which the consumer catches to skip
    re-processing.

    On ingestion *failure* the row is deleted so that a later retry is allowed.
    """

    __tablename__ = "processed_messages"

    paper_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<ProcessedMessage(paper_id={self.paper_id!r})>"


##Sample entity mapping
class SystemInfo(Base):
    """System information table for connectivity checks and metadata."""

    __tablename__ = "system_info"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return f"<SystemInfo(key={self.key!r})>"


