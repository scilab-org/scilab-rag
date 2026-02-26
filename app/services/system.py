"""
System service layer for database operations.
"""

import logging
from typing import Optional

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import SystemInfo


logger = logging.getLogger(__name__)


async def check_db_connection(db: AsyncSession) -> bool:
    """
    Execute a simple query to verify database connectivity.

    Args:
        db: Async database session

    Returns:
        True if database is reachable, False otherwise
    """
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        return True
    except Exception as e:
        logger.error("Database connection check failed: %s", e)
        return False


async def get_system_info(db: AsyncSession, key: str) -> Optional[SystemInfo]:
    """
    Retrieve a system info entry by key.

    Args:
        db: Async database session
        key: The key to look up

    Returns:
        SystemInfo if found, None otherwise
    """
    stmt = select(SystemInfo).where(SystemInfo.key == key)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def upsert_system_info(
    db: AsyncSession,
    key: str,
    value: Optional[str] = None,
) -> SystemInfo:
    """
    Create or update a system info entry.

    Args:
        db: Async database session
        key: The key to upsert
        value: The value to set

    Returns:
        The created or updated SystemInfo
    """
    existing = await get_system_info(db, key)

    if existing:
        existing.value = value
        await db.commit()
        await db.refresh(existing)
        return existing

    new_entry = SystemInfo(key=key, value=value)
    db.add(new_entry)
    await db.commit()
    await db.refresh(new_entry)
    return new_entry
