"""
System service layer for database operations.
"""

import logging

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

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


