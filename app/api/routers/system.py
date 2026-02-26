"""
System router for database and infrastructure endpoints.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.api_models.response import DbStatusResponse, SystemInfoResponse
from app.api.api_models.request import SystemInfoRequest
from app.core.database import get_db
from app.services import system as system_service


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/system",
    tags=["System"],
)


@router.get("/db", response_model=DbStatusResponse)
async def check_database(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DbStatusResponse:
    """
    Check database connectivity.

    Executes a simple query to verify the database is reachable.
    """
    is_connected = await system_service.check_db_connection(db)

    if not is_connected:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed",
        )

    return DbStatusResponse(status="ok")


