from fastapi import APIRouter, Depends

from app.api.routers import graph, papers, sessions, chat, formatter
from app.auth import get_current_user

# api_router = APIRouter(dependencies=[Depends(get_current_user)])
api_router = APIRouter()

api_router.include_router(papers.router)
api_router.include_router(graph.router)
api_router.include_router(sessions.router)
api_router.include_router(chat.router)
api_router.include_router(formatter.router)

    