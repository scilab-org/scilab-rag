from fastapi import APIRouter

from app.api.routers import papers, rag, system


api_router = APIRouter()

api_router.include_router(papers.router)
api_router.include_router(rag.router)
api_router.include_router(system.router)
    