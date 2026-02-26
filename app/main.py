"""Scilab-AI Service - Graph RAG API.

A FastAPI-based service for document processing and knowledge graph querying
using LlamaIndex, Neo4j, and community detection.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup: Initialize connections
    logger.info("Starting Scilab-AI Service...")
    logger.info("Neo4j URI: %s", settings.NEO4J_URI)
    
    yield
    
    # Shutdown: Cleanup temp files
    logger.info("Shutting down Scilab-AI Service...")
    


app = FastAPI(
    title="Scilab-AI Service",
    description="""
Graph RAG API for academic paper analysis.

""",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "scilab-ai"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
