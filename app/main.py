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
from app.auth import CurrentUser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup: Initialize connections
    logger.info("Starting Scilab-AI Service ...")
    logger.info("Neo4j URI: %s", settings.NEO4J_URI)

    # --- RabbitMQ consumer ---
    from app.messaging import connection as rmq_connection
    from app.messaging.consumer import start_consumer

    try:
        await rmq_connection.connect()
        consumer_tag = await start_consumer()
        logger.info(
            "RabbitMQ consumer is running (tag=%s). "
            "Queue '%s' should be bound to exchange '%s'.",
            consumer_tag,
            settings.RABBITMQ_INGEST_QUEUE,
            settings.RABBITMQ_INGEST_EXCHANGE,
        )
    except Exception:
        logger.exception(
            "Failed to start RabbitMQ consumer – the service will still "
            "serve HTTP requests but ingestion via messaging is unavailable."
        )

    yield

    # Shutdown: close RabbitMQ
    logger.info("Shutting down Scilab-AI Service...")
    try:
        await rmq_connection.close()
    except Exception:
        logger.exception("Error closing RabbitMQ connection.")

app = FastAPI(
    title="Scilab-AI Service",
    description="""
Graph RAG API for academic paper analysis.

""",
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_init_oauth={
        "clientId": settings.KEYCLOAK_CLIENT_ID,
        "usePkceWithAuthorizationCodeGrant": True,
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, settings.GATEWAY_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/health")
async def health_check(user: CurrentUser):
    """Health check endpoint (requires authentication)."""
    return {
        "status": "healthy",
        "service": "scilab-ai",
        "user": user.username,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
