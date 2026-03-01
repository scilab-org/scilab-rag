"""
Dependency injection for shared resources.
"""

from functools import lru_cache
from typing import Generator

from llama_index.core import Settings as LlamaSettings
from llama_index.llms.openrouter import OpenRouter

from app.core.config import settings
from app.services.store import GraphRAGStore


@lru_cache
def get_llm() -> OpenRouter:
    """Get cached LLM instance."""
    llm = OpenRouter(
        api_base=settings.OPENROUTER_API_URL_BASE,
        model=settings.LLM_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        max_tokens=settings.LLM_MAX_TOKENS,
        context_window=settings.LLM_CONTEXT_WINDOW,
        temperature=settings.LLM_TEMPERATURE,
        timeout=settings.LLM_TIMEOUT,
    )
    # Set as global default
    LlamaSettings.llm = llm
    return llm


@lru_cache
def get_graph_store() -> GraphRAGStore:
    """Get cached graph store instance (singleton for connection pooling)."""
    llm = get_llm()
    graph_store = GraphRAGStore(
        llm=llm,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        url=settings.NEO4J_URI,
        database=settings.NEO4J_DATABASE
    )
    return graph_store


def init_llama_settings():
    """Initialize LlamaIndex global settings."""
    get_llm()
