"""
Dependency injection for shared resources.
"""

from functools import lru_cache
from llama_index.core import Settings as LlamaSettings
from llama_index.llms.openrouter import OpenRouter
from app.core.config import settings
from app.services.openrouter_embedding import OpenRouterEmbedding
from app.services.store import GraphRAGStore
from keycloak import KeycloakOpenID

@lru_cache
def get_llm() -> OpenRouter:
    """Get cached LLM instance."""
    llm = get_chat_llm()
    LlamaSettings.llm = llm
    return llm

@lru_cache
def get_embed_llm() -> OpenRouterEmbedding:
    """Get cached embedding LLM instance."""
    llm =OpenRouterEmbedding(
        model_name=settings.OPENROUTER_EMBED_MODEL,
        api_key=settings.OPEN_ROUTER_API_KEY_EMBED_MODEL,
        api_base=settings.OPENROUTER_API_URL_BASE,
    )
    return llm

@lru_cache
def get_image_llm() -> OpenRouter:
    """Get cached image description LLM instance."""
    llm = OpenRouter(
        api_base=settings.OPENROUTER_API_URL_BASE,
        model=settings.OPENROUTER_IMAGE_MODEL,
        api_key=settings.OPEN_ROUTER_API_KEY_IMAGE_MODEL,
        max_tokens=settings.IMAGE_MODEL_MAX_TOKENS,
        context_window=settings.IMAGE_MODEL_CONTEXT_WINDOW,
        temperature=settings.IMAGE_MODEL_TEMPERATURE,
        timeout=settings.LLM_TIMEOUT,
    )
    return llm

@lru_cache
def get_summary_llm() -> OpenRouter:
    """Get cached summary LLM instance."""
    llm = OpenRouter(
        api_base=settings.OPENROUTER_API_URL_BASE,
        model=settings.OPENROUTER_SUMMARY_MODEL,
        api_key=settings.OPEN_ROUTER_API_KEY_SUMMARY_MODEL,
        max_tokens=settings.SUMMARY_MODEL_MAX_TOKENS,
        context_window=settings.SUMMARY_MODEL_CONTEXT_WINDOW,
        temperature=settings.SUMMARY_MODEL_TEMPERATURE,
        timeout=settings.LLM_TIMEOUT,
    )
    return llm

@lru_cache
def get_chat_llm() -> OpenRouter:
    """Get cached chat LLM instance."""
    llm = OpenRouter(
        api_base=settings.OPENROUTER_API_URL_BASE,
        model=settings.OPENROUTER_CHAT_MODEL,
        api_key=settings.OPEN_ROUTER_API_KEY_CHAT_MODEL,
        max_tokens=settings.CHAT_MODEL_MAX_TOKENS,
        context_window=settings.CHAT_MODEL_CONTEXT_WINDOW,
        temperature=settings.CHAT_MODEL_TEMPERATURE,
        timeout=settings.LLM_TIMEOUT,
    )
    return llm

@lru_cache
def get_extract_llm() -> OpenRouter:
    """Get cached extract LLM instance."""
    llm = OpenRouter(
        api_base=settings.OPENROUTER_API_URL_BASE,
        model=settings.OPENROUTER_EXTRACT_MODEL,
        api_key=settings.OPEN_ROUTER_API_KEY_EXTRACT_MODEL,
        max_tokens=settings.EXTRACT_MODEL_MAX_TOKENS,
        context_window=settings.EXTRACT_MODEL_CONTEXT_WINDOW,
        temperature=settings.EXTRACT_MODEL_TEMPERATURE,
        timeout=settings.LLM_TIMEOUT,
    )
    return llm

@lru_cache
def get_graph_store() -> GraphRAGStore:
    """Get cached graph store instance (singleton for connection pooling)."""
    llm = get_summary_llm()
    graph_store = GraphRAGStore(
        llm=llm,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        url=settings.NEO4J_URI,
        database=settings.NEO4J_DATABASE
    )
    return graph_store

@lru_cache
def get_keycloak_openid() -> KeycloakOpenID:
    """Get cached KeycloakOpenID instance."""
    keycloak_openid = KeycloakOpenID(
        server_url=settings.KEYCLOAK_SERVER_URL,
        client_id=settings.KEYCLOAK_CLIENT_ID,
        realm_name=settings.KEYCLOAK_REALM,
        client_secret_key=settings.KEYCLOAK_CLIENT_SECRET,
    )
    return keycloak_openid

def init_llama_settings():
    """Initialize LlamaIndex global settings."""
    get_llm()


# ── Writing feature singletons ──────────────────────────────────────────

@lru_cache
def get_writing_orchestrator():
    """Get cached WritingOrchestrator (uses the chat LLM for classification)."""
    from app.agents.writing.orchestrator import WritingOrchestrator
    return WritingOrchestrator(llm=get_chat_llm())

@lru_cache
def get_planning_agent():
    """Get cached PlanningAgent (uses chat LLM + Graph RAG)."""
    from app.agents.writing.planning_agent import PlanningAgent
    return PlanningAgent(
        llm=get_chat_llm(),
        graph_store=get_graph_store(),
        embed_model=get_embed_llm(),
        similarity_top_k=settings.SIMILARITY_TOP_K,
    )

@lru_cache
def get_writing_agent():
    """Get cached WritingAgent (uses the chat LLM for LaTeX generation)."""
    from app.agents.writing.writing_agent import WritingAgent
    return WritingAgent(llm=get_chat_llm())

@lru_cache
def get_validation_agent():
    """Get cached ValidationAgent (uses chat LLM + Graph store for checks)."""
    from app.agents.writing.validation_agent import ValidationAgent
    return ValidationAgent(
        llm=get_chat_llm(),
        graph_store=get_graph_store(),
    )
