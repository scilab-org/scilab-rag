"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Keycloak Configuration
    KEYCLOAK_SERVER_URL: str = "http://localhost:8080/"
    KEYCLOAK_REALM: str = "myrealm"
    KEYCLOAK_CLIENT_ID: str = "my-client"
    KEYCLOAK_CLIENT_SECRET: Optional[str] = None

    # PostgreSQL Database
    POSTGRESQL_URI: str = "postgresql+asyncpg://user:password@localhost:5432/your_database_name"
    
    # OpenRouter API
    OPENROUTER_API_KEY: str
    OPENROUTER_API_URL_BASE: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_URL_CHAT: str = "https://openrouter.ai/api/v1/chat/completions"

    # Frontend and gateway URLs
    FRONTEND_URL: str = "http://localhost:3000"
    GATEWAY_URL: str = "http://localhost:8080"
    
    # OpenRouter API Keys for Different Models
    OPEN_ROUTER_API_KEY_EMBED_MODEL: str
    OPEN_ROUTER_API_KEY_IMAGE_MODEL: str
    OPEN_ROUTER_API_KEY_SUMMARY_MODEL: str
    OPEN_ROUTER_API_KEY_CHAT_MODEL: str
    OPEN_ROUTER_API_KEY_EXTRACT_MODEL: str
    # Neo4j Database
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"
    
    # LLM Configuration
    LLM_MAX_TOKENS: int = 512
    LLM_CONTEXT_WINDOW: int = 2048
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: float = 60.0
    
    # Document Processing
    HYBRID_MAX_TOKENS: int = 1500
    
    # Graph extraction (ceiling for dynamic per-chunk formula)
    MAX_TRIPLETS_PER_CHUNK: int = 20

    # Auto tagging
    SUMMARY_CHUNK_SIZE: int = 1500
    SUMMARY_CHUNK_OVERLAP: int = 20
    
    # Query Engine
    SIMILARITY_TOP_K: int = 10

    # OpenRouter Models
    OPENROUTER_EMBED_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    OPENROUTER_IMAGE_MODEL: str = "openai/gpt-4o-mini"
    OPENROUTER_SUMMARY_MODEL: str = "openai/gpt-4o-mini"
    OPENROUTER_CHAT_MODEL: str = "openai/gpt-4o"
    OPENROUTER_EXTRACT_MODEL: str = "openai/gpt-4o-mini"

    # Image Model Configuration
    IMAGE_MODEL_MAX_TOKENS: int = 1024
    IMAGE_MODEL_CONTEXT_WINDOW: int = 4096
    IMAGE_MODEL_TEMPERATURE: float = 0.2

    # Summary Model Configuration
    SUMMARY_MODEL_MAX_TOKENS: int = 1024
    SUMMARY_MODEL_CONTEXT_WINDOW: int = 2048
    SUMMARY_MODEL_TEMPERATURE: float = 0.1

    # Chat Model Configuration
    CHAT_MODEL_MAX_TOKENS: int = 4096
    CHAT_MODEL_CONTEXT_WINDOW: int = 16384
    CHAT_MODEL_TEMPERATURE: float = 0.3

    # Extract Model Configuration
    EXTRACT_MODEL_MAX_TOKENS: int = 6000
    EXTRACT_MODEL_CONTEXT_WINDOW: int = 12000
    EXTRACT_MODEL_TEMPERATURE: float = 0.1
    
    HISTORY_LIMIT: int = 10

    

@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
