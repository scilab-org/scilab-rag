"""
Application configuration using Pydantic Settings.
"""

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
    
    # OpenRouter API
    OPENROUTER_API_KEY: str
    OPENROUTER_API_URL_BASE: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_URL_CHAT: str = "https://openrouter.ai/api/v1/chat/completions"

    # Neo4j Database
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"
    
    # LLM Configuration
    LLM_MODEL: str = "openai/gpt-4o-mini"
    VLM_MODEL: str = "qwen/qwen-2-vl-7b-instruct"
    LLM_MAX_TOKENS: int = 512
    LLM_CONTEXT_WINDOW: int = 2048
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: float = 60.0
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Document Processing
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 50
    
    SUMMARY_CHUNK_SIZE: int = 1500
    SUMMARY_CHUNK_OVERLAP: int = 20
    MAX_TRIPLETS_PER_CHUNK: int = 8
    
    # Query Engine
    SIMILARITY_TOP_K: int = 20
    
    # Community Detection
    MAX_CLUSTER_SIZE: int = 5

    # PostgreSQL Database
    POSTGRESQL_URI: str = "postgresql+asyncpg://user:password@localhost:5432/your_database_name"

@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
