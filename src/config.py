"""
config.py — Centralised configuration for the FCA RAG system (Free Stack).

WHY THIS MATTERS IN BANKING:
- Secrets (API keys) must NEVER be hardcoded in source code.
  In a regulated environment, this is both a security policy
  violation and a potential audit finding.
- Pydantic Settings validates all config at startup, so a
  misconfigured deployment fails immediately rather than
  silently at runtime.

FREE STACK CHOICES:
- Embeddings: sentence-transformers all-MiniLM-L6-v2 (local, no API)
- LLM: Groq API with llama-3.3-70b-versatile (free tier, no credit card)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.
    All fields are validated at startup — fail fast on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore unknown env vars — safe in shared environments
    )

    # --- Groq (FREE LLM) ---
    # Sign up at https://console.groq.com — no credit card needed
    groq_api_key: str = Field(..., description="Groq API key (free tier)")
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model name. Options: llama-3.3-70b-versatile, llama-3.1-8b-instant",
    )

    # --- Embeddings: LOCAL, zero cost ---
    # Model is ~90MB, downloaded once and cached in ~/.cache/huggingface
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformers model (runs locally, no API key needed)",
    )

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Directory where ChromaDB persists data",
    )
    chroma_collection_name: str = Field(
        default="fca_compliance",
        description="ChromaDB collection name",
    )

    # --- Retrieval ---
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve per query",
    )

    # --- Chunking ---
    chunk_size: int = Field(
        default=600,
        description="Target chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap between consecutive chunks in characters",
    )

    # --- Prompt versioning ---
    prompt_version: str = Field(
        default="v1",
        description="Which prompt version folder to load from prompts/",
    )

    # --- API Security ---
    api_secret_key: str = Field(
        default="change-me-in-production",
        description="Bearer token for protecting admin endpoints",
    )


# Singleton — import this everywhere rather than re-instantiating
settings = Settings()
