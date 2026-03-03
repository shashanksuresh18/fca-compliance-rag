"""
config.py — Phase 2 settings for the FCA RAG system (Free Stack).

Phase 2 adds:
  - use_hybrid: toggle BM25 + vector hybrid retrieval
  - use_reranker: toggle cross-encoder re-ranking
  - reranker_model: which cross-encoder to use
  - prompt_version: now defaults to v2 (chain-of-thought)
  - candidate_k_multiplier: how many candidates to pre-fetch per retrieval method
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Groq (FREE LLM) ---
    groq_api_key: str = Field(..., description="Groq API key (console.groq.com)")
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model. Options: llama-3.3-70b-versatile, llama-3.1-8b-instant",
    )

    # --- Embeddings: LOCAL, zero cost ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformers model (runs locally, no API key)",
    )

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(default="./chroma_db")
    chroma_collection_name: str = Field(default="fca_compliance")

    # --- Retrieval (Phase 2) ---
    top_k: int = Field(default=5, ge=1, le=20)

    # Hybrid retrieval: combine BM25 keyword + dense vector search
    # Set to False for Phase 1 (vector-only) behaviour
    use_hybrid: bool = Field(
        default=True,
        description="Enable BM25 + vector hybrid retrieval (Phase 2)",
    )

    # Cross-encoder re-ranking: re-scores fused candidates for higher precision
    # Adds ~100-300ms latency on CPU. Set False for speed over quality.
    use_reranker: bool = Field(
        default=True,
        description="Enable cross-encoder re-ranking (Phase 2)",
    )

    # Cross-encoder model — 22MB, runs on CPU, no API key
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Sentence-transformers CrossEncoder model for re-ranking",
    )

    # How many candidates to fetch per method before fusion + re-ranking
    # e.g. top_k=5, multiplier=2 → fetch 10 from vector + 10 from BM25
    candidate_k_multiplier: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Candidate pool multiplier for hybrid retrieval",
    )

    # --- Chunking ---
    chunk_size: int = Field(default=600)
    chunk_overlap: int = Field(default=100)

    # --- Prompt versioning ---
    # Now defaults to v2 (chain-of-thought + enhanced citations)
    prompt_version: str = Field(
        default="v2",
        description="Prompt version folder under prompts/. Change to v1 to rollback.",
    )

    # --- API Security ---
    api_secret_key: str = Field(default="change-me-in-production")


settings = Settings()
