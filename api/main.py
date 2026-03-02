"""
main.py — FastAPI server for the FCA Compliance RAG system.

ENDPOINTS:
  GET  /health        — System health check (ChromaDB connection + stats)
  POST /query         — Ask a compliance question, get a cited answer
  POST /ingest        — Ingest a directory of documents into ChromaDB
                        (protected by API key in production)

WHY FASTAPI FOR BANKING:
━━━━━━━━━━━━━━━━━━━━━━━
- Automatic OpenAPI/Swagger docs at /docs — easy for bank IT teams
  to understand and integrate the service.
- Pydantic models enforce strict request/response schemas, preventing
  malformed inputs reaching the RAG pipeline.
- Async-ready: in Phase 2 we can parallelise retrieval + re-ranking
  without blocking the event loop.

PRODUCTION PITFALLS:
━━━━━━━━━━━━━━━━━━━
- The /ingest endpoint MUST be behind authentication in production.
  An exposed ingest endpoint allows an adversary to poison your
  vector store with malicious documents (prompt injection via RAG).
- All queries and answers should be logged to an immutable audit
  store (S3 / Azure Blob) with timestamps for regulatory compliance.
- Rate limiting is essential: Groq free tier has 14,400 req/day.
  Add slowapi rate limiting middleware in production.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings
from src.generator import generate_answer
from src.ingest import ingest_directory
from src.store import get_collection_stats

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FCA Compliance RAG API",
    description=(
        "Production-grade Retrieval-Augmented Generation system for FCA regulatory "
        "compliance documents. All answers are grounded in FCA documentation with "
        "mandatory citations. Hallucination is guarded against at the architecture level."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)


# ---------------------------------------------------------------------------
# Request / Response models (Pydantic — strict typing)
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The compliance question to answer",
        example="What are the conduct rules under SM&CR?"
    )
    doc_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional: Filter search to a specific document type. "
            "Options: handbook, policy, guidance, internal, basel, mifid"
        ),
        example="handbook"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=15,
        description="Number of document chunks to retrieve (default: 5)",
    )


class CitationItem(BaseModel):
    source_file: str
    page_number: int | str
    doc_type: str


class QueryResponse(BaseModel):
    answer: str = Field(description="Grounded answer with FCA citations, or decline message")
    citations: list[CitationItem] = Field(description="Source documents retrieved for this answer")
    declined: bool = Field(description="True if the system declined to answer (insufficient evidence)")
    decline_reason: Optional[str] = Field(default=None, description="Why the system declined, if applicable")
    prompt_version: str = Field(description="Prompt template version used for this query")
    chunks_retrieved: int = Field(description="Number of document chunks used as context")


class IngestRequest(BaseModel):
    directory: str = Field(
        ...,
        description="Absolute or relative path to the directory containing documents",
        example="data/mock"
    )
    doc_type: str = Field(
        ...,
        description="Document type label for all files in this directory",
        example="handbook"
    )


class IngestResponse(BaseModel):
    status: str
    chunks_stored: int
    directory: str
    doc_type: str


class HealthResponse(BaseModel):
    status: str
    collection: str
    total_chunks: int
    embedding_model: str
    llm_model: str


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Verifies ChromaDB is accessible and returns collection stats.
    Use this in your Kubernetes/Docker liveness probes.
    """
    stats = get_collection_stats()
    if "error" in stats:
        raise HTTPException(status_code=503, detail=f"ChromaDB unavailable: {stats['error']}")

    return HealthResponse(
        status="healthy",
        collection=stats.get("collection", "unknown"),
        total_chunks=stats.get("total_chunks", 0),
        embedding_model=stats.get("embedding_model", "unknown"),
        llm_model=settings.llm_model,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Ask a compliance question grounded in FCA documentation.

    The system will:
    1. Retrieve the top-k most relevant FCA document chunks
    2. Generate an answer using Groq (llama-3.3-70b)
    3. Enforce that the answer contains document citations
    4. If no evidence found, return a structured decline message

    **In a banking context**: Every answer includes the exact source
    document and page number, enabling compliance officers to verify
    the answer against the primary source.
    """
    logger.info(f"Query received: '{request.question[:80]}'...")

    try:
        result = generate_answer(
            question=request.question,
            doc_type=request.doc_type,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    return QueryResponse(**result)


@app.post("/ingest", response_model=IngestResponse, tags=["Administration"])
async def ingest(
    request: IngestRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Ingest a directory of PDF and Markdown documents into ChromaDB.

    **Protected endpoint**: Requires X-API-Key header matching the
    server's API_SECRET_KEY setting in production.

    IMPORTANT: In production, restrict this endpoint to internal
    network only — never expose it on a public-facing API gateway.
    """
    # Simple API key check — upgrade to OAuth2 / JWT in production
    if settings.api_secret_key != "change-me-in-production":
        if x_api_key != settings.api_secret_key:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: valid X-API-Key header required for /ingest"
            )

    logger.info(f"Ingest request: dir='{request.directory}', type='{request.doc_type}'")

    try:
        chunks_stored = ingest_directory(request.directory, request.doc_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")

    return IngestResponse(
        status="success",
        chunks_stored=chunks_stored,
        directory=request.directory,
        doc_type=request.doc_type,
    )


# ---------------------------------------------------------------------------
# Run directly for local dev
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
