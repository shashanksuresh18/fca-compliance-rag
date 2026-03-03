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

from fastapi import FastAPI, HTTPException, Header, Security, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.graph import rag_graph
from src.ingest import ingest_directory
from src.otel import setup_otel
from src.store import get_collection_stats, get_vector_store
from src.pii import mask_pii, has_pii
from src.injection_guard import check_query_for_injection
from src.auth import get_current_user, UserContext, verify_admin
from src.audit_log import audit_logger
from src.utils import Timer

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FCA Compliance RAG API",
    description=(
        "Barclays-Grade Compliance Assistant — Phase 4 Hardened.\n\n"
        "Security: JWT RBAC, PII masking, injection blocking.\n"
        "Reliability: LangGraph orchestration, evidence grading, audit logging."
    ),
    version="4.0.0",
    redoc_url="/redoc",
)

# Phase 5: Serve the Premium UI
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to the UI."""
    return RedirectResponse(url="/static/index.html")


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
    # --- Phase 2 retrieval controls ---
    use_hybrid: Optional[bool] = Field(
        default=None,
        description="Phase 2: Enable BM25 + vector hybrid retrieval. Defaults to settings.use_hybrid.",
    )
    use_reranker: Optional[bool] = Field(
        default=None,
        description="Phase 2: Enable cross-encoder re-ranking. Defaults to settings.use_reranker.",
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
    retrieval_strategy: Optional[str] = Field(default=None, description="Phase 2: hybrid or vector-only")
    evidence_support_rate: Optional[float] = Field(
        default=None,
        description="Phase 4: Fraction of answer claims grounded in source documents (1.0 = no hallucination)"
    )
    pii_detected: bool = Field(
        default=False,
        description="Phase 4: True if PII was detected in the query (masked before processing)"
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Phase 4: Unique trace ID for audit logging and debugging"
    )


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
async def query(
    request: QueryRequest,
    user: UserContext = Security(get_current_user)
):
    """
    Ask a compliance question grounded in FCA documentation.

    **Phase 4 pipeline**:
    1. JWT Auth — verifies user role (admin, compliance, analyst)
    2. PII scan — detects + masks UK PII in query before logging
    3. Injection guard — blocks prompt manipulation attempts
    4. Retrieval ACLs — filters chunks based on user role
    5. Evidence grader — atomic claim verification (Guardrail 3)
    6. Audit Logging — structured trace recorded for every request
    """
    # -------------------------------------------------------------------
    # Phase 4 Wave 3: LangGraph Orchestration
    # -------------------------------------------------------------------
    # The graph handles PII, injection, retrieval, generation, and grading.
    # We pass the full state and let the graph execute the nodes.
    
    initial_state = {
        "question": request.question,
        "user_role": user.role,
        "doc_type": request.doc_type,
        "top_k": request.top_k or settings.top_k,
        "use_hybrid": request.use_hybrid if request.use_hybrid is not None else settings.use_hybrid,
        "use_reranker": request.use_reranker if request.use_reranker is not None else settings.use_reranker,
        "pii_detected": False,
        "is_injection": False,
        "chunks": [],
        "raw_answer": "",
        "graded_answer": None
    }

    with Timer() as timer:
        try:
            # Invoke the compiled state machine
            final_state = rag_graph.invoke(initial_state)
            result = final_state["final_output"]
        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    # Add retrieval strategy for response metadata
    use_hybrid = initial_state["use_hybrid"]
    use_reranker = initial_state["use_reranker"]
    strategy = "hybrid-reranked" if use_hybrid and use_reranker else \
               "hybrid" if use_hybrid else "vector-only"
    result["retrieval_strategy"] = strategy
    
    # Trace ID from Audit Log
    # Note: We use safe_question (masked) for the audit log
    safe_question, _ = mask_pii(request.question)
    
    trace_id = audit_logger.log_event(
        user_id=user.user_id,
        user_role=user.role,
        query_masked=safe_question,
        response_data=result,
        latency_ms=timer.interval
    )
    result["trace_id"] = trace_id

    # Clean up verbose fields for the API response
    result.pop("evidence_claims", None)
    result.pop("rejected_answer", None)

    return QueryResponse(**result)


@app.post("/ingest", response_model=IngestResponse, tags=["Administration"])
async def ingest(
    request: IngestRequest,
    admin_user: UserContext = Security(verify_admin),
):
    """
    Ingest a directory of PDF and Markdown documents into ChromaDB.

    **Protected endpoint**: Requires a JWT with 'admin' role.
    """

    logger.info(f"Ingest request: dir='{request.directory}', type='{request.doc_type}'")

    try:
        chunks_stored = ingest_directory(request.directory, request.doc_type)
        # Phase 2: Invalidate BM25 cache so new chunks are included in keyword search
        from src.retriever import invalidate_bm25_cache
        invalidate_bm25_cache()
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
