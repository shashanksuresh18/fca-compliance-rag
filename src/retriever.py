"""
retriever.py — Phase 2 Upgrade: Hybrid BM25 + Vector Retrieval with Cross-Encoder Re-ranking.

PHASE 2 CHANGES vs PHASE 1:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 used dense vector search only.
Phase 2 adds:
  1. BM25 keyword retrieval (Okapi BM25 over stored chunks)
  2. Reciprocal Rank Fusion (RRF) to merge vector + BM25 result lists
  3. Cross-encoder re-ranking of the fused candidates

WHY HYBRID SEARCH FOR FCA COMPLIANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dense vectors understand MEANING but struggle with exact strings.
Problem: A query like "COBS 4.2.1" — the embedding model has no
special awareness that "COBS 4.2.1" is a regulatory rule reference.
It might match "COBS" broadly instead of the exact sub-rule.

BM25 keyword search solves this: "COBS 4.2.1" scores extremely high
for chunks containing that exact string. Combined via RRF, each
retrieval method covers the other's blind spots:

  - Vector: semantic, handles paraphrasing
    ("what rules apply to retail customer suitability?")
  - BM25:   lexical, handles exact regulatory references
    ("COBS 9.2.1R", "SM&CR", "MiFID II Article 25")

CROSS-ENCODER RE-RANKING:
━━━━━━━━━━━━━━━━━━━━━━━━━
After fusion, we retrieve 20 candidates (10 from each method)
and re-rank them using a cross-encoder. Unlike bi-encoders (used for
embedding), cross-encoders see BOTH query and document together,
giving much more accurate relevance scores.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22MB — tiny, fast, runs on CPU
  - Strong MRR@10 on MS-MARCO passage ranking benchmark
  - No API key, runs locally

RECIPROCAL RANK FUSION (RRF):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each candidate, RRF score = Σ 1/(k + rank_in_each_list)
where k=60 is a smoothing constant. This combines ranks rather than
raw scores (which have different scales across methods).

PRODUCTION PITFALLS:
━━━━━━━━━━━━━━━━━━━
- BM25 index is built in-memory at startup from ChromaDB chunks.
  For very large corpora (>100k chunks), consider persisting the
  BM25 index to disk with pickle.
- Cross-encoder adds ~100-300ms latency. For sub-100ms SLAs,
  skip re-ranking or use a smaller model.
- BM25 doesn't support metadata filtering natively — we post-filter
  the BM25 results by doc_type after retrieval.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings
from src.store import get_vector_store
from src.acl_filter import filter_chunks_by_role

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level BM25 index cache
# Built once on first retrieval call, then reused
# ---------------------------------------------------------------------------
_bm25_index = None          # BM25 instance
_bm25_corpus: List[Document] = []  # The chunk documents the index is built over

# Phase 4.7: Simple result cache for identical queries (TTL: 5 min)
_retrieval_cache = {}
_cache_expiry = {}


def _build_bm25_index(doc_type_filter: Optional[str] = None):
    """
    Build (or rebuild) a BM25 index from all chunks currently in ChromaDB.

    BM25 needs access to the raw text of every stored chunk.
    We pull them all from ChromaDB once and build the in-memory index.

    Args:
        doc_type_filter: If provided, only index chunks of this type.
    """
    global _bm25_index, _bm25_corpus

    from rank_bm25 import BM25Okapi

    logger.info("Building BM25 index from ChromaDB corpus...")
    store = get_vector_store()

    # Pull all stored chunks from ChromaDB
    # ChromaDB's .get() returns dicts; max 10000 chunks per call
    try:
        collection = store._collection
        data = collection.get(include=["documents", "metadatas"])
        docs_text = data.get("documents", [])
        metadatas = data.get("metadatas", [])
    except Exception as e:
        logger.error(f"Failed to load corpus from ChromaDB for BM25: {e}")
        return

    # Build Document objects for the BM25 corpus
    corpus_docs = []
    for text, meta in zip(docs_text, metadatas):
        if doc_type_filter and meta.get("doc_type") != doc_type_filter:
            continue
        corpus_docs.append(Document(page_content=text, metadata=meta))

    if not corpus_docs:
        logger.warning("BM25 corpus is empty — no chunks found in ChromaDB.")
        return

    # Tokenise: simple whitespace split, lowercase
    # For FCA docs, this preserves rule references like "COBS" and "4.2.1"
    tokenised = [doc.page_content.lower().split() for doc in corpus_docs]

    _bm25_index = BM25Okapi(tokenised)
    _bm25_corpus = corpus_docs

    logger.info(f"  BM25 index built over {len(corpus_docs)} chunks")


def _bm25_retrieve(
    query: str,
    top_k: int,
    doc_type: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve top-k chunks using BM25 keyword scoring.

    WHY BM25 FOR FCA RULES:
    BM25 gives high scores for exact term matches. Queries like
    "COBS 4.2.1" or "Article 25 MiFID" work perfectly here because
    BM25 counts exact token occurrences — no semantic approximation.
    """
    global _bm25_index, _bm25_corpus

    # Build index if not yet built (or rebuild if corpus changed)
    if _bm25_index is None:
        _build_bm25_index(doc_type_filter=doc_type)

    if not _bm25_index or not _bm25_corpus:
        logger.warning("BM25 index unavailable — skipping BM25 retrieval.")
        return []

    # Score and rank
    tokenised_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenised_query)

    # Get top-k indices sorted by score descending
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include chunks with positive BM25 score
            doc = _bm25_corpus[idx]
            # Apply metadata filter post-retrieval for BM25
            if doc_type and doc.metadata.get("doc_type") != doc_type:
                continue
            results.append(doc)

    logger.info(f"  BM25 retrieved {len(results)} chunks")
    return results


def _reciprocal_rank_fusion(
    vector_results: List[Document],
    bm25_results: List[Document],
    k: int = 60,
) -> List[Document]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ  1 / (k + rank_i(d))
    where rank_i(d) is the position of document d in list i (1-indexed).

    WHY RRF OVER SCORE NORMALISATION:
    Vector distances and BM25 scores have completely different scales
    and distributions. RRF works purely on rank positions, making it
    scale-invariant and surprisingly robust.

    The k=60 constant smooths the steep 1/rank curve, preventing the
    #1 result from dominating too strongly.
    """
    scores: dict[str, float] = {}
    doc_store: dict[str, Document] = {}

    def _doc_id(doc: Document) -> str:
        """Stable ID for deduplication: source + page + first 50 chars."""
        src = doc.metadata.get("source_file", "")
        pg = str(doc.metadata.get("page_number", ""))
        return f"{src}:{pg}:{doc.page_content[:50]}"

    # Score from vector list (rank 1, 2, 3 ...)
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = _doc_id(doc)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_store[doc_id] = doc

    # Score from BM25 list
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = _doc_id(doc)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_store[doc_id] = doc

    # Sort by combined RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_store[doc_id] for doc_id in sorted_ids]


def _cross_encoder_rerank(
    query: str,
    candidates: List[Document],
    top_k: int,
) -> List[Document]:
    """
    Re-rank candidate chunks using a cross-encoder model.

    Cross-encoders are far more accurate than bi-encoders for relevance
    because they attend to BOTH the query and document simultaneously.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - 22MB model, runs on CPU
      - Returns a relevance score per (query, document) pair
      - Higher score = more relevant

    BANKING RELEVANCE:
    After BM25+vector fusion, we might have 20 candidates. The cross-encoder
    picks the 5 most relevant with much higher precision, reducing the
    noise in what we send to the LLM — which directly reduces hallucination risk.
    """
    if not candidates:
        return candidates

    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Build pairs of (query, chunk_text) for scoring
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = model.predict(pairs)

        # Sort by score descending and return top_k
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = [doc for _, doc in scored[:top_k]]
        logger.info(
            f"  Cross-encoder re-ranked {len(candidates)} candidates "
            f"-> top {len(reranked)} selected"
        )
        return reranked

    except Exception as e:
        logger.warning(
            f"Cross-encoder re-ranking failed ({e}). "
            f"Falling back to RRF order."
        )
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Public API — used by generator.py
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: Optional[int] = None,
    doc_type: Optional[str] = None,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    user_role: str = "analyst", # Phase 4.3: Default to lowest privilege
) -> List[Document]:
    """
    Phase 2 retrieval: Hybrid BM25 + Vector Search with Cross-Encoder Re-ranking.

    Pipeline:
      1. Dense vector search (top 2*k candidates)
      2. BM25 keyword search (top 2*k candidates)
      3. Reciprocal Rank Fusion to merge both lists
      4. Cross-encoder re-ranking to select final top_k

    Args:
        query:       The user's compliance question.
        top_k:       Final number of chunks to return (default: settings.top_k).
        doc_type:    Optional metadata filter (e.g. "handbook", "internal").
        use_hybrid:  Set False to use vector-only (Phase 1 behaviour).
        use_reranker: Set False to skip cross-encoder (faster, lower quality).

    Returns:
        List of the most relevant Document chunks, ordered by relevance.
    """
    global _retrieval_cache, _cache_expiry

    # ------------------------------------------------------------------
    # Phase 4.7: Cache check
    # ------------------------------------------------------------------
    cache_key = (query, doc_type, top_k, use_hybrid, use_reranker, user_role)
    if cache_key in _retrieval_cache:
        if time.time() < _cache_expiry.get(cache_key, 0):
            logger.info("  ✓ Retrieval Cache Hit")
            return _retrieval_cache[cache_key]
        else:
            del _retrieval_cache[cache_key]
            del _cache_expiry[cache_key]
    k = top_k or settings.top_k
    # Retrieve 2x candidates from each method to give the re-ranker
    # a rich pool to work with
    candidate_k = min(k * 2, 20)

    logger.info(
        f"Hybrid retrieval: query='{query[:70]}...', "
        f"target_k={k}, candidate_pool={candidate_k}, "
        f"hybrid={use_hybrid}, reranker={use_reranker}"
    )

    # ------------------------------------------------------------------
    # Step 1: Dense vector retrieval (with Phase 4.7 Timeout)
    # ------------------------------------------------------------------
    store = get_vector_store()
    where_filter = {"doc_type": {"$eq": doc_type}} if doc_type else None

    try:
        # ChromaDB doesn't support a direct timeout param in similarity_search 
        # but we could wrap it if needed. For now, adding log before/after.
        vector_results = store.similarity_search(
            query=query,
            k=candidate_k,
            filter=where_filter,
        )
    except Exception as e:
        logger.error(f"Vector retrieval failed: {e}")
        vector_results = []

    logger.info(f"  Vector retrieved {len(vector_results)} candidates")

    if not use_hybrid:
        # Phase 1 compatibility mode — vector only
        final = vector_results[:k]
    else:
        # ----------------------------------------------------------
        # Step 2: BM25 keyword retrieval
        # ----------------------------------------------------------
        bm25_results = _bm25_retrieve(query, top_k=candidate_k, doc_type=doc_type)

        # ----------------------------------------------------------
        # Step 3: Reciprocal Rank Fusion
        # ----------------------------------------------------------
        fused = _reciprocal_rank_fusion(vector_results, bm25_results)
        logger.info(f"  After RRF fusion: {len(fused)} unique candidates")

        # ----------------------------------------------------------
        # Step 4: Cross-encoder Re-ranking
        # ----------------------------------------------------------
        if use_reranker and fused:
            final = _cross_encoder_rerank(query, fused, top_k=k)
        else:
            final = fused[:k]

    # --------------------------------------------------------------
    # Step 5: ACL Filtering (Phase 4.3)
    # --------------------------------------------------------------
    final, dropped = filter_chunks_by_role(final, user_role)

    # Observability log — important for audit trails in banking
    logger.info(f"  Final: {len(final)} chunks returned for generation")
    for i, doc in enumerate(final):
        src = doc.metadata.get("source_file", "?")
        pg = doc.metadata.get("page_number", "?")
        logger.debug(f"  [{i+1}] {src} page {pg}: {doc.page_content[:60]}...")

    # ------------------------------------------------------------------
    # Phase 4.7: Cache store (TTL: 5 min)
    # ------------------------------------------------------------------
    _retrieval_cache[cache_key] = final
    _cache_expiry[cache_key] = time.time() + 300 # 5 minutes

    return final


def invalidate_bm25_cache():
    """
    Force BM25 index rebuild on next retrieval call.
    Call this after ingesting new documents so BM25 picks up new chunks.
    """
    global _bm25_index, _bm25_corpus, _retrieval_cache, _cache_expiry
    _bm25_index = None
    _bm25_corpus = []
    _retrieval_cache = {}
    _cache_expiry = {}
    logger.info("BM25 and result caches invalidated.")


def retrieve_with_scores(
    query: str,
    top_k: Optional[int] = None,
    doc_type: Optional[str] = None,
) -> List[tuple]:
    """
    Retrieve chunks with similarity scores (vector-only, for RAGAS eval).
    Used by Phase 3 evaluation pipeline.
    """
    k = top_k or settings.top_k
    store = get_vector_store()
    where_filter = {"doc_type": {"$eq": doc_type}} if doc_type else None

    try:
        if where_filter:
            return store.similarity_search_with_score(query=query, k=k, filter=where_filter)
        return store.similarity_search_with_score(query=query, k=k)
    except Exception as e:
        logger.error(f"Scored retrieval failed: {e}")
        return []
