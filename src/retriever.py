"""
retriever.py — Vector retrieval from ChromaDB for FCA compliance queries.

WHY THIS MATTERS IN BANKING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dense vector retrieval finds semantically similar content even when
the user doesn't use the exact regulatory terminology. For example:
  - Query: "rules about selling products to vulnerable customers"
  - Retrieves chunks about: COBS 4.2.1, Consumer Duty, MCOB 5A
  (even though the user never said "COBS" or "Consumer Duty")

This semantic understanding is the core advantage over keyword search
(BM25) alone — hybrid retrieval in Phase 2 combines both.

METADATA FILTERING:
━━━━━━━━━━━━━━━━━━
In a bank, different teams need different document scopes:
  - Legal team: "only search the FCA Handbook"
  - Product team: "only search Policy Statements"
  - Compliance: "search everything"
We support this via optional doc_type filtering.

PRODUCTION PITFALLS:
━━━━━━━━━━━━━━━━━━━
- Top-K sensitivity: Too few (k=3) misses relevant chunks for
  multi-part compliance questions. Too many (k=10) sends irrelevant
  context to the LLM, increasing hallucination risk.
  k=5 is a good default; expose it as a tunable parameter.
- Score thresholding: Add a minimum similarity score threshold to
  reject truly irrelevant chunks rather than passing them to the LLM.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings
from src.store import get_vector_store

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    doc_type: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        query:    The user's compliance question.
        top_k:    Number of chunks to return. Defaults to settings.top_k.
        doc_type: Optional filter. If provided, only chunks with this
                  doc_type metadata value are searched.
                  E.g. "policy", "handbook", "internal", "mifid"

    Returns:
        List of Document objects, each with metadata:
            - source_file: filename of the source document
            - page_number: page in the original document
            - doc_type:    type label for the document
            - chunk_index: position within the source document

    IMPORTANT: Returns empty list (not an exception) if nothing is found.
    The generator layer handles the empty-result case with a decline message.
    """
    k = top_k or settings.top_k
    store = get_vector_store()

    logger.info(f"Retrieving top-{k} chunks for query: '{query[:80]}...'")

    # Build optional metadata filter
    # ChromaDB uses a `where` dict with MongoDB-style operators
    where_filter: Optional[dict] = None
    if doc_type:
        where_filter = {"doc_type": {"$eq": doc_type}}
        logger.info(f"  Filter applied: doc_type = '{doc_type}'")

    try:
        if where_filter:
            results = store.similarity_search(
                query=query,
                k=k,
                filter=where_filter,
            )
        else:
            results = store.similarity_search(
                query=query,
                k=k,
            )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

    logger.info(f"  ✓ Retrieved {len(results)} chunks")

    # Log retrieved sources for observability (useful in audit logs)
    for i, doc in enumerate(results):
        src = doc.metadata.get("source_file", "unknown")
        pg = doc.metadata.get("page_number", "?")
        logger.debug(f"  [{i+1}] {src} (page {pg}): {doc.page_content[:80]}...")

    return results


def retrieve_with_scores(
    query: str,
    top_k: Optional[int] = None,
    doc_type: Optional[str] = None,
) -> List[tuple[Document, float]]:
    """
    Retrieve chunks with their similarity scores.
    Scores are used in Phase 2 for re-ranking and threshold filtering.

    Returns:
        List of (Document, score) tuples. Lower score = more similar
        in ChromaDB's L2 distance metric (0.0 = perfect match).
    """
    k = top_k or settings.top_k
    store = get_vector_store()

    where_filter: Optional[dict] = None
    if doc_type:
        where_filter = {"doc_type": {"$eq": doc_type}}

    try:
        if where_filter:
            results = store.similarity_search_with_score(query=query, k=k, filter=where_filter)
        else:
            results = store.similarity_search_with_score(query=query, k=k)
        return results
    except Exception as e:
        logger.error(f"Scored retrieval failed: {e}")
        return []
