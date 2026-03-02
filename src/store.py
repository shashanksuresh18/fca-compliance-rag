"""
store.py — ChromaDB embedding store (Free Stack: local HuggingFace embeddings).

FREE STACK:
- Embeddings: sentence-transformers all-MiniLM-L6-v2
  - Runs 100% locally on CPU — no API key, no cost
  - 90MB model, downloaded once to ~/.cache/huggingface
  - 384-dimensional embeddings, strong semantic similarity for English legal text

WHY THIS MATTERS IN BANKING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PERSISTENCE: ChromaDB persists to disk so your embedded FCA corpus
   survives API restarts — you don't re-embed on every startup.

2. DEDUPLICATION: Re-ingesting the same document without ID-based
   dedup creates duplicate chunks that bias retrieval scores.
   We hash content + source + page to generate stable chunk IDs.

3. METADATA: doc_type, source_file, page_number stored per chunk
   enables Phase 2 metadata filtering (e.g. "handbook only").

PRODUCTION PITFALLS:
━━━━━━━━━━━━━━━━━━━
- If you ever change the embedding model, existing embeddings are
  incompatible. Rename the collection (e.g. "fca_compliance_minilm_v2")
  and re-ingest everything.
- For >1M chunks, consider Pinecone, Weaviate, or pgvector.
"""

import hashlib
import logging
import sys
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings

logger = logging.getLogger(__name__)

# Module-level cache so we don't reload the ~90MB model on every call
_embeddings_cache = None


def _build_embeddings() -> HuggingFaceEmbeddings:
    """
    Build (or return cached) local HuggingFace embedding function.

    all-MiniLM-L6-v2 is a strong general-purpose model with good
    performance on English legal/regulatory text. It runs on CPU
    in ~50ms per query — fast enough for real-time RAG.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        logger.info(f"Loading embedding model '{settings.embedding_model}' locally...")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},   # Change to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},  # Cosine sim friendly
        )
        logger.info("  ✓ Embedding model loaded (cached for reuse)")
    return _embeddings_cache


def get_vector_store() -> Chroma:
    """
    Return (or create) the persisted ChromaDB vector store.
    Safe to call multiple times — returns existing collection if present.
    """
    embeddings = _build_embeddings()
    store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    return store


def _make_chunk_id(chunk: Document) -> str:
    """
    Generate a stable, deterministic ID for each chunk.
    Hashing content + source + page ensures re-ingesting the same
    document is idempotent — no duplicate chunks created.
    """
    content = chunk.page_content
    source = chunk.metadata.get("source_file", "unknown")
    page = str(chunk.metadata.get("page_number", 0))
    chunk_idx = str(chunk.metadata.get("chunk_index", 0))
    raw = f"{source}::{page}::{chunk_idx}::{content[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def add_documents_to_store(chunks: List[Document]) -> None:
    """
    Upsert chunks into ChromaDB with deduplication.

    Args:
        chunks: List of LangChain Documents (with metadata) to store.
    """
    if not chunks:
        logger.warning("add_documents_to_store called with empty chunk list.")
        return

    store = get_vector_store()
    ids = [_make_chunk_id(chunk) for chunk in chunks]

    # Check for existing IDs to avoid re-embedding (saves time, idempotent)
    existing_ids = set()
    try:
        existing_data = store.get(ids=ids)
        existing_ids = set(existing_data.get("ids", []))
    except Exception:
        pass  # Empty collection on first run — this is fine

    new_chunks = [c for c, cid in zip(chunks, ids) if cid not in existing_ids]
    new_ids = [cid for cid in ids if cid not in existing_ids]

    if not new_chunks:
        logger.info("All chunks already exist in store — skipping (dedup).")
        return

    logger.info(
        f"Embedding {len(new_chunks)} new chunks locally "
        f"({len(existing_ids)} duplicates skipped)..."
    )
    store.add_documents(documents=new_chunks, ids=new_ids)
    logger.info(
        f"  ✓ Stored {len(new_chunks)} chunks in collection "
        f"'{settings.chroma_collection_name}'"
    )


def get_collection_stats() -> dict:
    """
    Return basic stats about the vector store.
    Used by the /health endpoint and monitoring.
    """
    store = get_vector_store()
    try:
        count = store._collection.count()
        return {
            "collection": settings.chroma_collection_name,
            "total_chunks": count,
            "embedding_model": settings.embedding_model,
            "embedding_type": "local (sentence-transformers)",
            "persist_dir": settings.chroma_persist_dir,
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {"error": str(e)}
