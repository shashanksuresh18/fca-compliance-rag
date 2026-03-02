"""
ingest.py — Document ingestion and chunking pipeline.

WHY THIS MATTERS IN BANKING / COMPLIANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CITATION ACCURACY: We extract page numbers at ingestion time and
   attach them to every chunk. This is non-negotiable in regulated
   environments — a compliance officer needs to verify the source.

2. CHUNK SIZE (600 tokens / 100 overlap):
   - FCA Handbook rules often reference earlier sub-sections
     (e.g. "as defined in 3.1.2R above"). Chunks that are too small
     lose this inter-rule context.
   - Chunks > 800 tokens dilute the embedding meaning, causing
     poor retrieval precision on specific rule queries.
   - 100-token overlap ensures that sentences spanning chunk
     boundaries are captured by at least one chunk.

3. SCANNED PDF DETECTION: Many older FCA documents are image PDFs.
   pypdf will silently return empty text. We log a warning per page
   so operators know which documents need OCR preprocessing.

PRODUCTION PITFALLS:
━━━━━━━━━━━━━━━━━━━
- OCR: Use pytesseract or AWS Textract for scanned docs.
- PII: Bank policy documents may contain PII. Strip before ingest
  or ensure your ChromaDB instance is in a compliant data zone.
- Encoding: FCA PDFs can have non-UTF-8 characters. We handle this
  with error='replace' in the loader.

Usage:
    python src/ingest.py --dir data/mock --doc-type policy
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Add project root to path so we can import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings
from src.store import add_documents_to_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported document types and their metadata labels
# (Used for metadata filtering in Phase 2)
# ---------------------------------------------------------------------------
VALID_DOC_TYPES = {
    "handbook": "FCA Handbook",
    "policy": "FCA Policy Statement",
    "guidance": "FCA Guidance Note",
    "internal": "Internal Bank Policy",
    "basel": "Basel III",
    "mifid": "MiFID II",
}


def load_pdf(file_path: Path, doc_type: str) -> List[Document]:
    """
    Load a PDF and split into pages.
    Each page becomes a LangChain Document with rich metadata.

    IMPORTANT: We use PyPDFLoader which loads page-by-page, giving us
    accurate page numbers for citations. The alternative (loading the
    whole PDF as one string) loses page information entirely — unusable
    for compliance citations.
    """
    logger.info(f"Loading PDF: {file_path.name}")
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    enriched = []
    for page in pages:
        page_text = page.page_content.strip()

        # PITFALL: Detect silently empty pages (scanned image PDFs)
        if not page_text:
            logger.warning(
                f"  ⚠  Empty page detected in {file_path.name} "
                f"(page {page.metadata.get('page', '?')}). "
                f"This document may be a scanned image and require OCR."
            )
            continue

        # Enrich metadata for citations and filtering
        page.metadata.update({
            "source_file": file_path.name,
            "file_path": str(file_path),
            "doc_type": doc_type,
            "doc_type_label": VALID_DOC_TYPES.get(doc_type, doc_type),
            # PyPDFLoader page numbers are 0-indexed; make 1-indexed for humans
            "page_number": page.metadata.get("page", 0) + 1,
        })
        enriched.append(page)

    logger.info(f"  ✓ Loaded {len(enriched)} pages from {file_path.name}")
    return enriched


def load_markdown(file_path: Path, doc_type: str) -> List[Document]:
    """
    Load a Markdown file as a single Document.
    Markdown is treated as page 1 (no real page numbers in .md files).
    """
    logger.info(f"Loading Markdown: {file_path.name}")
    try:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
    except UnicodeDecodeError:
        # Fallback for non-UTF-8 encoded files
        loader = TextLoader(str(file_path), encoding="latin-1")
        docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source_file": file_path.name,
            "file_path": str(file_path),
            "doc_type": doc_type,
            "doc_type_label": VALID_DOC_TYPES.get(doc_type, doc_type),
            "page_number": 1,  # Markdown has no pages
        })

    logger.info(f"  ✓ Loaded {file_path.name}")
    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks for retrieval.

    Strategy: RecursiveCharacterTextSplitter
    - Splits on paragraph breaks first (\\n\\n), then sentences (\\n),
      then words. This preserves semantic boundaries better than
      naive character splitting.
    - chunk_size and chunk_overlap come from validated settings.

    WHY 600 TOKEN CHUNKS FOR FCA DOCS:
    FCA rules frequently contain cross-references within the same
    sub-section (e.g. COBS 4.2.1 referencing COBS 4.1.5). A 600-token
    window captures ~3-4 sub-rules plus their context, giving the LLM
    enough information to answer without being too diluted.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,  # character-based; tiktoken optional for strict token counts
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index within the source document
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    logger.info(f"  ✓ Created {len(chunks)} chunks "
                f"(size≈{settings.chunk_size} tokens, "
                f"overlap={settings.chunk_overlap})")
    return chunks


def ingest_directory(directory: str, doc_type: str) -> int:
    """
    Main ingestion entry point.
    Walks a directory, loads all PDFs and Markdown files,
    chunks them, and upserts into ChromaDB.

    Returns the number of chunks successfully stored.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if doc_type not in VALID_DOC_TYPES:
        raise ValueError(
            f"Invalid doc_type '{doc_type}'. "
            f"Must be one of: {list(VALID_DOC_TYPES.keys())}"
        )

    all_documents: List[Document] = []

    # Collect all supported files
    pdf_files = list(dir_path.rglob("*.pdf"))
    md_files = list(dir_path.rglob("*.md"))

    logger.info(f"Found {len(pdf_files)} PDFs and {len(md_files)} Markdown files")

    for pdf_path in pdf_files:
        docs = load_pdf(pdf_path, doc_type)
        all_documents.extend(docs)

    for md_path in md_files:
        docs = load_markdown(md_path, doc_type)
        all_documents.extend(docs)

    if not all_documents:
        logger.warning("No documents were loaded. Check directory path and file types.")
        return 0

    # Chunk all loaded documents
    chunks = chunk_documents(all_documents)

    # Store in ChromaDB
    add_documents_to_store(chunks)

    logger.info(f"✅ Ingestion complete: {len(chunks)} chunks stored in ChromaDB")
    return len(chunks)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest FCA documents into ChromaDB vector store"
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing PDF and/or Markdown documents",
    )
    parser.add_argument(
        "--doc-type",
        required=True,
        choices=list(VALID_DOC_TYPES.keys()),
        help=f"Document type for metadata. One of: {list(VALID_DOC_TYPES.keys())}",
    )
    args = parser.parse_args()

    count = ingest_directory(args.dir, args.doc_type)
    print(f"\n✅ Done. {count} chunks stored in ChromaDB.")
