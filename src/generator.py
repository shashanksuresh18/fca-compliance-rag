"""
generator.py — LLM answer generation with citation enforcement guardrails.

THIS IS THE MOST IMPORTANT MODULE FOR BANKING/COMPLIANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In a regulated environment, a wrong answer isn't just a UX failure —
it's a legal risk. A compliance officer who acts on a hallucinated
FCA rule number could expose the bank to regulatory enforcement action
(FCA fines, business restrictions, public censure).

TWO GUARDRAILS IMPLEMENTED HERE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUARDRAIL 1 — EMPTY CONTEXT DECLINE:
  If the retriever returns zero chunks, we never call the LLM.
  This prevents the model from falling back on its training data
  (which may be outdated or simply hallucinated for niche FCA rules).

GUARDRAIL 2 — CITATION ENFORCEMENT:
  After the LLM generates its answer, we apply a regex check:
  the answer MUST contain at least one [Source: ..., Page N] citation.
  If it doesn't, the answer is REJECTED and replaced with a decline
  message — regardless of how confident the LLM sounds.

  LLMs can "fake confidence" even when instructed to cite sources.
  This hard post-generation check is the safety net.

FREE STACK:
  Groq API with llama-3.3-70b-versatile — free, fast (~200 tokens/sec)
  14,400 free requests/day — more than enough for portfolio demos.

PRODUCTION UPGRADES (Phase 2+):
  - Add LLM-as-judge faithfulness check (RAGAS)
  - Add cross-reference verifier: confirm cited page numbers actually
    exist in the retrieved chunks (prevents invented page numbers)
  - Route to different models based on question complexity
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

import yaml
from langchain_groq import ChatGroq
from langchain.schema import Document, HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings
from src.retriever import retrieve

logger = logging.getLogger(__name__)

# --- Decline message returned when we cannot ground an answer ---
DECLINE_MESSAGE = (
    "INSUFFICIENT_EVIDENCE: The provided FCA documentation does not contain "
    "enough information to answer this question. Please consult the relevant "
    "FCA Handbook section directly or seek qualified legal advice."
)


def _load_prompt_config() -> dict:
    """
    Load the versioned prompt template from YAML.
    Version is controlled by settings.prompt_version.

    WHY YAML VERSIONING: Every prompt change is a code change with a
    Git commit hash. Regulators can ask "what prompt was used in March 2026?"
    and you can answer precisely.
    """
    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / settings.prompt_version
        / "qa_prompt.yaml"
    )
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}. "
            f"Check settings.prompt_version = '{settings.prompt_version}'"
        )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_context_block(chunks: list[Document]) -> str:
    """
    Format retrieved chunks into a structured context block for the LLM.

    Each chunk is labelled with its source and page number so the LLM
    can directly embed citations from the provided text.
    The structured format (CHUNK [N]) helps the model track which
    evidence came from which source.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source_file", "Unknown Document")
        page = chunk.metadata.get("page_number", "?")
        doc_type = chunk.metadata.get("doc_type_label", "FCA Document")

        parts.append(
            f"--- CHUNK [{i}] ---\n"
            f"Document: {source} ({doc_type})\n"
            f"Page: {page}\n"
            f"Content:\n{chunk.page_content.strip()}\n"
        )
    return "\n".join(parts)


def _check_citation_present(answer: str, citation_pattern: str) -> bool:
    """
    GUARDRAIL 2: Verify the LLM's answer contains at least one valid citation.

    Pattern from YAML: \\[Source: .+, Page \\d+\\]
    Example match:  [Source: fca_handbook.pdf, Page 42]

    Returns True if at least one citation is present, False otherwise.
    """
    return bool(re.search(citation_pattern, answer))


def generate_answer(
    question: str,
    doc_type: Optional[str] = None,
    top_k: Optional[int] = None,
    use_hybrid: Optional[bool] = None,
    use_reranker: Optional[bool] = None,
) -> dict:
    """
    Full RAG pipeline: Retrieve → Format Context → Generate → Guardrail Check.

    Args:
        question:     The compliance question from the user.
        doc_type:     Optional. Filter retrieval to a specific doc type.
        top_k:        Optional. Override the default number of chunks retrieved.
        use_hybrid:   Phase 2. If None, uses settings.use_hybrid.
                      True = BM25 + vector + RRF. False = vector only.
        use_reranker: Phase 2. If None, uses settings.use_reranker.
                      True = cross-encoder re-ranking. False = RRF order only.

    Returns:
        A dict with answer, citations, declined, decline_reason,
        prompt_version, chunks_retrieved.
    """
    # Resolve retrieval strategy
    _use_hybrid = use_hybrid if use_hybrid is not None else settings.use_hybrid
    _use_reranker = use_reranker if use_reranker is not None else settings.use_reranker

    # ----------------------------------------------------------------
    # STEP 1: Load prompt configuration
    # ----------------------------------------------------------------
    prompt_config = _load_prompt_config()
    system_prompt = prompt_config["system_prompt"]
    user_prompt_template = prompt_config["user_prompt_template"]
    citation_pattern = prompt_config["citation_pattern"]

    # ----------------------------------------------------------------
    # STEP 2: Retrieve relevant chunks (Phase 2: hybrid retrieval)
    # ----------------------------------------------------------------
    chunks = retrieve(
        question,
        top_k=top_k,
        doc_type=doc_type,
        use_hybrid=_use_hybrid,
        use_reranker=_use_reranker,
    )

    # Build citations list from retrieved chunks regardless of outcome
    citations = [
        {
            "source_file": c.metadata.get("source_file", "unknown"),
            "page_number": c.metadata.get("page_number", "?"),
            "doc_type": c.metadata.get("doc_type_label", "FCA Document"),
        }
        for c in chunks
    ]

    # ----------------------------------------------------------------
    # GUARDRAIL 1: Empty context — decline immediately, no LLM call
    # ----------------------------------------------------------------
    if not chunks:
        logger.warning(
            f"No chunks retrieved for query: '{question[:80]}'. "
            f"Declining to answer (Guardrail 1 — empty context)."
        )
        return {
            "answer": DECLINE_MESSAGE,
            "citations": [],
            "declined": True,
            "decline_reason": "no_chunks_retrieved",
            "prompt_version": settings.prompt_version,
            "chunks_retrieved": 0,
        }

    # ----------------------------------------------------------------
    # STEP 3: Format context block with source labels
    # ----------------------------------------------------------------
    context_block = _build_context_block(chunks)
    user_message = user_prompt_template.format(
        context=context_block,
        question=question,
    )

    # ----------------------------------------------------------------
    # STEP 4: Call Groq LLM (free, fast)
    # ----------------------------------------------------------------
    logger.info(
        f"Calling Groq ({settings.llm_model}) with {len(chunks)} context chunks..."
    )

    llm = ChatGroq(
        model=settings.llm_model,
        groq_api_key=settings.groq_api_key,
        temperature=0,           # 0 = deterministic, better for compliance
        max_tokens=2048,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw_answer = response.content.strip()

    logger.info(f"  ✓ LLM responded ({len(raw_answer)} chars)")

    # ----------------------------------------------------------------
    # GUARDRAIL 2: Citation enforcement — reject if no [Source: ...] found
    # ----------------------------------------------------------------
    # Allow the LLM's own INSUFFICIENT_EVIDENCE responses to pass through
    if raw_answer.startswith(prompt_config["decline_prefix"]):
        logger.info("LLM self-declined — passing through decline message.")
        return {
            "answer": raw_answer,
            "citations": citations,
            "declined": True,
            "decline_reason": "llm_self_declined",
            "prompt_version": settings.prompt_version,
            "chunks_retrieved": len(chunks),
        }

    if not _check_citation_present(raw_answer, citation_pattern):
        logger.warning(
            "GUARDRAIL 2 TRIGGERED: LLM answer contains no valid citations. "
            "Rejecting and returning decline message."
        )
        return {
            "answer": DECLINE_MESSAGE,
            "citations": citations,
            "declined": True,
            "decline_reason": "citation_check_failed",
            "prompt_version": settings.prompt_version,
            "chunks_retrieved": len(chunks),
            "rejected_answer": raw_answer,  # Logged for debugging — not shown to user
        }

    # ----------------------------------------------------------------
    # STEP 5: Return grounded, cited answer
    # ----------------------------------------------------------------
    return {
        "answer": raw_answer,
        "citations": citations,
        "declined": False,
        "prompt_version": settings.prompt_version,
        "chunks_retrieved": len(chunks),
    }
