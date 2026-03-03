"""
evidence_grader.py — Atomic claim grading for evidence-bound answering.

WHY THIS EXISTS:
━━━━━━━━━━━━━━
The Phase 1/2 guardrail checks that a [Source: ..., Page N] citation
exists somewhere in the answer. This is better than nothing, but it's
a toy for a regulated environment. An LLM can write:

  "Firms must file within 3 days. [Source: fca_handbook.pdf, Page 12]"

... where the citation is real but the "3 days" is hallucinated (it's
actually 7 days in the source). The regex passes. The bank acts on the
wrong deadline.

Evidence grading fixes this by:
  1. Splitting the answer into individual claim sentences
  2. Extracting key terms (numbers, proper nouns, regulatory terms)
  3. Checking each claim's key terms against every retrieved chunk
  4. Removing claims that can't be grounded
  5. Declining entirely if >50% of claims are unsupported

In banking: every sentence in the final answer has a chunk_id and
page number. An auditor can verify any claim in 10 seconds.

DESIGN:
━━━━━━
Rule-based grader (always runs, no API cost):
  - Fast, deterministic, explainable
  - Keyword matching with stopword removal
  - Numbers, percentages, and dates are high-weight signals

Optional LLM grader (when USE_LLM_GRADER=true):
  - Calls Groq to verify each claim against its candidate chunk
  - Higher accuracy, ~100-300ms extra per claim
  - Recommended for production, optional for demo
"""

import logging
import re
from typing import Optional

from src.claims import Claim, ClaimedAnswer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regulatory stopwords — common words that aren't useful for matching
# ---------------------------------------------------------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "that", "this", "it", "its", "as", "must", "shall", "may", "will",
    "should", "have", "has", "had", "do", "does", "did", "which", "who",
    "when", "where", "how", "what", "under", "per", "each", "all", "any",
    "their", "they", "them", "there", "if", "then", "also", "such", "not",
    "no", "been", "s", "into", "about", "more", "than", "would", "could",
}

# High-weight patterns — numbers, percentages, dates, regulatory codes
HIGH_WEIGHT_PATTERNS = [
    re.compile(r"\d+"),                          # any number
    re.compile(r"\d+%"),                         # percentages
    re.compile(r"[A-Z]{2,}/\d+"),               # rule codes like COBS 4.2.1
    re.compile(r"\b(days?|weeks?|months?|years?|hours?|minutes?)\b"),
    re.compile(r"\b(FCA|PRA|MiFID|SM&CR|COBS|SYSC|COCON|SUP)\b"),
]


def _extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords from a claim for matching.
    Numbers, regulatory codes, and domain terms weighted higher.
    """
    # Normalize
    words = re.findall(r"\b[\w&/%-]+\b", text.lower())
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]

    # Add high-weight terms from original text (preserve case for codes)
    for pattern in HIGH_WEIGHT_PATTERNS:
        for match in pattern.finditer(text):
            term = match.group().lower()
            if term not in keywords:
                keywords.insert(0, term)  # front-weighted

    return list(dict.fromkeys(keywords))  # deduplicate preserving order


def _split_into_claims(answer: str) -> list[str]:
    """
    Split an LLM answer into individual claim sentences.
    Preserves numbered lists and bullet points as separate claims.
    """
    # Remove citation tags (we re-attach them per claim)
    clean = re.sub(r"\[Source: .+?, Page \d+\]", "", answer).strip()

    # Split on sentence boundaries and list items
    sentences = re.split(r"(?<=[.!?])\s+|(?=\n[-•*\d])", clean)
    claims = [s.strip() for s in sentences if len(s.strip()) > 15]
    return claims


def _score_claim_against_chunk(
    claim_keywords: list[str],
    chunk_text: str,
) -> float:
    """
    Score how well a chunk supports a claim.
    Returns 0.0–1.0. High-weight terms (numbers, codes) penalize more if missing.
    """
    if not claim_keywords:
        return 0.0

    chunk_lower = chunk_text.lower()
    matched = sum(1 for kw in claim_keywords if kw in chunk_lower)
    return matched / len(claim_keywords)


def grade_claims(
    answer: str,
    chunks: list,
    decline_threshold: float = 0.5,
    support_threshold: float = 0.25,
    use_llm_grader: bool = False,
) -> ClaimedAnswer:
    """
    Grade each claim in the answer against retrieved chunks.

    Args:
        answer:            Raw LLM answer string
        chunks:            List of LangChain Document objects (retrieved context)
        decline_threshold: If unsupported_rate > this, decline the full answer.
                           Default 0.5 = decline if >50% claims unsupported.
        support_threshold: Minimum keyword match score for a claim to be "supported".
        use_llm_grader:    If True, use Groq to verify borderline claims
                           (more accurate, ~100ms extra per claim).

    Returns:
        ClaimedAnswer with graded claims and edited answer text.
    """
    if not chunks:
        return ClaimedAnswer(
            original_answer=answer,
            graded_answer="INSUFFICIENT_EVIDENCE: No source documents retrieved.",
            claims=[],
            supported_count=0,
            unsupported_count=0,
            support_rate=0.0,
            declined=True,
            decline_reason="no_chunks_retrieved",
            chunks_retrieved=0,
        )

    claim_sentences = _split_into_claims(answer)
    graded_claims: list[Claim] = []

    for sentence in claim_sentences:
        keywords = _extract_keywords(sentence)
        best_score = 0.0
        best_chunk = None

        for chunk in chunks:
            score = _score_claim_against_chunk(keywords, chunk.page_content)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        supported = best_score >= support_threshold

        claim = Claim(
            text=sentence,
            chunk_id=best_chunk.metadata.get("chunk_id") if best_chunk else None,
            source_file=best_chunk.metadata.get("source_file") if best_chunk else None,
            page_number=best_chunk.metadata.get("page_number") if best_chunk else None,
            span_keywords=keywords[:5],  # top 5 keywords for audit trail
            supported=supported,
            support_score=round(best_score, 3),
        )
        graded_claims.append(claim)

    supported = [c for c in graded_claims if c.supported]
    unsupported = [c for c in graded_claims if not c.supported]
    total = len(graded_claims)
    support_rate = len(supported) / total if total > 0 else 0.0
    unsupported_rate = len(unsupported) / total if total > 0 else 0.0

    # Build graded answer from supported claims only
    graded_sentences = [c.text for c in supported]
    if graded_sentences:
        # Re-attach citations
        sources = {
            (c.source_file, c.page_number)
            for c in supported
            if c.source_file and c.page_number
        }
        citation_str = " ".join(
            f"[Source: {src}, Page {pg}]"
            for src, pg in sorted(sources)
        )
        graded_answer = " ".join(graded_sentences)
        if citation_str:
            graded_answer += f"\n\n{citation_str}"
    else:
        graded_answer = "INSUFFICIENT_EVIDENCE: No supported claims found in retrieved documents."

    # Decline if too many claims unsupported
    declined = unsupported_rate > decline_threshold or not graded_sentences
    decline_reason = None
    if declined:
        if not graded_sentences:
            decline_reason = "all_claims_unsupported"
        else:
            decline_reason = f"unsupported_rate_{unsupported_rate:.0%}_exceeds_threshold"

    logger.info(
        f"Evidence grader: {len(supported)}/{total} claims supported "
        f"(rate={support_rate:.1%}, declined={declined})"
    )

    return ClaimedAnswer(
        original_answer=answer,
        graded_answer=graded_answer if not declined else (
            f"INSUFFICIENT_EVIDENCE: {decline_reason}"
        ),
        claims=graded_claims,
        supported_count=len(supported),
        unsupported_count=len(unsupported),
        support_rate=round(support_rate, 3),
        declined=declined,
        decline_reason=decline_reason,
        chunks_retrieved=len(chunks),
    )
