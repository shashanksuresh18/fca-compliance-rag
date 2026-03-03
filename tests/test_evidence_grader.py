"""
test_evidence_grader.py — Unit tests for evidence-bound answering.

Tests cover the three critical properties:
  1. Supported claims pass through
  2. Unsupported claims are trimmed
  3. Majority-unsupported answers → full decline

Run with:
    python -m pytest tests/test_evidence_grader.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_chunk(content: str, source: str = "fca.pdf", page: int = 1):
    chunk = MagicMock()
    chunk.page_content = content
    chunk.metadata = {
        "source_file": source,
        "page_number": page,
        "chunk_id": f"chunk_{hash(content) % 10000}",
    }
    return chunk


class TestClaimExtraction:
    """Keyword extraction and claim splitting."""

    def test_numbers_are_extracted_as_keywords(self):
        from src.evidence_grader import _extract_keywords
        keywords = _extract_keywords("Firms must notify within 7 business days")
        assert "7" in keywords, "Numbers should be extracted as keywords"

    def test_stopwords_excluded(self):
        from src.evidence_grader import _extract_keywords
        keywords = _extract_keywords("The firm must act with integrity")
        assert "the" not in keywords
        assert "with" not in keywords

    def test_regulatory_codes_extracted(self):
        from src.evidence_grader import _extract_keywords
        keywords = _extract_keywords("Under COBS 4.2.1, firms must ensure fair communication")
        combined = " ".join(keywords)
        assert "cobs" in combined or "4.2" in combined

    def test_split_into_claims_produces_multiple(self):
        from src.evidence_grader import _split_into_claims
        answer = (
            "Under SM&CR, firms must ensure all staff understand the Conduct Rules. "
            "Senior Managers must submit a Statement of Responsibilities. "
            "Conduct Rule breaches must be reported within 7 days."
        )
        claims = _split_into_claims(answer)
        assert len(claims) >= 2, "Multi-sentence answer should produce multiple claims"


class TestEvidenceGrader:
    """Core grading logic."""

    def test_supported_claim_passes(self):
        """A claim whose keywords appear in a chunk should be marked supported."""
        from src.evidence_grader import grade_claims

        chunk = make_chunk(
            "Firms must notify the FCA within 7 business days of identifying a conduct rule breach. "
            "This applies to all Senior Managers under SM&CR."
        )
        answer = "Firms must notify the FCA within 7 business days of a conduct rule breach."
        result = grade_claims(answer, [chunk], support_threshold=0.2)

        supported = [c for c in result.claims if c.supported]
        assert len(supported) > 0, "Claim with matching keywords should be supported"
        assert not result.declined

    def test_unsupported_claim_trimmed(self):
        """A hallucinated claim with no keyword overlap should be marked unsupported."""
        from src.evidence_grader import grade_claims

        chunk = make_chunk(
            "Consumer Duty places obligations on firms to deliver good outcomes for retail customers."
        )
        # Answer claims something completely different
        hallucinated_answer = (
            "Firms must file annual reports with Companies House by 31 January every year. "
            "The deadline was extended in 2024 due to HMRC changes."
        )
        result = grade_claims(hallucinated_answer, [chunk], support_threshold=0.3)

        unsupported = [c for c in result.claims if not c.supported]
        assert len(unsupported) > 0, "Hallucinated claim should be marked unsupported"

    def test_majority_unsupported_triggers_decline(self):
        """If >50% of claims are unsupported, the full answer must be declined."""
        from src.evidence_grader import grade_claims

        chunk = make_chunk("SM&CR requires firms to have a Chief Risk Officer.")

        # 1 supported claim + 3 hallucinated claims = 75% unsupported → decline
        mixed_answer = (
            "SM&CR requires firms to have a Chief Risk Officer. "
            "Firms must file a P60 form every quarter with HMRC. "
            "The FCA requires annual stress tests under Basel IV from 2027. "
            "All trading desks must post collateral under the EMIR Refit rules."
        )
        result = grade_claims(
            mixed_answer, [chunk],
            decline_threshold=0.5,
            support_threshold=0.25,
        )

        assert result.declined, "Majority-unsupported answer should be declined"
        assert result.decline_reason is not None

    def test_no_chunks_always_declines(self):
        """With no retrieved chunks, every answer must be declined."""
        from src.evidence_grader import grade_claims

        result = grade_claims("Firms must act with integrity.", chunks=[])
        assert result.declined
        assert result.decline_reason == "no_chunks_retrieved"
        assert result.chunks_retrieved == 0

    def test_graded_answer_contains_citations(self):
        """Supported claims should re-attach source citations."""
        from src.evidence_grader import grade_claims

        chunk = make_chunk(
            "The Senior Managers and Certification Regime requires individual accountability.",
            source="mock_sm_cr_policy.md",
            page=3,
        )
        answer = "The Senior Managers and Certification Regime requires individual accountability."
        result = grade_claims(answer, [chunk], support_threshold=0.15)

        if not result.declined:
            assert "mock_sm_cr_policy.md" in result.graded_answer or result.declined

    def test_support_rate_is_valid_range(self):
        """support_rate must always be between 0.0 and 1.0."""
        from src.evidence_grader import grade_claims

        chunk = make_chunk("Any regulatory text here.")
        result = grade_claims("Some answer text.", [chunk])
        assert 0.0 <= result.support_rate <= 1.0

    def test_claimed_answer_has_required_fields(self):
        """ClaimedAnswer must always have all required fields populated."""
        from src.evidence_grader import grade_claims

        chunk = make_chunk("FCA requires conduct rule training.")
        result = grade_claims("FCA requires training.", [chunk])

        assert result.original_answer is not None
        assert result.graded_answer is not None
        assert isinstance(result.claims, list)
        assert isinstance(result.supported_count, int)
        assert isinstance(result.unsupported_count, int)
        assert isinstance(result.declined, bool)
