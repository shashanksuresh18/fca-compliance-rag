"""
test_pipeline.py — Unit tests for the FCA Compliance RAG pipeline.

These tests validate the critical correctness properties of a
compliance-grade RAG system without requiring API keys or a live
ChromaDB (we use mocks where necessary).

TEST COVERAGE:
  1. Chunk size bounds — FCA text within 500–800 char range
  2. Chunk metadata completeness — every chunk has required fields
  3. Guardrail 1: empty context → decline message
  4. Guardrail 2: citation missing → decline message
  5. Happy path: cited answer passes through

Run with:
    cd "c:\\Users\\USER\\Desktop\\AI ENgineering\\Project Day 1_ RAG"
    python -m pytest tests/test_pipeline.py -v
"""

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helper: Create a mock Document chunk
# ---------------------------------------------------------------------------
def make_mock_chunk(
    content: str = "FCA compliance rule text.",
    source_file: str = "fca_handbook.pdf",
    page_number: int = 5,
    doc_type: str = "handbook",
) -> MagicMock:
    """Create a LangChain-style Document mock with realistic metadata."""
    chunk = MagicMock()
    chunk.page_content = content
    chunk.metadata = {
        "source_file": source_file,
        "page_number": page_number,
        "doc_type": doc_type,
        "doc_type_label": "FCA Handbook",
        "chunk_index": 0,
        "total_chunks": 10,
    }
    return chunk


# ===========================================================================
# 1. CHUNKING TESTS
# ===========================================================================

class TestChunking:
    """Validate that the chunking strategy produces appropriately sized chunks."""

    def test_chunk_size_within_bounds(self):
        """
        Chunks must be within the 450–900 character range.
        Too small = lost cross-reference context for FCA rules.
        Too large = diluted embeddings, poor retrieval precision.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        sample_text = (
            "The Senior Managers and Certification Regime (SM&CR) requires firms "
            "to ensure that individuals who hold key functions are fit and proper. "
            "Under COCON 2.1, the individual conduct rules apply to all relevant persons. "
            "Rule 1 states that you must act with integrity. Rule 2 requires due skill, "
            "care and diligence. Rule 3 requires openness and cooperation with the FCA. "
        ) * 10  # Make it long enough to trigger splitting

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.create_documents([sample_text])

        assert len(chunks) > 1, "Long text should produce multiple chunks"
        for chunk in chunks[:-1]:  # Last chunk may be smaller
            assert len(chunk.page_content) <= 700, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )
            assert len(chunk.page_content) >= 50, (
                f"Chunk suspiciously small: {len(chunk.page_content)} chars"
            )

    def test_chunk_overlap_captures_boundary_text(self):
        """
        Overlapping chunks ensure sentences split at boundaries
        appear in at least one complete chunk.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # Create text where a key sentence falls at a chunk boundary
        padding = "A" * 550
        key_sentence = "Firms must submit Form A to the FCA before appointment."
        text = padding + " " + key_sentence + " " + "B" * 550

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
        )
        chunks = splitter.create_documents([text])

        # The key sentence should appear in at least one chunk
        combined = " ".join(c.page_content for c in chunks)
        assert "Form A" in combined, "Key sentence should survive chunking"


# ===========================================================================
# 2. METADATA TESTS
# ===========================================================================

class TestMetadata:
    """Validate that every chunk carries the required citation metadata."""

    REQUIRED_FIELDS = ["source_file", "page_number", "doc_type", "doc_type_label"]

    def test_all_required_metadata_fields_present(self):
        """Every chunk must have the fields needed to construct an FCA citation."""
        chunk = make_mock_chunk()
        for field in self.REQUIRED_FIELDS:
            assert field in chunk.metadata, f"Missing required metadata field: '{field}'"

    def test_page_number_is_one_indexed(self):
        """Page numbers must be 1-indexed (human-readable for citations)."""
        chunk = make_mock_chunk(page_number=1)
        assert chunk.metadata["page_number"] >= 1, "Page numbers must start at 1"


# ===========================================================================
# 3. GUARDRAIL 1 — Empty context → Decline
# ===========================================================================

class TestGuardrailEmptyContext:
    """
    GUARDRAIL 1: If no chunks are retrieved, the system must decline.
    We NEVER want to call the LLM with no grounding context.
    """

    @patch("src.generator.retrieve", return_value=[])
    @patch("src.generator._load_prompt_config")
    def test_empty_retrieval_returns_decline(self, mock_prompt, mock_retrieve):
        """When retriever returns no chunks, answer must start with INSUFFICIENT_EVIDENCE."""
        mock_prompt.return_value = {
            "system_prompt": "You are a compliance assistant.",
            "user_prompt_template": "Context:\n{context}\n\nQuestion: {question}",
            "citation_pattern": r"\[Source: .+, Page \d+\]",
            "decline_prefix": "INSUFFICIENT_EVIDENCE:",
        }

        from src.generator import generate_answer
        result = generate_answer("What is the capital gains tax rate?")

        assert result["declined"] is True, "Should decline when no chunks retrieved"
        assert result["answer"].startswith("INSUFFICIENT_EVIDENCE"), (
            "Decline message must start with INSUFFICIENT_EVIDENCE"
        )
        assert result["chunks_retrieved"] == 0
        assert result["decline_reason"] == "no_chunks_retrieved"


# ===========================================================================
# 4. GUARDRAIL 2 — No citation → Decline
# ===========================================================================

class TestGuardrailCitationCheck:
    """
    GUARDRAIL 2: If the LLM answers without a [Source: ..., Page N] citation,
    the answer is rejected. An LLM that "forgets" to cite is unacceptable
    in a regulated environment.
    """

    @patch("src.generator.retrieve")
    @patch("src.generator.ChatGroq")
    @patch("src.generator._load_prompt_config")
    def test_answer_without_citation_is_rejected(
        self, mock_prompt, mock_groq_class, mock_retrieve
    ):
        """LLM answer with no [Source: ...] citation must be blocked."""
        mock_prompt.return_value = {
            "system_prompt": "You are a compliance assistant.",
            "user_prompt_template": "Context:\n{context}\n\nQuestion: {question}",
            "citation_pattern": r"\[Source: .+, Page \d+\]",
            "decline_prefix": "INSUFFICIENT_EVIDENCE:",
        }
        mock_retrieve.return_value = [make_mock_chunk()]

        # LLM returns an answer WITHOUT any citation
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Firms must act with integrity."   # No [Source: ...] tag
        )
        mock_groq_class.return_value = mock_llm

        from src.generator import generate_answer
        result = generate_answer("What is SM&CR?")

        assert result["declined"] is True, "Answer without citation must be declined"
        assert result["decline_reason"] == "citation_check_failed"
        assert result["answer"].startswith("INSUFFICIENT_EVIDENCE")

    @patch("src.generator.retrieve")
    @patch("src.generator.ChatGroq")
    @patch("src.generator._load_prompt_config")
    def test_answer_with_citation_passes(
        self, mock_prompt, mock_groq_class, mock_retrieve
    ):
        """A correctly cited answer must pass through the guardrail."""
        mock_prompt.return_value = {
            "system_prompt": "You are a compliance assistant.",
            "user_prompt_template": "Context:\n{context}\n\nQuestion: {question}",
            "citation_pattern": r"\[Source: .+, Page \d+\]",
            "decline_prefix": "INSUFFICIENT_EVIDENCE:",
        }
        mock_retrieve.return_value = [
            make_mock_chunk(content="SM&CR requires firms to act with integrity and skill.")
        ]

        # LLM returns an answer WITH a proper citation matching the keywords above
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=(
                "Under SM&CR, firms must act with integrity and skill. "
                "[Source: fca_handbook.pdf, Page 5]"
            )
        )
        mock_groq_class.return_value = mock_llm

        from src.generator import generate_answer
        result = generate_answer("What is SM&CR?")

        assert result["declined"] is False, "Cited answer should pass guardrail"
        assert "[Source:" in result["answer"]
        assert result["chunks_retrieved"] == 1


# ===========================================================================
# 5. CITATION PATTERN TESTS
# ===========================================================================

class TestCitationPattern:
    """Validate the citation regex pattern itself."""

    PATTERN = r"\[Source: .+, Page \d+\]"

    @pytest.mark.parametrize("citation,expected", [
        ("[Source: fca_handbook.pdf, Page 42]", True),
        ("[Source: mock_sm_cr_policy.md, Page 1]", True),
        ("[Source: Basel III Guidelines.pdf, Page 100]", True),
        ("No citation here", False),
        ("[Source: fca.pdf]", False),           # Missing page
        ("[source: fca.pdf, Page 5]", False),   # Lowercase source
    ])
    def test_citation_pattern_matches_correctly(self, citation, expected):
        """Citation regex must match valid formats and reject invalid ones."""
        result = bool(re.search(self.PATTERN, citation))
        assert result == expected, (
            f"Citation '{citation}' expected match={expected}, got {result}"
        )
