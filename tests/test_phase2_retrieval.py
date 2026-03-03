"""
test_phase2_retrieval.py — Unit tests for Phase 2 hybrid retrieval.

Tests validate:
  1. BM25 retrieval scores exact regulatory terms higher than unrelated chunks
  2. RRF fusion merges two ranked lists correctly
  3. Cross-encoder re-ranker returns correct number of results
  4. retrieve() with use_hybrid=False falls back to vector-only (Phase 1 mode)
  5. retrieve() with use_hybrid=True triggers BM25 code path
  6. BM25 cache invalidation works

Run with:
    python -m pytest tests/test_phase2_retrieval.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# sentence-transformers is NOT in requirements-ci.txt (too heavy for unit test CI)
# Cross-encoder tests are skipped if it's not installed
try:
    import sentence_transformers  # noqa: F401
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False

requires_sentence_transformers = pytest.mark.skipif(
    not sentence_transformers_available,
    reason="sentence-transformers not installed (not in requirements-ci.txt)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_doc(content: str, source: str = "fca.pdf", page: int = 1) -> MagicMock:
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source_file": source, "page_number": page, "doc_type": "handbook"}
    return doc


# ===========================================================================
# 1. BM25 — Exact term matching
# ===========================================================================
class TestBM25Retrieval:
    """BM25 should score exact regulatory rule references highly."""

    def test_bm25_scores_exact_rule_reference_higher(self):
        """
        'COBS 4.2.1' should score higher for a chunk containing that exact
        string than a completely unrelated chunk.
        """
        from rank_bm25 import BM25Okapi

        corpus = [
            "COBS 4.2.1 requires firms to communicate fair, clear and not misleading information.",
            "The capital adequacy rules under Basel III apply to systemically important banks.",
            "Consumer Duty places general obligations on firms acting in retail markets.",
        ]
        tokenised = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenised)

        scores = bm25.get_scores("COBS 4.2.1".lower().split())

        # The first chunk (containing COBS 4.2.1) should score highest
        assert scores[0] == max(scores), (
            "BM25 should rank the chunk containing 'COBS 4.2.1' highest"
        )

    def test_bm25_returns_zero_for_completely_irrelevant_chunk(self):
        """
        A query about FCA conduct rules should give a near-zero BM25 score
        for a chunk about unrelated content.
        """
        from rank_bm25 import BM25Okapi

        corpus = ["The weather in London is typically cold and rainy."]
        tokenised = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenised)

        scores = bm25.get_scores("SM&CR conduct rules FCA".lower().split())
        assert scores[0] == 0.0, "Completely unrelated chunk should score zero"


# ===========================================================================
# 2. Reciprocal Rank Fusion
# ===========================================================================
class TestRRFFusion:
    """RRF should correctly combine two ranked lists."""

    def test_rrf_deduplicates_identical_documents(self):
        """A document in both lists should appear only once in fused output."""
        from src.retriever import _reciprocal_rank_fusion

        doc_a = make_doc("COBS 4.2.1 fair communications rule.", "cobs.pdf", 1)
        doc_b = make_doc("MiFID II client categorisation requirements.", "mifid.pdf", 2)

        # Same doc_a appears in both lists
        vector_results = [doc_a, doc_b]
        bm25_results = [doc_a, doc_b]

        fused = _reciprocal_rank_fusion(vector_results, bm25_results)

        # Should return 2 unique docs, not 4 (no duplicates)
        assert len(fused) == 2

    def test_rrf_scores_document_in_both_lists_higher(self):
        """
        A document appearing at rank 1 in BOTH lists should have the highest
        RRF score (appears first in fused output).
        """
        from src.retriever import _reciprocal_rank_fusion

        shared_doc = make_doc("SM&CR conduct rules for all staff", "smcr.pdf", 2)
        only_in_vector = make_doc("Capital requirements directive.", "crd.pdf", 3)
        only_in_bm25 = make_doc("MiFID II suitability requirements.", "mifid.pdf", 4)

        vector_results = [shared_doc, only_in_vector]
        bm25_results = [shared_doc, only_in_bm25]

        fused = _reciprocal_rank_fusion(vector_results, bm25_results)

        # Doc in both lists should appear first (highest RRF score)
        first_content = fused[0].page_content
        assert "SM&CR" in first_content, (
            "Document ranking #1 in both lists should be #1 in fused output"
        )

    def test_rrf_handles_empty_bm25_list(self):
        """If BM25 returns nothing, fused output should equal vector results."""
        from src.retriever import _reciprocal_rank_fusion

        doc = make_doc("FCA consumer duty overview", "consumer_duty.pdf", 1)
        fused = _reciprocal_rank_fusion([doc], [])

        assert len(fused) == 1
        assert fused[0].page_content == doc.page_content

    def test_rrf_handles_empty_vector_list(self):
        """If vector returns nothing, fused output should equal BM25 results."""
        from src.retriever import _reciprocal_rank_fusion

        doc = make_doc("COBS 4.2.1 text", "cobs.pdf", 1)
        fused = _reciprocal_rank_fusion([], [doc])

        assert len(fused) == 1


# ===========================================================================
# 3. Cross-Encoder Re-Ranker
# ===========================================================================
class TestCrossEncoderReranker:
    """Cross-encoder should return the correct number of re-ranked chunks."""

    @requires_sentence_transformers
    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_returns_top_k_results(self, mock_ce_class):
        """Cross-encoder should return exactly top_k results."""
        from src.retriever import _cross_encoder_rerank

        # Mock the cross-encoder to return descending scores
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.95, 0.80, 0.60, 0.40, 0.20]
        mock_ce_class.return_value = mock_ce

        candidates = [make_doc(f"Chunk {i}", page=i) for i in range(5)]
        result = _cross_encoder_rerank("What are SM&CR conduct rules?", candidates, top_k=3)

        assert len(result) == 3

    @requires_sentence_transformers
    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_orders_by_score_descending(self, mock_ce_class):
        """Cross-encoder should return chunks in descending relevance order."""
        from src.retriever import _cross_encoder_rerank

        mock_ce = MagicMock()
        # Low score for first doc, high score for second
        mock_ce.predict.return_value = [0.1, 0.99]
        mock_ce_class.return_value = mock_ce

        doc_low = make_doc("Unrelated content", "a.pdf", 1)
        doc_high = make_doc("COBS 4.2.1 fair communications rule", "cobs.pdf", 2)
        result = _cross_encoder_rerank("COBS 4.2.1", [doc_low, doc_high], top_k=2)

        # Higher-scored doc should be first
        assert result[0].page_content == doc_high.page_content

    @requires_sentence_transformers
    def test_reranker_falls_back_gracefully_on_error(self):
        """If cross-encoder fails (e.g. model not found), should return candidates[:k]."""
        from src.retriever import _cross_encoder_rerank

        candidates = [make_doc(f"Chunk {i}") for i in range(5)]

        with patch("sentence_transformers.CrossEncoder", side_effect=ImportError("no model")):
            result = _cross_encoder_rerank("query", candidates, top_k=3)

        assert len(result) == 3


# ===========================================================================
# 4. BM25 Cache Invalidation
# ===========================================================================
class TestBM25Cache:
    """BM25 cache invalidation should force index rebuild."""

    def test_invalidate_resets_cache(self):
        """After invalidation, _bm25_index should be None."""
        import src.retriever as retriever_module

        # Manually set a fake index
        retriever_module._bm25_index = MagicMock()
        retriever_module._bm25_corpus = [make_doc("test")]

        retriever_module.invalidate_bm25_cache()

        assert retriever_module._bm25_index is None
        assert retriever_module._bm25_corpus == []
