"""
test_acl.py — Unit tests for Retrieval-time Access Control Lists (ACLs).

These tests verify that users with limited roles (Analyst) are
prevented from seeing confidential documents (Internal/Confidential),
even if those documents are the best semantic matches for their query.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

# Add project root so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acl_filter import filter_chunks_by_role
from src.auth import UserRole

def make_chunk(doc_id: str, doc_type: str) -> Document:
    return Document(
        page_content=f"Content for {doc_id}",
        metadata={"chunk_id": doc_id, "doc_type": doc_type}
    )

class TestACLFiltering:
    def test_analyst_can_see_handbook(self):
        chunks = [make_chunk("c1", "handbook")]
        filtered, dropped = filter_chunks_by_role(chunks, UserRole.ANALYST)
        assert len(filtered) == 1
        assert dropped == 0
        assert filtered[0].metadata["chunk_id"] == "c1"

    def test_analyst_blocked_from_internal(self):
        chunks = [
            make_chunk("c1", "handbook"),
            make_chunk("c2", "internal"),
        ]
        filtered, dropped = filter_chunks_by_role(chunks, UserRole.ANALYST)
        assert len(filtered) == 1
        assert dropped == 1
        assert filtered[0].metadata["chunk_id"] == "c1"

    def test_compliance_can_see_internal(self):
        chunks = [
            make_chunk("c1", "handbook"),
            make_chunk("c2", "internal"),
        ]
        filtered, dropped = filter_chunks_by_role(chunks, UserRole.COMPLIANCE)
        assert len(filtered) == 2
        assert dropped == 0

    def test_admin_can_see_hr(self):
        chunks = [make_chunk("c1", "hr")]
        filtered, dropped = filter_chunks_by_role(chunks, UserRole.ADMIN)
        assert len(filtered) == 1
        assert dropped == 0

    def test_analyst_blocked_from_hr(self):
        chunks = [make_chunk("c1", "hr")]
        filtered, dropped = filter_chunks_by_role(chunks, UserRole.ANALYST)
        assert len(filtered) == 0
        assert dropped == 1

    def test_unknown_doc_type_defaults_to_allow(self):
        # We allow unknown types by default to avoid breaking existing setups
        chunks = [make_chunk("c1", "unknown_type")]
        filtered, dropped = filter_chunks_by_role(chunks, UserRole.ANALYST)
        assert len(filtered) == 1
        assert dropped == 0
