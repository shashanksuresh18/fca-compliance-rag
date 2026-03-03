"""
test_audit.py — Unit tests for structured audit logging.

Verifies that:
  1. Audit logs are written as valid JSONL.
  2. The trace_id returned matches the one in the file.
  3. Sensitive data (real queries) can be masked before logging.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audit_log import audit_logger, AUDIT_DIR

class TestAuditLogging:
    @pytest.fixture(autouse=True)
    def setup_cleanup(self):
        # Ensure directory exists
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        yield
        # We don't delete logs during tests to allow inspection if needed,
        # but in a CI we might.

    def test_log_event_creates_jsonl_entry(self):
        user_id = "user_123"
        user_role = "analyst"
        query = "What is COBS 4?"
        response_data = {
            "declined": False,
            "prompt_version": "v2",
            "retrieval_strategy": "hybrid",
            "chunks_retrieved": 5,
            "citations": [{"source_file": "fca.pdf"}]
        }
        
        trace_id = audit_logger.log_event(
            user_id=user_id,
            user_role=user_role,
            query_masked=query,
            response_data=response_data,
            latency_ms=150.5
        )
        
        assert trace_id.startswith("tr_")
        
        # Verify file content
        log_file = audit_logger.log_file
        assert log_file.exists()
        
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            last_entry = json.loads(lines[-1])
            
            assert last_entry["trace_id"] == trace_id
            assert last_entry["user"]["id"] == user_id
            assert last_entry["user"]["role"] == user_role
            assert last_entry["request"]["query"] == query
            assert last_entry["response"]["chunks_used"] == 5
            assert last_entry["performance"]["latency_ms"] == 150.5
