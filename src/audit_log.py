"""
audit_log.py — Structured audit logging for regulatory compliance.

WHY THIS EXISTS:
━━━━━━━━━━━━━━
In banking, "I don't know why the AI gave that answer" is not an acceptable
response to a regulator. You must be able to reconstruct any session.

This module provides an append-only, structured (JSON) audit log that
captures every request and decision. Each log entry is linked by a
unique `trace_id` that is also returned to the user.

FIELDS LOGGED:
  - trace_id: Unique ID for cross-referencing
  - timestamp: ISO8601 UTC
  - user_id/role: Who asked? (masked if PII)
  - query_masked: What was asked? (ALWAYS masked for PII)
  - prompt_version: Which rules were active?
  - doc_ids: Which exact chunks were retrieved?
  - decision: Answered or Declined?
  - support_rate: Evidence grader score
  - latency_ms: Performance metric
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# Ensure audit directory exists
AUDIT_DIR = Path("logs/audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

class AuditLogger:
    def __init__(self, service_name: str = "fca-rag"):
        self.service_name = service_name
        self.log_file = AUDIT_DIR / f"audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"

    def log_event(
        self,
        user_id: str,
        user_role: str,
        query_masked: str,
        response_data: Dict[str, Any],
        latency_ms: float,
        trace_id: Optional[str] = None
    ) -> str:
        """
        Record a structured audit event to the append-only JSONL log.
        """
        trace_id = trace_id or f"tr_{uuid.uuid4().hex[:12]}"
        
        event = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service_name,
            "user": {
                "id": user_id,
                "role": user_role,
            },
            "request": {
                "query": query_masked,
                "prompt_version": response_data.get("prompt_version"),
                "strategy": response_data.get("retrieval_strategy"),
            },
            "response": {
                "declined": response_data.get("declined", False),
                "decline_reason": response_data.get("decline_reason"),
                "support_rate": response_data.get("evidence_support_rate"),
                "chunks_used": response_data.get("chunks_retrieved", 0),
                # We log doc IDs but NOT the raw content to save space and reduce PII risk
                "doc_ids": [c.get("source_file") for c in response_data.get("citations", [])],
            },
            "performance": {
                "latency_ms": round(latency_ms, 2),
            },
            "pii_detected": response_data.get("pii_detected", False)
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"FATAL: Audit log write failed: {e}")
            # In a real bank, this might trigger a service shutdown or
            # emergency alert because unlogged operations are non-compliant.
            
        return trace_id

# Global instance
audit_logger = AuditLogger()
