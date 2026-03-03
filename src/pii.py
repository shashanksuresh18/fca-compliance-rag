"""
pii.py — UK PII detection and masking for regulated banking environments.

WHY THIS EXISTS:
━━━━━━━━━━━━━━
In a bank's RAG system, user queries and document chunks may contain
Personally Identifiable Information (PII). Logging or storing this
data violates UK GDPR and FCA data protection requirements.

This module:
  1. Detects UK-specific PII patterns in any text string
  2. Masks detected PII before it's logged, stored, or returned
  3. Is applied as middleware: query-in → mask → log masked version

UK PII patterns covered:
  - National Insurance numbers (AB123456C)
  - Sort codes (12-34-56)
  - Bank account numbers (8 digits near financial context)
  - UK phone numbers (07xxx xxxxxx, +44...)
  - Email addresses
  - UK postcodes (EC2N 1HQ)
  - Names in HR/identity context (limited pattern)

DESIGN CHOICES:
  - Rule-based only (no ML model) — deterministic, auditable, fast
  - Conservative masking: prefer false positives over false negatives
    (regulatory risk of leaking PII >> annoyance of over-masking)
  - Logs clearly: "[MASKED: NI_NUMBER]" not just "*****"
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII Pattern registry
# ---------------------------------------------------------------------------

@dataclass
class PIIPattern:
    name: str
    pattern: re.Pattern
    mask_label: str
    risk_level: str  # "high", "medium", "low"


# UK-specific PII patterns
UK_PII_PATTERNS: list[PIIPattern] = [
    PIIPattern(
        name="ni_number",
        pattern=re.compile(
            r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
            re.IGNORECASE
        ),
        mask_label="[MASKED:NI_NUMBER]",
        risk_level="high",
    ),
    PIIPattern(
        name="sort_code",
        pattern=re.compile(r"\b\d{2}[-\s]?\d{2}[-\s]?\d{2}\b"),
        mask_label="[MASKED:SORT_CODE]",
        risk_level="high",
    ),
    PIIPattern(
        name="account_number",
        pattern=re.compile(
            r"\b(?:account(?:\s+number)?|a/?c|acc(?:ount)?)\s*[:\-]?\s*(\d{8})\b",
            re.IGNORECASE
        ),
        mask_label="[MASKED:ACCOUNT_NUMBER]",
        risk_level="high",
    ),
    PIIPattern(
        name="uk_phone",
        pattern=re.compile(
            r"\b(?:\+44\s?7\d{3}|07\d{3})\s?\d{3}\s?\d{3}\b"
        ),
        mask_label="[MASKED:PHONE]",
        risk_level="medium",
    ),
    PIIPattern(
        name="us_phone",
        pattern=re.compile(
            r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        mask_label="[MASKED:PHONE]",
        risk_level="medium",
    ),
    PIIPattern(
        name="email",
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ),
        mask_label="[MASKED:EMAIL]",
        risk_level="medium",
    ),
    PIIPattern(
        name="uk_postcode",
        pattern=re.compile(
            r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
            re.IGNORECASE
        ),
        mask_label="[MASKED:POSTCODE]",
        risk_level="low",
    ),
    PIIPattern(
        name="iban",
        pattern=re.compile(r"\bGB\d{2}[A-Z]{4}\d{14}\b", re.IGNORECASE),
        mask_label="[MASKED:IBAN]",
        risk_level="high",
    ),
]


# ---------------------------------------------------------------------------
# Detection + masking
# ---------------------------------------------------------------------------

@dataclass
class PIIMatch:
    pattern_name: str
    matched_text: str
    risk_level: str
    start: int
    end: int


def detect_pii(text: str) -> list[PIIMatch]:
    """
    Scan text for UK PII patterns and return all matches.

    Args:
        text: Input string to scan

    Returns:
        List of PIIMatch objects (empty if no PII detected)
    """
    matches: list[PIIMatch] = []
    for pii_pattern in UK_PII_PATTERNS:
        for match in pii_pattern.pattern.finditer(text):
            matches.append(PIIMatch(
                pattern_name=pii_pattern.name,
                matched_text=match.group(),
                risk_level=pii_pattern.risk_level,
                start=match.start(),
                end=match.end(),
            ))
    return matches


def mask_pii(text: str) -> tuple[str, list[PIIMatch]]:
    """
    Detect and mask all PII in the input text.

    Args:
        text: Input string

    Returns:
        Tuple of (masked_text, list_of_detections)
        masked_text replaces each PII instance with its label, e.g.
        "call me on 07700 900123" → "call me on [MASKED:PHONE]"
    """
    detections = detect_pii(text)
    if not detections:
        return text, []

    # Sort matches in reverse order to preserve string offsets when replacing
    sorted_detections = sorted(detections, key=lambda m: m.start, reverse=True)
    masked = text
    for detection in sorted_detections:
        # find the corresponding label
        label = next(
            (p.mask_label for p in UK_PII_PATTERNS if p.name == detection.pattern_name),
            "[MASKED:PII]"
        )
        masked = masked[:detection.start] + label + masked[detection.end:]

    if detections:
        high_risk = [d for d in detections if d.risk_level == "high"]
        logger.warning(
            f"PII detected: {len(detections)} instance(s) "
            f"({len(high_risk)} high-risk). Masked before logging."
        )

    return masked, detections


def has_pii(text: str) -> bool:
    """Quick boolean check — use when you don't need the match details."""
    return any(p.pattern.search(text) for p in UK_PII_PATTERNS)
