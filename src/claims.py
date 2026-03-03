"""
claims.py — Atomic claim schema for evidence-bound answering.

Instead of regex-checking that a [Source:] tag exists, we decompose
the LLM answer into individual claims and verify each one against
the retrieved document chunks. This is the correct approach for a
regulated environment — you can show an auditor exactly which
sentence in the source document supports each claim in the answer.

In banking: a claim like "firms must notify the FCA within 7 days"
must be traceable to an exact chunk. If it can't be, it's removed.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single atomic claim extracted from an LLM answer."""

    text: str = Field(
        description="The verbatim claim sentence from the LLM answer"
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="ID of the chunk that supports this claim"
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Source document filename"
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number in source document (1-indexed)"
    )
    span_keywords: list[str] = Field(
        default_factory=list,
        description="Key terms from the claim used to locate it in the chunk"
    )
    supported: bool = Field(
        default=False,
        description="True if this claim is grounded in the retrieved context"
    )
    support_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of span_keywords found in the supporting chunk"
    )


class ClaimedAnswer(BaseModel):
    """
    The output of the evidence grader — an answer broken into
    individually-verified claims, with unsupported content removed.
    """

    original_answer: str = Field(
        description="The raw LLM-generated answer before grading"
    )
    graded_answer: str = Field(
        description="Answer with unsupported claims removed"
    )
    claims: list[Claim] = Field(
        description="All extracted claims with their support status"
    )
    supported_count: int = Field(
        description="Number of claims with verified evidence"
    )
    unsupported_count: int = Field(
        description="Number of claims removed due to lack of evidence"
    )
    support_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of claims that were grounded (1.0 = no hallucination)"
    )
    declined: bool = Field(
        description="True if majority of claims were unsupported → full decline"
    )
    decline_reason: Optional[str] = Field(
        default=None,
        description="Why the answer was declined, if applicable"
    )
    prompt_version: str = Field(
        default="v2",
        description="Prompt template version used"
    )
    chunks_retrieved: int = Field(
        default=0,
        description="Number of chunks retrieved for this query"
    )
