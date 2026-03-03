"""
prompter.py — Prompt loading and formatting utilities.
"""

import logging
import re
import yaml
from pathlib import Path
from langchain_core.documents import Document
from src.config import settings

logger = logging.getLogger(__name__)

def load_prompt_config() -> dict:
    """Load the versioned prompt template from YAML."""
    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / settings.prompt_version
        / "qa_prompt.yaml"
    )
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_context_block(chunks: list[Document]) -> str:
    """Format retrieved chunks into a structured context block for the LLM."""
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


def check_citation_present(answer: str, citation_pattern: str) -> bool:
    """Verify the LLM's answer contains at least one valid citation."""
    return bool(re.search(citation_pattern, answer))
