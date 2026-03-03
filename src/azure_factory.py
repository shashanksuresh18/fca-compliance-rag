"""
azure_factory.py — Wave 3: Azure Service Factory for Regulated Banking Environments.

WHY THIS EXISTS:
━━━━━━━━━━━━━━
A "Barclays-Grade" system must run on approved cloud infrastructure (Azure).
However, developers usually want to test locally with Groq (Fast/Free).

This factory provides a clean abstraction:
  - If LLM_PROVIDER="AZURE", it returns authenticated Azure OpenAI clients.
  - If LLM_PROVIDER="GROQ", it returns the Llama-3 client.
  - Same for Vector Search (Azure AI Search vs. Local Chroma).

This allows the system to be developed locally and deployed to the bank's
private cloud with zero code changes — just environment variable updates.
"""

import logging
from typing import Optional, Any
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from src.config import settings

logger = logging.getLogger(__name__)

def get_llm():
    """Returns the LLM client based on the provider setting."""
    if settings.llm_provider.upper() == "AZURE":
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            logger.error("Azure OpenAI credentials missing. Falling back to GROQ.")
            return _get_groq_llm()
            
        logger.info(f"Using Azure OpenAI: {settings.azure_openai_deployment_name}")
        return AzureChatOpenAI(
            azure_deployment=settings.azure_openai_deployment_name,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            temperature=0,
            max_tokens=2048,
            request_timeout=30.0,
            max_retries=3
        )
    else:
        return _get_groq_llm()

def _get_groq_llm():
    """Fallback / Default Groq LLM."""
    logger.info(f"Using Groq LLM: {settings.llm_model}")
    return ChatGroq(
        model=settings.llm_model,
        groq_api_key=settings.groq_api_key,
        temperature=0,
        request_timeout=30.0,
        max_retries=3
    )

def get_retriever_config():
    """
    Returns retrieval configuration (e.g., Azure AI Search vs Local Chroma).
    Note: The actual retrieval logic resides in src/retriever.py 
    but can be branched based on these settings.
    """
    return {
        "provider": settings.llm_provider.upper(),
        "azure_endpoint": settings.azure_search_endpoint,
        "azure_index": settings.azure_search_index_name
    }
