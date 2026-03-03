"""
graph.py — Wave 3: LangGraph Orchestration for the FCA RAG Pipeline.

WHY LANGGRAPH?
━━━━━━━━━━━━━
A linear pipeline (Phase 1-3) is easy to build but hard to debug and extend. 
LangGraph transforms the RAG flow into a state-driven graph. This provides:
  1. Observability: Each "node" (Retrieve, Generate, Grade) is a discrete step.
  2. Conditional Logic: We can bail out early if injection is detected.
  3. Cycle Support: Future phases can "self-correct" if grading fails.
  4. State Persistence: The context of the entire run is preserved in RAGState.

GRAPH TOPOLOGY:
  [Start] 
     ↓
  [Guardrails] ─(is_injection?)─→ [Finalize]
     ↓ (safe)
  [Retrieve] ──(no chunks?)────→ [Finalize]
     ↓ (has chunks)
  [Generate]
     ↓
  [Grade]
     ↓
  [Finalize] 
     ↓
  [End]
"""

import logging
from typing import Annotated, Dict, List, Optional, TypedDict, Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.config import settings
from src.retriever import retrieve
from src.prompter import load_prompt_config, build_context_block, check_citation_present
from src.evidence_grader import grade_claims
from src.pii import has_pii, mask_pii
from src.injection_guard import check_query_for_injection, sanitize_chunks
from src.claims import ClaimedAnswer
from src.utils import llm_circuit_breaker, Timer
from src.otel import get_tracer
from src.azure_factory import get_llm

tracer = get_tracer()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    """
    The state object passed between nodes. 
    It tracks the inputs, intermediate results, and final output.
    """
    # Inputs from user/request
    question: str
    user_role: str
    doc_type: Optional[str]
    top_k: int
    use_hybrid: bool
    use_reranker: bool
    
    # Security items
    pii_detected: bool
    is_injection: bool
    
    # Pipeline items
    chunks: List[Document]
    raw_answer: str
    graded_answer: Optional[ClaimedAnswer]
    
    # Execution metrics
    latency_ms: float
    
    # Final response
    final_output: Dict[str, Any]

# ---------------------------------------------------------------------------
# 2. Nodes (The "Workhorses")
# ---------------------------------------------------------------------------

def guardrails_node(state: RAGState) -> Dict:
    """Check for PII and prompt injection before starting retrieval."""
    with tracer.start_as_current_span("node.guardrails") as span:
        logger.info("Node: Guardrails")
        q = state["question"]
        
        # Injection Check
        is_inj, matches = check_query_for_injection(q)
        
        # PII Check
        pii_found = has_pii(q)
        
        span.set_attribute("is_injection", is_inj)
        span.set_attribute("pii_detected", pii_found)
        
        return {
            "is_injection": is_inj,
            "pii_detected": pii_found
        }


def retrieve_node(state: RAGState) -> Dict:
    """Retrieve document chunks from ChromaDB and apply ACLs."""
    with tracer.start_as_current_span("node.retrieve") as span:
        logger.info("Node: Retrieve")
        
        chunks = retrieve(
            query=state["question"],
            top_k=state["top_k"],
            doc_type=state["doc_type"],
            use_hybrid=state["use_hybrid"],
            use_reranker=state["use_reranker"],
            user_role=state["user_role"]
        )
        
        # P4.2: Sanitize chunks against indirect injection
        sanitized, _ = sanitize_chunks(chunks)
        
        span.set_attribute("chunk_count", len(sanitized))
        return {"chunks": sanitized}


def generate_node(state: RAGState) -> Dict:
    """Draft an answer using the LLM with retrieved chunks."""
    with tracer.start_as_current_span("node.generate") as span:
        logger.info("Node: Generate")
        
        if not state["chunks"]:
            return {"raw_answer": "INSUFFICIENT_EVIDENCE: No chunks retrieved."}
            
        prompt_config = load_prompt_config()
        context_block = build_context_block(state["chunks"])
        
        system_prompt = prompt_config["system_prompt"]
        user_message = prompt_config["user_prompt_template"].format(
            context=context_block,
            question=state["question"]
        )
        
        llm = get_llm()
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        
        try:
            # Reliability: Use circuit breaker
            response = llm_circuit_breaker.call(llm.invoke, messages)
            return {"raw_answer": response.content.strip()}
        except Exception as e:
            logger.error(f"LLM Node failure: {e}")
            span.record_exception(e)
            return {"raw_answer": "INSUFFICIENT_EVIDENCE: LLM failure."}


def grade_node(state: RAGState) -> Dict:
    """Verify atomic claims in the LLM draft against the evidence chunks."""
    with tracer.start_as_current_span("node.grade") as span:
        logger.info("Node: Grade")
        
        raw = state.get("raw_answer", "")
        if "INSUFFICIENT_EVIDENCE" in raw:
            return {"graded_answer": None}
            
        graded = grade_claims(raw, state["chunks"])
        span.set_attribute("support_rate", graded.support_rate if graded else 0.0)
        return {"graded_answer": graded}


def finalize_node(state: RAGState) -> Dict:
    """Package the results and handle decline logic."""
    with tracer.start_as_current_span("node.finalize") as span:
        logger.info("Node: Finalize")
    
    # Case A: Injection Blocked
    if state.get("is_injection"):
        return {
            "final_output": {
                "answer": "Query rejected due to safety filter violation.",
                "declined": True,
                "decline_reason": "prompt_injection_detected"
            }
        }
    
    # Case B: No chunks found
    if not state.get("chunks"):
        return {
            "final_output": {
                "answer": "INSUFFICIENT_EVIDENCE: The provided context is empty.",
                "declined": True,
                "decline_reason": "empty_retrieval"
            }
        }
        
    # Case C: Citation check (Guardrail 2)
    prompt_config = load_prompt_config()
    raw_answer = state.get("raw_answer", "")
    if not check_citation_present(raw_answer, prompt_config["citation_pattern"]):
        if not raw_answer.startswith(prompt_config["decline_prefix"]):
            return {
                "final_output": {
                    "answer": "INSUFFICIENT_EVIDENCE: Answer provided without citations.",
                    "declined": True,
                    "decline_reason": "missing_citations"
                }
            }

    # Case D: Evidence Grader declined
    graded = state.get("graded_answer")
    if graded and graded.declined:
        return {
            "final_output": {
                "answer": DECLINE_MESSAGE,
                "declined": True,
                "decline_reason": "insufficient_evidence",
                "evidence_support_rate": graded.support_rate
            }
        }
        
    # Case D: Success
    ans = graded.answer if graded else state["raw_answer"]
    
    # Build citations JSON
    citations = []
    seen = set()
    for chunk in state["chunks"]:
        cid = f"{chunk.metadata.get('source_file')}:{chunk.metadata.get('page_number')}"
        if cid not in seen:
            citations.append({
                "source_file": chunk.metadata.get("source_file"),
                "page_number": chunk.metadata.get("page_number"),
                "doc_type": chunk.metadata.get("doc_type")
            })
            seen.add(cid)

    return {
        "final_output": {
            "answer": ans,
            "citations": citations,
            "declined": False,
            "prompt_version": settings.prompt_version,
            "chunks_retrieved": len(state["chunks"]),
            "evidence_support_rate": graded.support_rate if graded else 0.0,
            "pii_detected": state["pii_detected"]
        }
    }

# ---------------------------------------------------------------------------
# 3. Graph Assembly
# ---------------------------------------------------------------------------

def decider_after_guardrails(state: RAGState) -> Literal["retrieve", "finalize"]:
    if state["is_injection"]:
        return "finalize"
    return "retrieve"

def decider_after_retrieve(state: RAGState) -> Literal["generate", "finalize"]:
    if not state["chunks"]:
        return "finalize"
    return "generate"

def create_rag_graph():
    workflow = StateGraph(RAGState)
    
    # Add Nodes
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("finalize", finalize_node)
    
    # Build Edges
    workflow.set_entry_point("guardrails")
    
    workflow.add_conditional_edges(
        "guardrails",
        decider_after_guardrails
    )
    
    workflow.add_conditional_edges(
        "retrieve",
        decider_after_retrieve
    )
    
    workflow.add_edge("generate", "grade")
    workflow.add_edge("grade", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# Global compiled graph
rag_graph = create_rag_graph()
