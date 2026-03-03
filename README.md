# 🏦 FCA Compliance RAG System

**Production-grade Retrieval-Augmented Generation for UK Banking Regulatory Compliance**

> Built as a portfolio project for AI Engineer roles at UK banks (HSBC, Barclays, Lloyds, NatWest).  
> Answers FCA regulatory questions grounded in real documentation — with mandatory citations and zero-hallucination guardrails.

![CI](https://github.com/shashanksuresh18/fca-compliance-rag/actions/workflows/ci.yml/badge.svg)

---

## 🏗️ Architecture (Phase 2)

```
User Question
     │
     ▼
 [FastAPI /query]  ← per-request use_hybrid / use_reranker toggles
     │
     ▼
 [Phase 2 Retriever]
   ├── BM25 keyword search   ← exact rule refs: "COBS 4.2.1", "SM&CR"
   ├── Dense vector search   ← semantic: "rules on vulnerable customers"
   ├── Reciprocal Rank Fusion (RRF) ← merge + deduplicate
   └── Cross-encoder Re-ranking     ← ms-marco-MiniLM-L-6-v2 (22MB, CPU)
     │
     ▼
 [Generator]  ──── Groq llama-3.3-70b — FREE, 200 tok/sec
     │
     ▼
 [Guardrails]
  ✓ Guardrail 1: No chunks retrieved → DECLINE (LLM never called)
  ✓ Guardrail 2: No [Source: ..., Page N] citation → REJECT answer
     │
     ▼
 Grounded Answer + Citations
     │
     ▼
 [Phase 3 CI — runs on every push to main]
  ✓ RAGAS Faithfulness ≥ 0.85 gate
  ✓ Citation Coverage = 100%
  ✓ Decline Accuracy = 100%
```

---

## 💰 Free Stack (Zero Cost)

| Component | Technology | Cost |
|-----------|-----------|------|
| Embeddings | `all-MiniLM-L6-v2` (runs locally, no API) | **$0** |
| Keyword search | BM25 (`rank-bm25`) | **$0** |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, 22MB) | **$0** |
| Vector Store | ChromaDB (local, on-disk) | **$0** |
| LLM | Groq — `llama-3.3-70b-versatile` | **$0** (14.4k req/day free) |
| Eval | RAGAS + Groq | **$0** |
| CI | GitHub Actions | **$0** |

---

## ⚡ Quickstart (5 steps)

### 1. Get your free Groq API key
Sign up at **[console.groq.com](https://console.groq.com)** — no credit card needed.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First run downloads `all-MiniLM-L6-v2` (~90MB) and `ms-marco-MiniLM-L-6-v2` (~22MB) once. All local, no API.

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...your_key...
```

### 4. Ingest mock FCA documents
```bash
python src/ingest.py --dir data/mock --doc-type internal
```

### 5. Start the API
```bash
python -m uvicorn api.main:app --reload
```
Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 🔍 Example Queries

```bash
# SM&CR conduct rules (semantic query → vector search shines)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the five conduct rules under SM&CR?"}'

# Exact rule reference (BM25 keyword search shines)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "COBS 4.2.1 requirements"}'

# Out-of-scope → decline guardrail
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the FCA view on Bitcoin staking?"}'

# Phase 2 toggle: use vector-only (Phase 1 mode) for comparison
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "MiFID II client categories", "use_hybrid": false}'
```

**Example response:**
```json
{
  "answer": "Under SM&CR, there are five Individual Conduct Rules...\n[Source: mock_sm_cr_policy.md, Page 2]",
  "citations": [{"source_file": "mock_sm_cr_policy.md", "page_number": 2, "doc_type": "Internal Bank Policy"}],
  "declined": false,
  "prompt_version": "v2",
  "chunks_retrieved": 5,
  "retrieval_strategy": "hybrid-reranked"
}
```

---

## 📁 Project Structure

```
fca-compliance-rag/
├── src/
│   ├── config.py        # Pydantic settings with Phase 2 flags
│   ├── ingest.py        # PDF/Markdown ingestion + chunking (600 chars, 100 overlap)
│   ├── store.py         # ChromaDB with SHA-256 dedup upsert
│   ├── retriever.py     # Phase 2: BM25 + Vector + RRF + Cross-encoder re-ranker
│   └── generator.py    # LLM inference + 2 citation guardrails
├── api/
│   └── main.py          # FastAPI v2.0.0: /query, /ingest, /health
├── prompts/
│   ├── v1/qa_prompt.yaml    # Basic citation prompt
│   └── v2/qa_prompt.yaml    # Chain-of-thought + confidence flagging (default)
├── eval/
│   ├── golden_dataset.json  # 25 curated Q&A pairs (SM&CR + MiFID II + decline tests)
│   └── evaluate.py          # RAGAS eval: faithfulness, relevancy, citation coverage
├── tests/
│   ├── test_pipeline.py          # Phase 1: chunking, metadata, guardrails
│   └── test_phase2_retrieval.py  # Phase 2: BM25, RRF, cross-encoder, cache
├── .github/
│   └── workflows/ci.yml     # GitHub Actions: unit tests + RAGAS faithfulness gate
├── data/
│   └── mock/            # SM&CR policy + MiFID II client categorisation demo docs
├── requirements.txt     # Full stack (includes torch)
├── requirements-ci.txt  # Lightweight CI deps (no torch — fast unit tests)
└── .env.example
```

---

## 🧪 Testing

### Unit tests (no API keys needed — ~30 seconds)
```bash
python -m pytest tests/ -v
```
**23 tests** across Phase 1 + Phase 2:
- ✅ Chunk size bounds + overlap correctness
- ✅ Chunk metadata completeness (source, page, doc_type)
- ✅ Guardrail 1: empty context → decline (LLM never called)
- ✅ Guardrail 2: uncited answer → rejected
- ✅ BM25 scores exact rule references highest
- ✅ RRF fusion deduplication and ranking
- ✅ Cross-encoder top-k ordering
- ✅ BM25 cache invalidation

### RAGAS evaluation (requires Groq key — ~5 minutes)
```bash
python eval/evaluate.py
```
Runs 25 golden Q&A pairs and scores 4 metrics:

| Metric | What it measures | CI threshold |
|--------|-----------------|-------------|
| **Faithfulness** | Hallucination rate | ≥ 0.85 (hard gate) |
| **Answer Relevancy** | Response quality | Monitored |
| **Citation Coverage** | % answers with `[Source:]` tag | 100% expected |
| **Decline Accuracy** | Out-of-scope refusal rate | 100% expected |

---

## � Phase 2: Hybrid Retrieval Explained

Phase 1 used dense vector search only. Phase 2 adds:

**Why hybrid matters for FCA compliance:**

| Query type | Phase 1 (vector only) | Phase 2 (hybrid) |
|-----------|----------------------|-----------------|
| `"COBS 4.2.1 requirements"` | ❌ May miss exact term | ✅ BM25 exact match |
| `"rules for vulnerable customers"` | ✅ Semantic match | ✅ Both methods agree |
| `"SM&CR Article 25"` | ⚠️ Partial | ✅ BM25 + semantic |

**Pipeline:** BM25 (top-10) + Vector (top-10) → RRF fusion → Cross-encoder re-ranks to top-5

**Toggle per-request:**
```json
{"question": "...", "use_hybrid": true, "use_reranker": true}   // Phase 2 (default)
{"question": "...", "use_hybrid": false}                        // Phase 1 mode
```

---

## 🔑 Why These Design Choices Matter in Banking

### Citation Guardrails (Two Layers)
1. **Empty context check**: If ChromaDB returns nothing, the LLM is never called. Prevents the model answering from training data — which may contain outdated or hallucinated rule numbers.
2. **Post-generation citation regex**: The response must contain at least one `[Source: filename, Page N]` tag. An uncited answer is rejected. In a regulated environment, a wrong answer is a legal risk.

### Prompt Versioning (YAML)
Every prompt change is a Git commit. A regulator asking *"what exact prompt was used in March 2026?"* gets a precise answer. CI runs on every prompt change.

### RAGAS Evaluation Gate
A prompt change that silently drops faithfulness from 0.92 to 0.78 would reach production undetected without automated eval. The CI gate blocks deployment when faithfulness < 0.85.

### Metadata Filtering
Legal team queries only the FCA Handbook; product teams query Policy Statements. `doc_type` prevents cross-contamination of retrieval context.

---

## 🎤 Demo Script for Hiring Managers

> *"This system ingests FCA regulatory documents and answers compliance questions with mandatory source citations and automated hallucination detection. Let me show you..."*

1. **Semantic query**: `"What are the five conduct rules under SM&CR?"`  
   → Cited answer with exact document and page.

2. **Exact rule reference**: `"COBS 4.2.1 requirements"` (Phase 2 BM25 shines here)  
   → Shows hybrid retrieval catching exact regulatory identifiers.

3. **Guardrail demo**: `"What is the current FCA base rate?"`  
   → `INSUFFICIENT_EVIDENCE` — system refuses to speculate outside corpus.

4. **Phase 2 comparison**: Same question with `use_hybrid: false` vs `true`  
   → Shows measurable retrieval improvement.

5. **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)  
   → Auto-generated OpenAPI docs from Pydantic models — bank IT teams love this.

---

## 🛣️ Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| **Phase 1** | Core RAG: ingest, chunk, embed, generate, guardrails | ✅ Complete |
| **Phase 2** | Hybrid BM25 + Vector retrieval, cross-encoder re-ranker, v2 prompt | ✅ Complete |
| **Phase 3** | RAGAS eval, 25-question golden dataset, GitHub Actions CI gate | ✅ Complete |
| **Phase 4** | Streaming responses, async retrieval, Docker + Kubernetes deploy | 🔜 Next |

---

## 📄 Ingesting Real FCA Documents

```bash
# FCA Handbook chapters
python src/ingest.py --dir data/raw/handbook --doc-type handbook

# FCA Policy Statements
python src/ingest.py --dir data/raw/policy_statements --doc-type policy

# Internal bank policies
python src/ingest.py --dir data/raw/internal --doc-type internal
```

**Supported doc types**: `handbook` · `policy` · `guidance` · `internal` · `basel` · `mifid`

After ingesting new documents, BM25 index rebuilds automatically on the next query.
