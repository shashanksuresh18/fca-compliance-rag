# FCA Compliance RAG System — README

# 🏦 FCA Compliance RAG System

**Production-grade Retrieval-Augmented Generation for UK Banking Regulatory Compliance**

> Built as a portfolio project for AI Engineer roles at UK banks (HSBC, Barclays, Lloyds, NatWest).  
> Answers FCA regulatory questions grounded in real documentation — with mandatory citations and zero-hallucination guardrails.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
 [FastAPI /query]
     │
     ▼
 [Retriever]  ──── ChromaDB (sentence-transformers embeddings)
     │                   └── FCA Handbook, Policy Statements, MiFID II, SM&CR ...
     │
     ▼
 [Generator]  ──── Groq LLM (llama-3.3-70b) — FREE, 200 tok/sec
     │
     ▼
 [Guardrails]
  ✓ Guardrail 1: No chunks retrieved → DECLINE (never calls LLM)
  ✓ Guardrail 2: No [Source: ...] citation → REJECT answer
     │
     ▼
 Grounded Answer + Citations
```

---

## 💰 Free Stack (No Credit Card Required)

| Component | Technology | Cost |
|-----------|-----------|------|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) | **$0** |
| Vector Store | ChromaDB (local, on-disk) | **$0** |
| LLM | Groq API — `llama-3.3-70b-versatile` | **$0** (14.4k req/day free) |
| API | FastAPI + Uvicorn | **$0** |

---

## ⚡ Quickstart

### 1. Get your free Groq API key
Sign up at **[console.groq.com](https://console.groq.com)** — no credit card needed.

### 2. Install dependencies
```bash
cd "Project Day 1_ RAG"
pip install -r requirements.txt
```
> ⚠️ First run downloads the ~90MB `all-MiniLM-L6-v2` model locally. Subsequent runs use cache.

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### 4. Ingest the mock documents
```bash
python src/ingest.py --dir data/mock --doc-type internal
```

### 5. Start the API
```bash
uvicorn api.main:app --reload
```

### 6. Ask a compliance question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the conduct rules under SM&CR?"}'
```

**Expected response** (condensed):
```json
{
  "answer": "Under SM&CR, there are five Individual Conduct Rules...\n[Source: mock_sm_cr_policy.md, Page 2]",
  "citations": [{"source_file": "mock_sm_cr_policy.md", "page_number": 2, "doc_type": "Internal Bank Policy"}],
  "declined": false,
  "prompt_version": "v1",
  "chunks_retrieved": 5
}
```

### 7. Test the decline guardrail
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital gains tax rate on Bitcoin?"}'
```

**Expected**: `"declined": true` with `INSUFFICIENT_EVIDENCE` message.

---

## 📁 Project Structure

```
fca-rag/
├── src/
│   ├── config.py       # Pydantic settings (validated at startup)
│   ├── ingest.py       # PDF/Markdown ingestion + chunking (600 tok, 100 overlap)
│   ├── store.py        # ChromaDB with dedup-safe upsert
│   ├── retriever.py    # Dense vector retrieval + metadata filtering
│   └── generator.py   # LLM inference + 2 citation guardrails ⚠️
├── api/
│   └── main.py         # FastAPI: /query, /ingest, /health
├── prompts/
│   └── v1/
│       └── qa_prompt.yaml   # Version-controlled prompt template
├── data/
│   └── mock/           # Sample FCA-style demo documents
├── tests/
│   └── test_pipeline.py     # Unit tests (no API keys needed)
├── .env.example
└── requirements.txt
```

---

## 🧪 Run Tests (No API Keys Needed)

```bash
python -m pytest tests/test_pipeline.py -v
```

Tests cover:
- ✅ Chunk size bounds (500–800 chars for FCA legal text)
- ✅ Chunk metadata completeness (source file, page number, doc type)
- ✅ Guardrail 1: empty context → decline (no LLM called)
- ✅ Guardrail 2: answer without citation → rejected
- ✅ Happy path: cited answer passes through
- ✅ Citation regex pattern correctness

---

## 🔑 Why These Design Choices Matter in Banking

### Chunking (600 tokens / 100 overlap)
FCA rules frequently cross-reference earlier sub-sections within the same rule. Chunks that are too small lose this context; chunks that are too large dilute embedding precision. 100-token overlap ensures sentences split at boundaries are fully captured.

### Citation Guardrails (Two Layers)
1. **Empty context check**: If ChromaDB returns nothing, we never call the LLM. This prevents the model from answering from training data — which may contain outdated or hallucinated rule numbers.
2. **Post-generation regex check**: Even if context exists, we verify the response contains at least one `[Source: filename, Page N]` tag. An uncited answer is silently rejected. In a regulated environment, a wrong answer is a legal risk — so no citation = no answer.

### Prompt Versioning (YAML)
Every prompt change is a Git commit. Regulators can ask "what exact prompt was used in March 2026?" and receive a precise answer. CI tests run on every prompt change before deployment.

### Metadata Filtering
Different teams query different document scopes. Legal team queries only the FCA Handbook; product teams query Policy Statements. The `doc_type` filter prevents cross-contamination of context.

---

## 🎤 Demo Script for Hiring Managers

> *"This system ingests FCA regulatory documents — PDF or Markdown — and answers compliance questions with mandatory source citations. Watch what happens with an answerable question..."*

1. **Happy path**: `"What are the five conduct rules under SM&CR?"`  
   → Cited answer with exact document and page number.

2. **Metadata filter**: Same question with `"doc_type": "handbook"` vs `"doc_type": "internal"`  
   → Shows document-scoped retrieval for team-specific compliance use cases.

3. **Guardrail demo**: `"What is the FCA's view on meme coins?"`  
   → `INSUFFICIENT_EVIDENCE` — system refuses to speculate.

4. **Swagger UI**: Show `http://localhost:8000/docs`  
   → Interactive API documentation auto-generated from Pydantic models.

---

## 🛣️ Roadmap

| Phase | Status |
|-------|--------|
| **Phase 1**: Core RAG Pipeline | ✅ Complete |
| **Phase 2**: Hybrid BM25 + Vector Retrieval, Re-ranker | 🔜 Next |
| **Phase 3**: RAGAS Evaluation + CI Gating (faithfulness ≥ 0.85) | 🔜 Soon |

---

## 📄 Ingesting Real FCA Documents

To ingest real FCA PDFs from [fca.org.uk](https://fca.org.uk):

```bash
# FCA Handbook (saves to data/raw/)
python src/ingest.py --dir data/raw/handbook --doc-type handbook

# FCA Policy Statements
python src/ingest.py --dir data/raw/policy_statements --doc-type policy
```

**Document types**: `handbook`, `policy`, `guidance`, `internal`, `basel`, `mifid`
