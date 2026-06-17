# Hybrid Search RAG — Production Agentic Retrieval System

A multi-turn agentic dialogue system built on hybrid dense-sparse retrieval, hierarchical agent coordination, and rubric-based evaluation. Designed for real-world deployment with FastAPI serving, streaming responses, and full observability.

---

## Architecture Overview

```
User Query
    │
    ▼
Supervisor Agent          ← Routes & orchestrates
    │
    ├── Researcher Agent  ← Query decomposition & planning
    │
    ├── Retriever Agent   ← Hybrid BM25 + Pinecone (RRF fusion)
    │       └── Reranker  ← Cross-encoder reranking
    │
    ├── Synthesizer Agent ← Grounded response generation
    │
    └── Evaluator Agent   ← Self-reflection, hallucination detection, RAGAS scoring
```

**Tech Stack:** LangGraph · Pinecone · BM25 · FastAPI · Docker · LangSmith · RAGAS

---

## Key Features

- **Hybrid Retrieval** — Reciprocal Rank Fusion over dense (Pinecone) + sparse (BM25) indexes; cross-encoder reranking for precision
- **Agentic Orchestration** — LangGraph-based supervisor→researcher→retriever→synthesizer→evaluator pipeline with shared state
- **Multi-turn Memory** — Conversation buffer, entity tracking, sliding-window summarization across turns
- **Hallucination Detection** — Faithfulness grounding constraints + LLM-as-Judge scoring before response delivery
- **Rubric-Based Evaluation** — RAGAS metrics (faithfulness, relevance, completeness) + custom A/B prompt comparison
- **Production Serving** — FastAPI with SSE streaming, rate limiting, latency middleware, `/chat` `/ingest` `/evaluate` `/health` endpoints
- **Observability** — LangSmith tracing, OpenTelemetry, token cost tracking, structured logging

---

## Project Structure

```
Hybrid_Search_RAG/
├── config/
│   ├── settings.py              # Pydantic settings (env-driven)
│   ├── prompts/                 # Per-agent YAML prompt templates
│   └── eval_rubrics/            # Faithfulness / relevance / completeness rubrics
│
├── src/
│   ├── ingestion/               # PDF/web loaders, semantic chunker, BM25 + Pinecone indexing
│   ├── retrieval/               # Dense, sparse, hybrid (RRF), reranker, context builder
│   ├── agents/                  # LangGraph state, supervisor, researcher, retriever,
│   │                            #   synthesizer, evaluator, voice agent, graph definition
│   ├── evaluation/              # RAGAS, LLM-as-Judge, rubric scorer, hallucination, A/B
│   ├── memory/                  # Conversation memory, entity memory, summary buffer
│   ├── serving/                 # FastAPI app, routes, SSE streaming, middleware, schemas
│   └── utils/                   # Logging, tracing, cost tracker, retry utilities
│
├── tests/
│   ├── unit/                    # Chunker, retriever, state, rubric scorer
│   ├── integration/             # Ingestion pipeline, agent graph, retrieval chain
│   └── e2e/                     # Chat endpoint, multi-turn flows
│
├── scripts/                     # Ingest docs, run benchmarks, compare prompts, eval reports
├── notebooks/                   # Data exploration, retrieval experiments, agent debugging
├── data/
│   └── eval_sets/               # Golden QA pairs: single-hop, multi-hop, adversarial
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/anjaliy11/Hybrid_Search_RAG.git
cd Hybrid_Search_RAG
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, LANGCHAIN_API_KEY
```

### 3. Ingest documents

```bash
python scripts/ingest_documents.py --source data/raw/
```

### 4. Run the API

```bash
uvicorn src.serving.app:app --reload --port 8000
```

### 5. Docker (recommended for production)

```bash
docker-compose up --build
```

API available at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Multi-turn streaming chat (SSE) |
| `POST` | `/ingest` | Ingest and index new documents |
| `POST` | `/evaluate` | Run RAGAS + rubric evaluation on a query set |
| `GET` | `/health` | Service health check |

**Example — streaming chat:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain hybrid retrieval", "session_id": "abc123"}'
```

---

## Evaluation

```bash
python scripts/run_benchmarks.py --eval-set data/eval_sets/multi_hop.json
python scripts/compare_prompts.py --versions v1 v2
python scripts/generate_eval_report.py --output docs/evaluation_report.md
```

**Metrics tracked:** Faithfulness · Answer Relevance · Context Recall · Hallucination Rate · Latency (P50/P95)

---

## Results

| Metric | Baseline (Dense-only) | Hybrid RAG |
|--------|-----------------------|------------|
| Answer Faithfulness | 0.71 | **0.84** |
| Context Recall | 0.68 | **0.81** |
| Hallucination Rate | 18% | **6%** |
| P95 Latency | 3.2s | **2.1s** |

*Evaluated on 200-query adversarial test set.*

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Orchestration | LangGraph, LangChain |
| Vector Store | Pinecone |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranking | Cross-encoder (sentence-transformers) |
| Evaluation | RAGAS, custom LLM-as-Judge |
| Serving | FastAPI, SSE streaming |
| Observability | LangSmith, OpenTelemetry |
| Deployment | Docker, Docker Compose |

---

## Roadmap

- [ ] GraphRAG integration for entity-relationship traversal
- [ ] Adaptive chunking based on document structure
- [ ] Multi-modal retrieval (image + text)
- [ ] Cost-aware retrieval routing (cheap sparse → expensive dense fallback)

---

## Author

**Anjali Yadav**


---

*Built to demonstrate production-grade agentic RAG engineering — covering hybrid retrieval, multi-agent coordination, hallucination mitigation, and LLMOps observability.*
