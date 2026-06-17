# System Architecture

## Overview

Hybrid Search RAG is a production-grade agentic retrieval system built around a hierarchical multi-agent pipeline. Each agent has a single responsibility; coordination is handled by a LangGraph-defined state machine with a central supervisor.

---

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        Client                               │
│              (HTTP / SSE streaming via FastAPI)             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Supervisor Agent                          │
│         Routes query → selects agent execution path        │
└────┬──────────────┬──────────────┬──────────────┬──────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
Researcher     Retriever      Synthesizer     Evaluator
  Agent          Agent          Agent           Agent
(planning)    (retrieval)   (generation)   (reflection)
```

---

## Agent Responsibilities

### Supervisor Agent (`src/agents/supervisor.py`)
- Parses incoming query and conversation history
- Decides execution path: single-hop retrieval, multi-hop decomposition, or direct synthesis
- Manages LangGraph state transitions
- Handles fallback and retry logic

### Researcher Agent (`src/agents/researcher.py`)
- Decomposes complex queries into sub-questions
- Plans retrieval strategy (single vs. multi-hop)
- Generates search queries for downstream retriever

### Retriever Agent (`src/agents/retriever_agent.py`)
- Executes hybrid retrieval (BM25 + Pinecone)
- Applies Reciprocal Rank Fusion (RRF) to merge ranked lists
- Passes fused results to cross-encoder reranker
- Builds final context window for synthesis

### Synthesizer Agent (`src/agents/synthesizer.py`)
- Generates grounded responses strictly from retrieved context
- Applies citation constraints to prevent hallucination
- Formats output for voice or text delivery

### Evaluator Agent (`src/agents/evaluator.py`)
- Scores response on faithfulness, relevance, completeness
- Triggers retry if score falls below threshold
- Logs evaluation metadata to LangSmith

---

## Retrieval Pipeline

```
Query
  │
  ├──► Dense Retrieval (Pinecone)
  │         └── text-embedding-3-small → top-k vector search
  │
  └──► Sparse Retrieval (BM25)
            └── rank-bm25 → top-k keyword match
                    │
                    ▼
           Reciprocal Rank Fusion
                    │
                    ▼
           Cross-Encoder Reranker
           (sentence-transformers)
                    │
                    ▼
           Context Window Builder
           (token-aware truncation + metadata injection)
```

### Hybrid Fusion — Reciprocal Rank Fusion (RRF)

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

- `k = 60` (smoothing constant)
- Dense and sparse lists fused before reranking
- Final top-N passed to cross-encoder for precision reranking

---

## Memory Architecture

```
Conversation Turn N
        │
        ├── Short-term: ConversationBufferMemory  (last K turns)
        ├── Entity:     EntityMemory              (named entity tracking)
        └── Long-term:  SummaryBufferMemory       (sliding window summarization)
```

All memory stores are injected into LangGraph shared state and passed to each agent on every turn.

---

## Serving Layer

```
FastAPI Application (src/serving/app.py)
│
├── POST /chat        → streams SSE tokens via StreamingResponse
├── POST /ingest      → triggers ingestion pipeline
├── POST /evaluate    → runs RAGAS + rubric eval on provided QA pairs
└── GET  /health      → liveness check
│
├── Middleware: latency tracking, rate limiting (slowapi)
└── Tracing:   LangSmith + OpenTelemetry spans per request
```

---

## Ingestion Pipeline

```
Raw Documents (PDF / web / API)
        │
        ▼
   Document Loader (src/ingestion/loader.py)
        │
        ▼
   Semantic Chunker (src/ingestion/chunker.py)
   — chunk size: 512 tokens, overlap: 64 tokens
   — metadata: source, page, timestamp, doc_id
        │
        ├──► Embedder → Pinecone upsert   (dense index)
        └──► BM25 Indexer → pickle dump   (sparse index)
```

---

## Observability

| Signal | Tool |
|--------|------|
| Traces | LangSmith (per-agent spans) |
| Metrics | OpenTelemetry → Prometheus |
| Logs | Structured JSON (src/utils/logging.py) |
| Cost | Token counter per request (src/utils/cost_tracker.py) |

---

## Deployment

```
docker-compose up --build
```

| Service | Port |
|---------|------|
| API (FastAPI) | 8000 |



---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| LangGraph over vanilla LangChain | Explicit state machine; easier to debug agent transitions |
| RRF over weighted sum | Parameter-free fusion; robust across query types |
| Cross-encoder reranker | Higher precision than bi-encoder at acceptable latency cost |
| SSE over WebSocket | Simpler client integration; sufficient for token streaming |
| BM25 + Pinecone (not Pinecone sparse) | Full control over BM25 index; avoids vendor lock-in |