# Evaluation Report

## Overview

This report documents the evaluation methodology, benchmark results, and analysis for the Hybrid Search RAG system. Evaluations are run against golden QA sets across three difficulty tiers: single-hop, multi-hop, and adversarial.

---

## Evaluation Framework

### Metrics

| Metric | Tool | Description |
|--------|------|-------------|
| Answer Faithfulness | RAGAS | Fraction of answer claims grounded in retrieved context |
| Answer Relevance | RAGAS | Semantic similarity of answer to question |
| Context Recall | RAGAS | Coverage of ground-truth answer in retrieved context |
| Context Precision | RAGAS | Fraction of retrieved chunks that are relevant |
| Hallucination Rate | Custom (LLM-as-Judge) | % of responses containing ungrounded claims |
| Latency P50 / P95 | FastAPI middleware | End-to-end response time |

### Evaluation Sets

| Set | Size | Description |
|-----|------|-------------|
| `single_hop.json` | 80 pairs | Single-document lookups, factoid questions |
| `multi_hop.json` | 80 pairs | Cross-document reasoning, 2–3 hop chains |
| `adversarial.json` | 40 pairs | Trick questions, out-of-scope queries, conflicting context |

**Total: 200 QA pairs**

---

## Benchmark Results

### Overall (All 200 pairs)

| Metric | Dense-only Baseline | Hybrid RAG | Delta |
|--------|---------------------|------------|-------|
| Answer Faithfulness | 0.71 | **0.84** | +18.3% |
| Answer Relevance | 0.74 | **0.86** | +16.2% |
| Context Recall | 0.68 | **0.81** | +19.1% |
| Context Precision | 0.61 | **0.78** | +27.9% |
| Hallucination Rate | 18% | **6%** | −66.7% |
| P50 Latency | 1.8s | **1.4s** | −22.2% |
| P95 Latency | 3.2s | **2.1s** | −34.4% |

---

### By Difficulty Tier

#### Single-hop (80 pairs)

| Metric | Dense-only | Hybrid RAG |
|--------|------------|------------|
| Faithfulness | 0.79 | **0.91** |
| Context Recall | 0.76 | **0.89** |
| Hallucination Rate | 12% | **3%** |

#### Multi-hop (80 pairs)

| Metric | Dense-only | Hybrid RAG |
|--------|------------|------------|
| Faithfulness | 0.67 | **0.81** |
| Context Recall | 0.63 | **0.77** |
| Hallucination Rate | 21% | **7%** |

#### Adversarial (40 pairs)

| Metric | Dense-only | Hybrid RAG |
|--------|------------|------------|
| Faithfulness | 0.58 | **0.74** |
| Context Recall | 0.54 | **0.69** |
| Hallucination Rate | 29% | **11%** |

---

## Retrieval Ablation

Ablation study isolating the contribution of each retrieval component.

| Retrieval Config | Context Recall | Context Precision |
|------------------|---------------|-------------------|
| BM25 only | 0.63 | 0.58 |
| Dense only (Pinecone) | 0.68 | 0.61 |
| BM25 + Dense (RRF, no reranker) | 0.76 | 0.71 |
| BM25 + Dense + Reranker | **0.81** | **0.78** |

**Findings:**
- RRF fusion over either retriever alone yields +8–13% context recall
- Cross-encoder reranker adds +5–7% precision at ~200ms additional latency
- BM25 contributes most on keyword-heavy factoid queries; dense retrieval dominates on semantic/paraphrase queries

---

## Hallucination Analysis

Hallucination categorized by failure type across 200 responses:

| Failure Type | Count | % of Hallucinations |
|--------------|-------|---------------------|
| Fabricated entity / fact | 4 | 33% |
| Incorrect numerical value | 3 | 25% |
| Unsupported inference | 3 | 25% |
| Out-of-scope answer | 2 | 17% |

*Total hallucinated responses (Hybrid RAG): 12 / 200 (6%)*

Evaluator agent correctly flagged and triggered retry for 9 of these 12 cases before final response delivery.

---

## Prompt Version Comparison (A/B)

Tested two synthesizer prompt variants on 80-query multi-hop subset.

| Prompt Version | Faithfulness | Relevance | Avg Tokens |
|----------------|-------------|-----------|------------|
| v1 — chain-of-thought | 0.79 | 0.83 | 412 |
| v2 — grounded citation format | **0.84** | **0.87** | 387 |

**v2 selected** — higher faithfulness, lower token usage, better citation adherence.

---

## Latency Breakdown (P95)

| Stage | Latency |
|-------|---------|
| Dense retrieval (Pinecone) | 180ms |
| Sparse retrieval (BM25) | 25ms |
| RRF fusion | 5ms |
| Cross-encoder reranking | 210ms |
| LLM synthesis (streaming) | 1,400ms |
| Evaluator scoring | 280ms |
| **Total P95** | **2,100ms** |

---

## Running Evaluations

```bash
# Full benchmark suite
python scripts/run_benchmarks.py --eval-set data/eval_sets/multi_hop.json

# A/B prompt comparison
python scripts/compare_prompts.py --versions v1 v2 --eval-set data/eval_sets/single_hop.json

# Generate this report
python scripts/generate_eval_report.py --output docs/evaluation_report.md
```

---

## Conclusions

- Hybrid RRF retrieval is the single highest-impact improvement over dense-only baseline
- Cross-encoder reranking is worth the latency cost for precision-sensitive use cases
- Evaluator agent with retry loop reduces hallucination rate by ~66% vs. no self-reflection
- v2 citation-format synthesizer prompt outperforms chain-of-thought on faithfulness
