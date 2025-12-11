The core logic is demonstrated in the following notebook: 📄 experiments.ipynb. The BM25 values are precomputed and located in: 📄 bm25_values.json.

## 🧠 Why Hybrid Search?

| Method                           | Pros                                         | Cons                             |
| -------------------------------- | -------------------------------------------- | -------------------------------- |
| **Vector Search**                | Captures meaning, excellent for paraphrasing | Struggles with rare domain words |
| **BM25 Sparse Search**           | Strong keyword matching                      | Misses semantic relevance        |
| **Hybrid Search**                | Combines both for best accuracy              | Requires score normalization     |

---

## Architecture

## Hybrid Retrieval Pipeline

 ┌───────────────────┐
 │   Text Corpus      │
 └─────────┬─────────┘
           │
           ▼
 ┌───────────────────┐
 │ Preprocessing &    │
 │ Tokenization       │
 └─────────┬─────────┘
           │
     ┌─────┴─────────────┐
     │                   │
     ▼                   ▼
