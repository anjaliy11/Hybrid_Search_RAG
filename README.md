## 🚀 Hybrid Search Retrieval System
Semantic + Lexical Search for High-Accuracy Information Retrieval
<p align="center"> <img src="https://img.shields.io/badge/Language-Python%203.10+-blue.svg"/> <img src="https://img.shields.io/badge/Notebook-Jupyter-orange.svg"/> <img src="https://img.shields.io/badge/Search-Hybrid%20(BM25%20%2B%20Embeddings)-green.svg"/> <img src="https://img.shields.io/badge/License-MIT-purple.svg"/> <img src="https://img.shields.io/github/last-commit/your-username/your-repo-name?color=yellow"/> </p>

## 📌 Overview

This repository contains an experimental implementation of a Hybrid Search Retrieval System, combining:

Semantic Vector Embeddings (meaning-based search)

BM25 Sparse Text Scoring (keyword-based search)

Hybrid search significantly improves retrieval performance for RAG systems, knowledge bases, question-answering, and domain-specific search engines.

The core logic is demonstrated inside a single notebook:
📄 experiments.ipynb

BM25 values are precomputed in:
📄 bm25_values.json

## 🧠 Why Hybrid Search?


## 🧬 Architecture

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
┌─────────┐       ┌──────────────┐
│ BM25    │       │ Embeddings   │
│ Sparse  │       │ (Semantic)   │
│ Scores  │       │ Vectors      │
└────┬────┘       └──────┬──────┘
     │                   │
     └──────┬────────────┘
            ▼
  ┌────────────────────────────┐
  │  Hybrid Score Fusion       │
  │ (α * semantic + β * bm25)  │
  └───────────┬────────────────┘
              │
              ▼
     ┌──────────────────────┐
     │  Ranked Search Result │
     └──────────────────────┘

---



## ⚙️ Installation
1️⃣ Clone the Repo

git clone https://github.com/anjaliy11/Hybrid_Search_RAG.git

cd hybrid-search

## 2️⃣ Create Virtual Environment

python -m venv venv

source venv/bin/activate   # macOS / Linux

venv\Scripts\activate      # Windows

## 3️⃣ Install Requirements
pip install -r requirements.txt

## ▶️ Usage
Run Jupyter Notebook

jupyter notebook experiments.ipynb


Inside the notebook you can:

✔ Load dataset
✔ Generate embeddings
✔ Load BM25 scores (or compute new ones)
✔ Perform vector, BM25, and hybrid searches
✔ Compare retrieval accuracy

## 🔍 Hybrid Score Formula
Weighted Hybrid Similarity:

HybridScore = α * SemanticScore + β * BM25Score


Where:

SemanticScore = cosine similarity of embeddings

BM25Score = sparse lexical score

α, β = tunable parameters (default: 0.5 each)

## 📊 Applications

This hybrid retrieval engine is ideal for:

RAG systems (Retrieval-Augmented Generation)

LLM chatbots with enhanced factual grounding

Search engines

Academic / legal / clinical document search

Enterprise knowledge management

FAQ and customer support bots

## 🚀 Sample Queries
"What is hybrid information retrieval?"
"Explain lexical vs semantic search"
"Find documents about index fusion technique"


The hybrid engine returns highly relevant results even when:

The query has paraphrasing

Documents use rare technical terms

There are multi-topic overlaps

## 📈 Future Improvements

🔹 Add re-rankers (BGE, ColBERT, Cross-Encoders)
🔹 Integrate Pinecone / Weaviate for vector storage
🔹 Build API (FastAPI) for hybrid search
🔹 Add Streamlit UI dashboard
🔹 Add Recall@k and nDCG evaluation metrics

## 🤝 Contributing

Pull requests and suggestions are welcome!
Please open an issue before major changes.
