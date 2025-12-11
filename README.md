## 🚀 Hybrid Search Retrieval System
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue.svg"/> <img src="https://img.shields.io/badge/RAG-Hybrid%20Search-brightgreen.svg"/> <img src="https://img.shields.io/badge/Framework-LangChain-orange.svg"/> <img src="https://img.shields.io/badge/VectorDB-Pinecone-blueviolet.svg"/> <img src="https://img.shields.io/badge/Agents-CrewAI-red.svg"/> <img src="https://img.shields.io/badge/License-MIT-purple.svg"/> </p>

## 📌 Overview

This project implements a full end-to-end Hybrid Search RAG pipeline using:
*   Pinecone Vector Database (dense embeddings)
*   BM25 Sparse Search (lexical retrieval)
*   LangChain (embedding, text splitting, retrieval, LLM pipeline)
*   CrewAI Agents (automated RAG workflow orchestration)
*   Hybrid search = Semantic + Sparse BM25 scoring, fused together to maximize accuracy, especially for technical or domain-heavy datasets.
  
The core logic is demonstrated inside a single notebook:
📄 experiments.ipynb

BM25 values are precomputed in:
📄 bm25_values.json

---

## ⚙️ Tech Stack

🧠 LLM & Embeddings

LangChain

Sentence Transformers or Any embedding model

🗂 Vector Store

PineconeDB (namespace-level storage, metadata filtering)

🔎 Sparse Search

BM25 using rank-bm25

🤖 Autonomous Agents

CrewAI

Multi-role retrieval & reasoning agent design

---
## ⚙️ Installation
1️⃣ Clone the Repo

git clone https://github.com/anjaliy11/Hybrid_Search_RAG.git



## 2️⃣ Create Virtual Environment

python -m venv venv

source venv/bin/activate      ## macOS / Linux

venv\Scripts\activate          ## Windows

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
