
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

| Method                           | Pros                                         | Cons                             |
| -------------------------------- | -------------------------------------------- | -------------------------------- |
| **Vector Search**                | Captures meaning, excellent for paraphrasing | Struggles with rare domain words |
| **BM25 Sparse Search**           | Strong keyword matching                      | Misses semantic relevance        |
| **Hybrid Search**                | Combines both for best accuracy              | Requires score normalization     |
