# 🧬 Biomedical Search & Reranking Engine

A modular semantic search engine for **biomedical literature**. This system combines dense embeddings, fast vector search, keyword retrieval, and LLM-based reranking to deliver high-quality, relevant results.

---

## ✨ Features

* 🔍 **Semantic Search** using Sentence-Transformers
* ⚡ **Fast Retrieval** with FAISS
* 🧠 **Reasoning-based Reranking** using Google Gemini
* 🔗 **Hybrid Search** (Vector + BM25 via RRF)
* 📊 **Evaluation Suite** with Precision, Recall, MRR, and NDCG

---

## 🚀 Setup

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd biomedical-search-engine
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Google Gemini API Key

Create a `.env` file in the project root and add your API key:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

> ⚠️ **Note:** Ensure `.env` is added to `.gitignore` to avoid exposing your API key.

---

## 🏃 Usage

### 🔎 Search Modes

You can perform searches using **three retrieval strategies**:

* **Vector Search** (FAISS)
* **Keyword Search** (BM25)
* **Hybrid Search** (Reciprocal Rank Fusion – *Recommended*)

---

### 🔥 Hybrid Search (Recommended)

```bash
python -m src.main --query "Is mitochondrial dysfunction associated with aging?" --method hybrid
```

---

### 📐 Vector-Only Search (FAISS)

```bash
python -m src.main --query "Is mitochondrial dysfunction associated with aging?" --method vector
```

---

### 🔑 Keyword-Only Search (BM25)

```bash
python -m src.main --query "Is mitochondrial dysfunction associated with aging?" --method bm25
```

---

## 📊 Evaluation

The engine includes an **evaluation mode** to benchmark retrieval quality using:

* Precision
* Recall
* Mean Reciprocal Rank (MRR)
* Normalized Discounted Cumulative Gain (NDCG)

A synthetic **ground-truth dataset** is generated from paper titles and questions.

---

### 🧪 Evaluate Hybrid Search

(Default: 50 samples)

```bash
python -m src.main --evaluate --method hybrid --samples 50
```

---

### 🧪 Evaluate BM25 Search

```bash
python -m src.main --evaluate --method bm25 --samples 100
```

---

## 📌 Notes

* Hybrid search typically yields the best overall ranking performance.
* Gemini reranking improves relevance by reasoning over retrieved abstracts.
* Cached embeddings are reused to improve runtime efficiency.

---

## 🙌 Acknowledgements

* Sentence-Transformers
* FAISS
* Google Gemini
* BM25 / IR community
