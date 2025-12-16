# Biomedical Search & Reranking Engine

This repository contains a modular semantic search engine for biomedical literature. It uses **Sentence-Transformers** for embedding generation, **FAISS** for fast retrieval, and **Google Gemini** for reasoning-based reranking.

## 📂 Structure
- `src/`: Source code for data loading, embedding, searching, and reranking.
- `data/`: Contains sample datasets and cached embeddings.
- `main.py`: CLI entry point.

## 🚀 Setup

1. **Clone the repository:**
   git clone <your-repo-url>
   cd biomedical-search-engine

2. **Install dependencies:**
   pip install -r requirements.txt

3. **Configure API Key:**
   Create a .env file in the root directory or add your Google Gemini API key in .env
   GEMINI_API_KEY=your_actual_api_key_here

## 🏃 Usage

Run the search engine using the command line:
- python -m src.main --query "Is mitochondrial dysfunction associated with aging?"
