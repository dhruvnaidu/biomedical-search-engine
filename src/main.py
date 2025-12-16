import argparse
import os
import pandas as pd
import numpy as np
from .data_loader import DataLoader
from .embeddings import EmbeddingGenerator
from .search_engine import SearchEngine
from .reranker import GeminiReranker
from .config import PROCESSED_TEXTS_PATH, EMBEDDINGS_PATH, DATA_PATH

def main():
    parser = argparse.ArgumentParser(description="Biomedical Search Engine")
    parser.add_argument("--query", type=str, required=True, help="The biomedical question to answer")
    
    # Defaults to DATA_PATH from config.py (ori_pqal.json)
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to input JSON data")
    
    parser.add_argument("--reindex", action="store_true", help="Force regeneration of embeddings")
    
    args = parser.parse_args()

    # 1. Load Data
    # Check if we have cached embeddings to save time
    if os.path.exists(PROCESSED_TEXTS_PATH) and os.path.exists(EMBEDDINGS_PATH) and not args.reindex:
        print("📂 Loading cached embeddings and data...")
        df = pd.read_csv(PROCESSED_TEXTS_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        embedder = EmbeddingGenerator() # Initialize to load model for query encoding
    else:
        # Load and Process Fresh Data from the JSON file
        loader = DataLoader(args.data)
        df = loader.load_and_process()
        
        # Generate Embeddings
        embedder = EmbeddingGenerator()
        embeddings = embedder.generate(df["text"].tolist())
        
        # Save Cache
        df.to_csv(PROCESSED_TEXTS_PATH, index=False)
        np.save(EMBEDDINGS_PATH, embeddings)
        print("💾 Embeddings saved.")

    # 2. Initial Semantic Search
    searcher = SearchEngine(embeddings, df)
    query_vec = embedder.encode_query(args.query)
    initial_results = searcher.search(query_vec, top_k=5)
    
    print(f"\n🔍 Found {len(initial_results)} initial matches via FAISS.")

    # 3. LLM Reranking
    reranker = GeminiReranker()
    final_results = reranker.rerank(args.query, initial_results)

    # 4. Output
    print("\n" + "="*50)
    print(f"🤖 FINAL ANSWER for: {args.query}")
    print("="*50)
    
    if "ranked_results" in final_results and final_results["ranked_results"]:
        for item in final_results["ranked_results"]:
            print(f"\n📄 Title: {item.get('title')}")
            print(f"⭐ Score: {item.get('relevance_score')}")
            print(f"💡 Reason: {item.get('relevance_reason')}")
    else:
        # Fallback if LLM fails
        print("Raw Search Results (LLM processing unavailable):")
        for res in initial_results:
             print(f"- {res['title']}")

if __name__ == "__main__":
    main()