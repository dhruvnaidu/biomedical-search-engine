import argparse
import os
import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.embeddings import EmbeddingGenerator  # FIXED: Changed from EmbeddingModel
from src.search_engine import SearchEngine
from src.reranker import GeminiReranker
from src.config import DATA_PATH, EMBEDDINGS_PATH
from src.evaluation import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Biomedical Search Engine")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation metrics (Precision, Recall, NDCG)")
    parser.add_argument("--method", type=str, default="hybrid", choices=["vector", "bm25", "hybrid"], help="Retrieval method")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples for evaluation")
    args = parser.parse_args()

    # 1. Load Data
    loader = DataLoader(DATA_PATH)
    df = loader.load_and_process()

    # 2. Load Embeddings
    # Initialize the generator (class name is EmbeddingGenerator)
    embedder = EmbeddingGenerator() 
    
    # Check if embeddings exist
    if os.path.exists(EMBEDDINGS_PATH):
        print(f"📂 Loading embeddings from {EMBEDDINGS_PATH}...")
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        print("⏳ Generating embeddings (this may take time)...")
        embeddings = embedder.generate(df["text"].tolist())
        np.save(EMBEDDINGS_PATH, embeddings)

    # 3. Initialize Search Engine
    # We pass the 'embedder' instance so the search engine can encode queries on the fly
    search_engine = SearchEngine(df, embeddings, embedder)

    # ---------------------------
    # EVALUATION MODE
    # ---------------------------
    if args.evaluate:
        evaluator = Evaluator(search_engine)
        print(f"📊 Generating synthetic ground truth from {args.samples} random papers...")
        ground_truth = evaluator.generate_known_item_ground_truth(sample_size=args.samples)
        
        # Run Evaluation
        results = evaluator.evaluate(ground_truth, k=5, method=args.method)
        
        print("\n" + "="*30)
        print(f"EVALUATION RESULTS ({args.method.upper()})")
        print("="*30)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("="*30)
        return

    # ---------------------------
    # SEARCH MODE
    # ---------------------------
    if not args.query:
        print("❌ Please provide a --query or use --evaluate")
        return

    print(f"\n🔍 Retrieving top 10 candidates using {args.method.upper()}...")
    
    # Step 1: Retrieval
    retrieved_docs = search_engine.search(args.query, top_k=10, method=args.method)

    print("\n🤖 Re-ranking with Gemini...")
    # Step 2: Reranking
    reranker = GeminiReranker()
    reranked_docs = reranker.rerank(args.query, retrieved_docs)

    print("\n" + "="*50)
    print(f"FINAL ANSWER for: '{args.query}'")
    print("="*50)
    
    if "ranked_results" in reranked_docs and reranked_docs["ranked_results"]:
        for item in reranked_docs["ranked_results"]:
            print(f"\n📄 Title: {item.get('title')}")
            print(f"⭐ Score: {item.get('relevance_score')}")
            print(f"💡 Reason: {item.get('relevance_reason')}")
    else:
        print("Raw Search Results (LLM processing unavailable):")
        for res in retrieved_docs:
             print(f"- {res.get('title', 'No Title')}: {res.get('text')[:100]}...")

if __name__ == "__main__":
    main()