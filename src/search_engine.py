import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from .embeddings import EmbeddingGenerator  # FIXED: Changed from EmbeddingModel

class SearchEngine:
    def __init__(self, df, embeddings, embedding_model):
        self.df = df
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        
        # 1. Initialize Vector Search (FAISS)
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings.astype("float32")) # Ensure float32 for FAISS
        
        # 2. Initialize Keyword Search (BM25)
        print("🏗️ Building BM25 index...")
        # Simple tokenization: lowercase and split by whitespace
        tokenized_corpus = [str(doc).lower().split() for doc in self.df["text"].tolist()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ BM25 index built.")

    def search_vector(self, query, top_k=10):
        """Standard Semantic Search using FAISS"""
        # Use encode_query directly from EmbeddingGenerator
        query_embedding = self.embedding_model.encode_query(query)
        
        D, I = self.index.search(query_embedding.astype("float32"), top_k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.df):
                row = self.df.iloc[idx]
                results.append({
                    "id": self.df.index[idx],
                    "title": row.get("QUESTION", "No Title"), # Add title for display
                    "text": str(row.get("text", "")),
                    "score": float(D[0][i]),
                    "rank": i + 1
                })
        return results

    def search_bm25(self, query, top_k=10):
        """Keyword Search using BM25"""
        tokenized_query = query.lower().split()
        # Get top_k scores
        scores = self.bm25.get_scores(tokenized_query)
        # Get indices of top_k scores
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_n_indices):
            row = self.df.iloc[idx]
            results.append({
                "id": self.df.index[idx],
                "title": row.get("QUESTION", "No Title"),
                "text": str(row.get("text", "")),
                "score": float(scores[idx]),
                "rank": i + 1
            })
        return results

    def search_hybrid(self, query, top_k=10):
        """
        Hybrid Search using Reciprocal Rank Fusion (RRF).
        Combines ranks from BM25 and Vector search.
        """
        # Get more candidates for fusion than the final top_k
        candidates_k = top_k * 2
        
        vector_results = self.search_vector(query, candidates_k)
        bm25_results = self.search_bm25(query, candidates_k)
        
        # RRF Dictionary: {doc_id: rrf_score}
        rrf_scores = {}
        k_const = 60 # Standard RRF constant
        
        # Process Vector Ranks
        for rank, res in enumerate(vector_results):
            doc_id = res['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k_const + rank + 1))
            
        # Process BM25 Ranks
        for rank, res in enumerate(bm25_results):
            doc_id = res['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k_const + rank + 1))
            
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        
        # Retrieve full details for sorted results
        final_results = []
        for rank, doc_id in enumerate(sorted_ids):
            # We need to find the original row data. 
            # Since doc_id is the index label, we use loc
            row = self.df.loc[doc_id]
            final_results.append({
                "id": doc_id,
                "title": row.get("QUESTION", "No Title"),
                "text": str(row.get("text", "")),
                "score": rrf_scores[doc_id],
                "rank": rank + 1,
                "method": "hybrid_rrf"
            })
            
        return final_results

    def search(self, query, top_k=10, method="hybrid"):
        if method == "vector":
            return self.search_vector(query, top_k)
        elif method == "bm25":
            return self.search_bm25(query, top_k)
        else:
            return self.search_hybrid(query, top_k)