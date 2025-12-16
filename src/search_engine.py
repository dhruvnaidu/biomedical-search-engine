import faiss
import numpy as np
import pandas as pd

class SearchEngine:
    def __init__(self, embeddings, metadata_df):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype("float32"))
        self.df = metadata_df.reset_index(drop=True)

    def search(self, query_vector, top_k=5):
        D, I = self.index.search(query_vector.astype("float32"), top_k)
        
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.df):
                row = self.df.iloc[idx]
                results.append({
                    "title": row.get("QUESTION", "No Title"),
                    "text": str(row.get("text", ""))[:500],
                    "raw_row": row
                })
        return results