import numpy as np
import pandas as pd
from tqdm import tqdm

class Evaluator:
    def __init__(self, search_engine):
        self.search_engine = search_engine

    def generate_known_item_ground_truth(self, sample_size=100):
        """
        Generates synthetic ground truth for evaluation.
        Assumption: The 'QUESTION' in the paper is the query, and the paper ID is the relevant doc.
        """
        df = self.search_engine.df
        # Filter for rows that actually have a question
        valid_df = df[df['QUESTION'].notna() & (df['QUESTION'] != "")]
        
        if len(valid_df) > sample_size:
            sample = valid_df.sample(sample_size, random_state=42)
        else:
            sample = valid_df
            
        ground_truth = {}
        for doc_id, row in sample.iterrows():
            query = row['QUESTION']
            # Dict mapping Query -> List of Relevant IDs
            ground_truth[query] = [doc_id] 
            
        return ground_truth

    def calculate_metrics(self, retrieved_ids, relevant_ids, k):
        """Calculates Precision@K, Recall@K, MRR@K, NDCG@K"""
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        # Precision @ K
        intersection = retrieved_set.intersection(relevant_set)
        precision = len(intersection) / k
        
        # Recall @ K
        recall = len(intersection) / len(relevant_set) if len(relevant_set) > 0 else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                mrr = 1 / (i + 1)
                break
                
        # NDCG (Normalized Discounted Cumulative Gain)
        dcg = 0
        idcg = 0
        
        # Calculate DCG
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                dcg += 1 / np.log2(i + 2) # i+2 because rank starts at 1 (log2(rank+1))
        
        # Calculate Ideal DCG (IDCG) - assuming we retrieved all relevant docs at the top
        num_relevant = len(relevant_ids)
        for i in range(min(num_relevant, k)):
            idcg += 1 / np.log2(i + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            f"Precision@{k}": precision,
            f"Recall@{k}": recall,
            f"MRR@{k}": mrr,
            f"NDCG@{k}": ndcg
        }

    def evaluate(self, ground_truth, k=5, method="hybrid"):
        """Runs evaluation over the ground truth set"""
        metrics_sum = {f"Precision@{k}": 0, f"Recall@{k}": 0, f"MRR@{k}": 0, f"NDCG@{k}": 0}
        
        print(f"Running evaluation on {len(ground_truth)} queries using method: {method.upper()}...")
        
        for query, relevant_ids in tqdm(ground_truth.items()):
            # Run Search
            results = self.search_engine.search(query, top_k=k, method=method)
            retrieved_ids = [res['id'] for res in results]
            
            # Calculate Metrics for this query
            query_metrics = self.calculate_metrics(retrieved_ids, relevant_ids, k)
            
            # Aggregate
            for key in metrics_sum:
                metrics_sum[key] += query_metrics[key]
                
        # Average
        final_metrics = {k: v / len(ground_truth) for k, v in metrics_sum.items()}
        return final_metrics