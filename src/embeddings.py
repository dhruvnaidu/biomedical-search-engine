import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from .config import EMBEDDING_MODEL_NAME

class EmbeddingGenerator:
    def __init__(self):
        print(f"🟢 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def generate(self, text_list, batch_size=32):
        print(f"⏳ Generating embeddings for {len(text_list)} documents...")
        embeddings = []
        
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i : i + batch_size]
            batch_emb = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_emb)
            
        return np.vstack(embeddings)

    def encode_query(self, query):
        return self.model.encode([query], convert_to_numpy=True)