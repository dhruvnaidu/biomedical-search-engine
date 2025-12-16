import os
from pathlib import Path

# Base project directory (biomedical-search-engine/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directory (biomedical-search-engine/data/)
DATA_DIR = BASE_DIR / "data"

# 👉 POINTING TO YOUR SPECIFIC NESTED PATH
# Path: biomedical-search-engine/data/pubmedqa/data/ori_pqal.json
DATA_PATH = DATA_DIR / "ori_pqal.json" 

# Paths for caching processed data
PROCESSED_TEXTS_PATH = DATA_DIR / "processed_texts.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

# Model Configs
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"