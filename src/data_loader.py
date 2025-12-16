import pandas as pd
import os
from .config import DATA_PATH, DATA_DIR

class DataLoader:
    def __init__(self, filepath=None):
        # Default to the PubMed path defined in config
        self.filepath = filepath if filepath else DATA_PATH

    def load_and_process(self):
        if not os.path.exists(self.filepath):
            print(f"❌ Error: File not found at {self.filepath}")
            print(f"📂 Contents of {DATA_DIR}:")
            print(os.listdir(DATA_DIR))
            raise FileNotFoundError(f"Please put 'ori_pqal.json' in the {DATA_DIR} folder.")

        print(f"📂 Loading data from {self.filepath}...")
        
        # Load JSON and Transpose (Specific for PubMedQA format)
        # The PubMedQA dataset is a dictionary of dictionaries, so we must transpose (.T)
        df = pd.read_json(self.filepath).T
        
        # Clean Data
        if 'YEAR' in df.columns:
            df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce').fillna(0).astype(int)
        
        # Create consolidated text column
        # We combine Question and Contexts for the search engine
        df["text"] = df["QUESTION"] + " " + df["CONTEXTS"].astype(str)
        
        print(f"✅ Loaded {len(df)} records.")
        return df