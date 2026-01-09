import pandas as pd
import json
from typing import List, Dict, Any
from pathlib import Path

class DatasetLoader:
    REQUIRED_FIELDS = ['query', 'expected_answer', 'retrieved_contexts']

    @staticmethod
    def load(path: str) -> List[Dict[str, Any]]:
        """
        Load dataset from a file (JSONL or CSV).
        Checks for required fields.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if path.endswith('.jsonl'):
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            df = pd.DataFrame(data)
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            raise ValueError("Unsupported format. Use .jsonl or .csv")

        # Validation
        missing = [field for field in DatasetLoader.REQUIRED_FIELDS if field not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required fields: {missing}")

        return df.to_dict('records')

class ModelOutputLoader:
    @staticmethod
    def load(path: str) -> List[Dict[str, Any]]:
        """
        Load model output from a JSONL file. 
        Expected to have 'query' and 'generated_answer' roughly aligned or ordered.
        For simplicity, we assume we return a list of dicts.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model output file not found: {path}")
            
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
