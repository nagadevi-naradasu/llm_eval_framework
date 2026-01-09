from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import yaml
import json
import os

class MetricName(str, Enum):
    BLEU = "bleu"
    ROUGE_L = "rouge-l"
    BERT_SCORE = "bert-score"
    FAITHFULNESS = "faithfulness"
    CONTEXT_RELEVANCY = "context-relevancy"
    ANSWER_RELEVANCY = "answer-relevancy"
    LLM_JUDGE = "llm-judge"

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LLMJudgeConfig(BaseModel):
    provider: Provider = Provider.OPENAI
    model: str = "gpt-4"
    dimensions: List[str] = ["coherence", "relevance", "safety"]
    temperature: float = 0.0
    api_key_env_var: str = "OPENAI_API_KEY"

class CustomMetricConfig(BaseModel):
    name: str
    class_path: str # e.g. "my_module.MyMetric"
    arguments: Dict[str, Any] = {}

class ModelConfig(BaseModel):
    name: str
    output_path: str
    
    @validator('output_path')
    def path_must_exist(cls, v):
        return v

class Config(BaseModel):
    dataset_path: str
    output_dir: str = "results"
    models: List[ModelConfig]
    metrics: List[Union[MetricName, str]] = [] # Allow strings for flexibility
    custom_metrics: List[CustomMetricConfig] = [] # Dedicated section for plugins
    judge_config: Optional[LLMJudgeConfig] = None
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            elif path.endswith('.json'):
                data = json.load(f)
            else:
                raise ValueError("Config file must be .yaml, .yml or .json")
        
        return cls(**data)
