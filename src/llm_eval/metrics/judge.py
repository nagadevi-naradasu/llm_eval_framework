from typing import Dict, Any
import os
import time
import logging
from .base import Metric
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class LLMJudgeMetric(Metric):
    def __init__(self, provider: str, model: str, dimension: str, api_key_env_var: str = "OPENAI_API_KEY", retries: int = 3):
        self.provider = provider
        self.model_name = model
        self.dimension = dimension
        self.retries = retries
        
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            logger.warning(f"API key for {provider} not found in environment: {api_key_env_var}")

        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @property
    def name(self) -> str:
        return f"judge-{self.dimension}"

    def compute(self, example: Dict[str, Any], model_output: Dict[str, Any]) -> float:
        query = example.get('query', '')
        expected = example.get('expected_answer', '')
        generated = model_output.get('generated_answer', '')
        
        prompt = self._build_prompt(query, expected, generated)
        
        for attempt in range(self.retries):
            try:
                if self.provider == "openai":
                    score = self._call_openai(prompt)
                else:
                    score = self._call_anthropic(prompt)
                return score
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed for {self.name}: {e}")
                time.sleep(2 ** attempt) # Exponential backoff
        
        logger.error(f"Failed to compute {self.name} after {self.retries} retries.")
        return 0.0

    def _build_prompt(self, query, expected, generated) -> str:
        return f"""
        You are an impartial judge. Evaluate the following AI generated response based on the '{self.dimension}' dimension.
        
        Query: {query}
        Expected Answer (Reference): {expected}
        Generated Answer: {generated}
        
        Scoring Rubric for {self.dimension}:
        - 1: Poor
        - 2: Fair
        - 3: Good
        - 4: Very Good
        - 5: Excellent
        
        Output strictly only the score as a single integer (1-5).
        """

    def _call_openai(self, prompt: str) -> float:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        try:
            return float(content) / 5.0 # Normalize to 0-1
        except ValueError:
            return 0.0

    def _call_anthropic(self, prompt: str) -> float:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=10,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = response.content[0].text.strip()
        try:
            return float(content) / 5.0
        except ValueError:
            return 0.0
