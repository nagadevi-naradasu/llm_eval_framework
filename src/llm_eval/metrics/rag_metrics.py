from typing import Dict, Any, List
import os
import time
import functools
import openai
from .base import Metric
import logging

logger = logging.getLogger(__name__)

# Very basic LLM client wrapper for RAG metrics which usually depend on LLM evaluation
class RagMetricBase(Metric):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None # Client will be None if no key provided

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

class FaithfulnessMetric(RagMetricBase):
    @property
    def name(self) -> str:
        return "faithfulness"

    def compute(self, example: Dict[str, Any], model_output: Dict[str, Any]) -> float:
        contexts = example.get('retrieved_contexts', [])
        answer = model_output.get('generated_answer', '')
        if not contexts or not answer:
            return 0.0
        
        context_text = "\n".join(contexts)
        prompt = f"""
        Given the following context and answer, determine if the answer is faithful to the context.
        Faithful means the answer contains only information found in the context.
        
        Context: {context_text}
        
        Answer: {answer}
        
        Return exactly '1' if faithful, '0' if not.
        """
        result = self._call_llm(prompt)
        return 1.0 if '1' in result else 0.0

class ContextRelevancyMetric(RagMetricBase):
    @property
    def name(self) -> str:
        return "context-relevancy"

    def compute(self, example: Dict[str, Any], model_output: Dict[str, Any]) -> float:
        contexts = example.get('retrieved_contexts', [])
        query = example.get('query', '')
        if not contexts or not query:
            return 0.0
        
        context_text = "\n".join(contexts)
        prompt = f"""
        Given the following query and context, determine if the context is relevant to the query.
        
        Query: {query}
        
        Context: {context_text}
        
        Return a score between 0.0 and 1.0, where 1.0 is highly relevant. Return only the number.
        """
        result = self._call_llm(prompt)
        try:
            return float(result)
        except ValueError:
            return 0.5 # Default fallback

class AnswerRelevancyMetric(RagMetricBase):
    @property
    def name(self) -> str:
        return "answer-relevancy"

    def compute(self, example: Dict[str, Any], model_output: Dict[str, Any]) -> float:
        query = example.get('query', '')
        answer = model_output.get('generated_answer', '')
        
        if not query or not answer:
            return 0.0
            
        prompt = f"""
        Given the following query and answer, determine if the answer is relevant to the query.
        
        Query: {query}
        
        Answer: {answer}
        
        Return a score between 0.0 and 1.0, where 1.0 is highly relevant. Return only the number.
        """
        result = self._call_llm(prompt)
        try:
            return float(result)
        except ValueError:
            return 0.5
