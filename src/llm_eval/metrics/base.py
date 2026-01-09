from abc import ABC, abstractmethod
from typing import Dict, Any, List

class Metric(ABC):
    @abstractmethod
    def compute(self, example: Dict[str, Any], model_output: Dict[str, Any]) -> float:
        """
        Compute the metric score for a single example.
        
        Args:
            example: A dictionary containing 'query', 'expected_answer', 'retrieved_contexts'
            model_output: A dictionary containing 'generated_answer'
            
        Returns:
            A float score.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
