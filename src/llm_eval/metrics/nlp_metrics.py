from typing import Dict, Any, List
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from .base import Metric
import logging

# Ensure nltk data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        # Prevent crash on import if download fails (will fail later if used)
        pass

logger = logging.getLogger(__name__)

class BleuMetric(Metric):
    def __init__(self, weights=(0.25, 0.25, 0.25, 0.25)):
        self.weights = weights
        self.smoother = SmoothingFunction()

    @property
    def name(self) -> str:
        return "bleu"

    def compute(self, example: Dict[str, Any], model_output: Dict[str, Any]) -> float:
        reference = example.get('expected_answer', '')
        candidate = model_output.get('generated_answer', '')
        
        # Simple tokenization
        ref_tokens = nltk.word_tokenize(reference)
        cand_tokens = nltk.word_tokenize(candidate)
        
        if not ref_tokens:
            return 0.0
            
        return sentence_bleu([ref_tokens], cand_tokens, weights=self.weights, smoothing_function=self.smoother.method1)

class RougeMetric(Metric):
    def __init__(self, rouge_type='rougeL'):
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)

    @property
    def name(self) -> str:
        return "rouge-l"

    def compute(self, example: Dict[str, Any], model_output: Dict[str, Any]) -> float:
        reference = example.get('expected_answer', '')
        candidate = model_output.get('generated_answer', '')
        
        if not reference:
            return 0.0
            
        scores = self.scorer.score(reference, candidate)
        return scores[self.rouge_type].fmeasure

class BertScoreMetric(Metric):
    def __init__(self, check_references: bool = True):
        # We initialize lazily or just call the function directly as the library handles caching model
        self.check_references = check_references

    @property
    def name(self) -> str:
        return "bert-score"

    def compute(self, example: Dict[str, Any], model_output: Dict[str, Any]) -> float:
        reference = example.get('expected_answer', '')
        candidate = model_output.get('generated_answer', '')
        
        if not reference or not candidate:
            return 0.0

        try:
            # bert_score expects lists
            P, R, F1 = bert_score([candidate], [reference], lang='en', verbose=False)
            return F1.item()
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            return 0.0
