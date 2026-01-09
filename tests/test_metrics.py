import pytest
from llm_eval.metrics.nlp_metrics import BleuMetric, RougeMetric, BertScoreMetric
from llm_eval.metrics.rag_metrics import FaithfulnessMetric, ContextRelevancyMetric
from llm_eval.metrics.judge import LLMJudgeMetric
from unittest.mock import MagicMock, patch

# Fixtures
@pytest.fixture
def example_data():
    return {
        "query": "test query",
        "expected_answer": "The quick brown fox jumps over the lazy dog.",
        "retrieved_contexts": ["The quick brown fox.", "The lazy dog."]
    }

@pytest.fixture
def model_output():
    return {
        "generated_answer": "The fast brown fox jumps over the lazy dog."
    }

# NLP Metrics Tests
def test_bleu_metric(example_data, model_output):
    metric = BleuMetric()
    score = metric.compute(example_data, model_output)
    assert 0.0 <= score <= 1.0

def test_rouge_metric(example_data, model_output):
    metric = RougeMetric()
    score = metric.compute(example_data, model_output)
    assert 0.0 <= score <= 1.0

# Mocking external calls for BERTScore to avoid downloading model in tests
@patch('llm_eval.metrics.nlp_metrics.bert_score')
def test_bert_score_metric(mock_bert_score, example_data, model_output):
    # Mock return value (P, R, F1)
    mock_bert_score.return_value = (MagicMock(), MagicMock(), MagicMock(item=lambda: 0.9))
    
    metric = BertScoreMetric()
    score = metric.compute(example_data, model_output)
    assert score == 0.9

# RAG Metrics Tests (Mocked)
@patch('llm_eval.metrics.rag_metrics.openai.OpenAI')
def test_faithfulness_metric(mock_openai, example_data, model_output):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "1"
    mock_openai.return_value = mock_client
    
    metric = FaithfulnessMetric()
    score = metric.compute(example_data, model_output)
    assert score == 1.0

@patch('llm_eval.metrics.rag_metrics.openai.OpenAI')
def test_context_relevancy_metric(mock_openai, example_data, model_output):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "0.8"
    mock_openai.return_value = mock_client
    
    metric = ContextRelevancyMetric()
    score = metric.compute(example_data, model_output)
    assert score == 0.8

# Judge Tests
@patch('llm_eval.metrics.judge.openai.OpenAI')
def test_judge_metric_openai(mock_openai, example_data, model_output):
    mock_client = MagicMock()
    # Mock returning "5" -> 5/5 = 1.0
    mock_client.chat.completions.create.return_value.choices[0].message.content = "5"
    mock_openai.return_value = mock_client
    
    metric = LLMJudgeMetric(provider="openai", model="gpt-4", dimension="coherence")
    score = metric.compute(example_data, model_output)
    assert score == 1.0
