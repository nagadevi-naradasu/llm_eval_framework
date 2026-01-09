# LLM-Eval Framework

A comprehensive, production-ready Python framework for systematically evaluating Large Language Model (LLM) applications. This tool provides a modular suite of metrics including classical NLP scores, RAG-specific assessments, and LLM-as-a-Judge capabilities.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

## Features

- **Multi-Modal Metrics**:
  - **Classical**: BLEU, ROUGE-L, BERTScore
  - **RAG-Specific**: Faithfulness, Context Relevancy, Answer Relevancy
  - **LLM-as-a-Judge**: Multi-dimensional evaluation (Coherence, Safety, Relevance) using GPT-4 or Claude.
- **CLI-Driven**: Easy-to-use command line interface for running evaluations.
- **Configurable**: YAML/JSON based configuration system.
- **Visualizations**: Automatic generation of Histograms and Radar charts to visualize performance.
- **CI/CD Ready**: Dockerized and designed for integration with GitHub Actions/GitLab CI.

## Installation

### Using Poetry (Recommended)

```bash
git clone https://github.com/yourusername/llm-eval.git
cd llm-eval
poetry install
```

### Using Pip

```bash
pip install .
```

### Protocol for Docker

```bash
docker-compose up --build
```

## Quick Start

1.  **Configure Environment**:
    Copy `.env.example` to `.env` and add your API keys.
    ```bash
    cp .env.example .env
    # Edit .env with your OPENAI_API_KEY
    ```

2.  **Prepare Data**:
    Ensure you have your benchmark dataset and model outputs. See `benchmarks/rag_benchmark.jsonl` for the expected format.

3.  **Run Evaluation**:
    ```bash
    llm-eval run examples/config.yaml
    ```

## Configuration

Control the evaluation pipeline using a YAML config file:

```yaml
dataset_path: "benchmarks/rag_benchmark.jsonl"
output_dir: "results"

models:
  - name: "my-model-v1"
    output_path: "path/to/predictions.jsonl"

metrics:
  - "bleu"
  - "rouge-l"
  - "faithfulness"
  - "llm-judge"

judge_config:
  provider: "openai"
  model: "gpt-4"
  dimensions: ["coherence", "relevance"]
```

## Architecture

The framework is built on a modular architecture:
- **Core**: Handles configuration, data loading, and orchestration.
- **Metrics**: Abstract base class with implementations for various scoring strategies.
- **Reporting**: Generates visual and text reports.

## Adding Custom Metrics

Inherit from the `Metric` base class in `src/llm_eval/metrics/base.py`:

```python
from llm_eval.metrics.base import Metric

class MyCustomMetric(Metric):
    @property
    def name(self):
        return "my-metric"
        
    def compute(self, example, output):
        # Your logic here
        return 0.5
```

## Project Structure

```
llm-eval-framework/
├── src/llm_eval/          # Core package source code
│   ├── metrics/           # Metric implementations (NLP, RAG, Judge)
│   ├── core/              # Configuration & Plugin logic
│   ├── reporting/         # Visualization & Report generation
│   └── cli/               # CLI entry point
├── benchmarks/            # Standardized benchmark datasets
├── examples/              # Configuration examples & sample outputs
├── tests/                 # Unit & Integration tests
├── results/               # Generated reports & plots (gitignored)
├── Dockerfile             # Container definition
└── docker-compose.yml     # Orchestration config
```

## Advanced Features

### Dynamic Plugin System
The framework supports loading custom metrics at runtime. Simply define your metric class and reference it in `config.yaml`:
```yaml
custom_metrics:
  - name: "my-custom-metric"
    class_path: "my_module.MyMetric"
```

### Robust Error Handling
Integrated exponential backoff ensures that transient API failures (e.g. Rate Limits) do not crash your long-running evaluations.

## Development

Run tests:
```bash
pytest tests/ --cov=llm_eval
```

Format code:
```bash
black src tests
```

