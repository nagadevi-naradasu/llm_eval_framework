import typer
from typing import Optional
from pathlib import Path
import json
import os
import pandas as pd
import logging
from ..core.config import Config
from ..core.loader import DatasetLoader, ModelOutputLoader
from ..metrics.nlp_metrics import BleuMetric, RougeMetric, BertScoreMetric
from ..metrics.rag_metrics import FaithfulnessMetric, ContextRelevancyMetric, AnswerRelevancyMetric
from ..metrics.judge import LLMJudgeMetric
from ..reporting.visualizer import Visualizer
from ..reporting.markdown import MarkdownReportGenerator

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ..core.plugins import load_custom_metric

def instantiate_metrics(config: Config):
    metrics = []
    # Standard metrics
    for m_name in config.metrics:
        if m_name == "bleu":
            metrics.append(BleuMetric())
        elif m_name == "rouge-l":
            metrics.append(RougeMetric())
        elif m_name == "bert-score":
            metrics.append(BertScoreMetric())
        elif m_name == "faithfulness":
            metrics.append(FaithfulnessMetric())
        elif m_name == "context-relevancy":
            metrics.append(ContextRelevancyMetric())
        elif m_name == "answer-relevancy":
            metrics.append(AnswerRelevancyMetric())
            
    # Judge metrics
    if config.judge_config:
        for dim in config.judge_config.dimensions:
            metrics.append(LLMJudgeMetric(
                provider=config.judge_config.provider,
                model=config.judge_config.model,
                dimension=dim,
                api_key_env_var=config.judge_config.api_key_env_var
            ))

    # Custom Plugin metrics
    for custom_cfg in config.custom_metrics:
        try:
            metric = load_custom_metric(custom_cfg.class_path, custom_cfg.arguments)
            metrics.append(metric)
            logger.info(f"Loaded custom metric: {custom_cfg.name}")
        except Exception as e:
            logger.error(f"Failed to load custom metric {custom_cfg.name}: {e}")

    return metrics

@app.command()
def version():
    """Print the version of the llm-eval tool."""
    typer.echo("llm-eval v0.1.0")


@app.command()
def run(config_path: Path):
    """
    Run the evaluation pipeline based on the provided configuration file.
    """
    logger.info(f"Loading config from {config_path}")
    try:
        config = Config.load(str(config_path))
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise typer.Exit(code=1)

    logger.info("Loading dataset...")
    try:
        dataset = DatasetLoader.load(config.dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise typer.Exit(code=1)

    metrics = instantiate_metrics(config)
    logger.info(f"Initialized {len(metrics)} metrics: {[m.name for m in metrics]}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    visualizer = Visualizer(config.output_dir)

    for model_cfg in config.models:
        logger.info(f"Evaluating model: {model_cfg.name}")
        try:
            outputs = ModelOutputLoader.load(model_cfg.output_path)
        except Exception as e:
            logger.error(f"Failed to load output for {model_cfg.name}: {e}")
            continue

        if len(outputs) != len(dataset):
            logger.warning(f"Mismatch in counts: Dataset ({len(dataset)}) vs Outputs ({len(outputs)}). Using intersection based on order.")
            # In production we'd do strict ID matching

        results = []
        for i, example in enumerate(dataset):
            if i >= len(outputs): break
            out = outputs[i]
            
            row = {"query": example.get('query'), "id": i}
            for metric in metrics:
                try:
                    score = metric.compute(example, out)
                    row[metric.name] = score
                except Exception as e:
                    logger.error(f"Metric {metric.name} failed for example {i}: {e}")
                    row[metric.name] = None
            results.append(row)

        df = pd.DataFrame(results)
        
        # Save detailed results
        json_path = Path(config.output_dir) / f"{model_cfg.name}_results.json"
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved results to {json_path}")
        
        # Calculate summary
        summary = df.mean(numeric_only=True).to_dict()
        with open(Path(config.output_dir) / f"{model_cfg.name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Visualization
        visualizer.generate_histograms(df)
        visualizer.generate_radar_chart(summary, model_cfg.name)

        # Markdown Report
        logger.info("Generating Markdown report...")
        md_generator = MarkdownReportGenerator(config.output_dir)
        report_path = md_generator.generate(model_cfg.name, df, summary)
        logger.info(f"Report saved to {report_path}")

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    app()
