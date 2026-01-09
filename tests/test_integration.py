import pytest
import os
import shutil
import json
import logging
from typer.testing import CliRunner
from llm_eval.cli.main import app
from pathlib import Path

runner = CliRunner()

@pytest.fixture(scope="module")
def setup_integration_env():
    # Setup paths
    base_dir = Path("./tests/integration_data")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    # Create a dummy config
    config_path = base_dir / "config.yaml"
    dataset_path = base_dir / "dataset.jsonl"
    output_path = base_dir / "model_out.jsonl"
    results_dir = base_dir / "results"
    
    # Dummy Dataset
    with open(dataset_path, 'w') as f:
        f.write(json.dumps({"query": "q1", "expected_answer": "a1", "retrieved_contexts": ["c1"]}) + "\n")
        f.write(json.dumps({"query": "q2", "expected_answer": "a2", "retrieved_contexts": ["c2"]}) + "\n")

    # Dummy Output
    with open(output_path, 'w') as f:
        f.write(json.dumps({"id": 0, "generated_answer": "a1"}) + "\n")
        f.write(json.dumps({"id": 1, "generated_answer": "wrong"}) + "\n")

    # Config content
    config_content = f"""
dataset_path: "{str(dataset_path).replace(os.sep, '/')}"
output_dir: "{str(results_dir).replace(os.sep, '/')}"

models:
  - name: "test_model"
    output_path: "{str(output_path).replace(os.sep, '/')}"

metrics:
  - "bleu"
  - "rouge-l"
"""
    with open(config_path, 'w') as f:
        f.write(config_content)
        
    yield config_path, results_dir
    
    # Cleanup
    if base_dir.exists():
        shutil.rmtree(base_dir)

def test_CLI_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "llm-eval v0.1.0" in result.stdout

def test_end_to_end_pipeline(setup_integration_env):
    """
    Runs the full CLI command using a real (temp) config and verifies
    that all output files (JSON results, Summary, Markdown report, plots) are created.
    """
    config_path, results_dir = setup_integration_env
    
    # Run CLI
    result = runner.invoke(app, ["run", str(config_path)])
    
    print(result.stdout)
    assert result.exit_code == 0
    
    # Check for expected artifacts
    assert (results_dir / "test_model_results.json").exists()
    assert (results_dir / "test_model_summary.json").exists()
    assert (results_dir / "test_model_report.md").exists()
    
    # Check Plots
    # Note: Column names might have spaces/slashes replaced
    # We expect 'bleu', 'rouge-l' histograms
    assert (results_dir / "hist_bleu.png").exists()
    assert (results_dir / "hist_rouge-l.png").exists()
    assert (results_dir / "radar_test_model.png").exists()
    
    # Verify Content
    with open(results_dir / "test_model_summary.json") as f:
        summary = json.load(f)
        assert "bleu" in summary
        assert "rouge-l" in summary
