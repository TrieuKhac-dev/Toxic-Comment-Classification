"""
reproduce_experiment.py

Script để reproduce một experiment từ MLflow.
Lấy params từ MLflow run, chạy lại pipeline, so sánh kết quả.

Usage:
    python scripts/reproduce_experiment.py --run-id <run_id>
    python scripts/reproduce_experiment.py --experiment toxic_comment_classification --best f1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import mlflow
import pandas as pd

from config.dataset_config import default_dataset_config
from config.path_config import default_path_config
from src.pipeline.cleaning_pipeline import clean_text_pipeline
from src.pipeline.preprocessing_pipeline import preprocess_text_pipeline
from src.tracking.experiment_manager import ExperimentManager

project_root = Path(__file__).resolve().parent.parent.parent


def reproduce_run(run_id: str) -> None:
    """Reproduce một MLflow run."""
    run = mlflow.get_run(run_id)
    if run is None:
        print(f"Run {run_id} not found!")
        return

    params = run.data.params
    print(f"Reproducing run: {run_id}")
    print(f"Params: {params}")

    # Load data
    dataset_name = params.get("dataset_name", "custom_dataset")
    version = params.get("dataset_version", "v1")
    raw_path = (
        Path(project_root)
        / default_path_config.get_raw_dir(dataset_name, version)
        / "raw_dataset.csv"
    )
    df = pd.read_csv(raw_path)

    # Run pipeline
    df_cleaned, _ = clean_text_pipeline(
        df,
        dataset_name=dataset_name,
        version=version,
        save_meta=True,
    )

    # Preprocess
    df_cleaned["processed_comment"] = df_cleaned[
        default_dataset_config.comment_col
    ].apply(
        lambda x: preprocess_text_pipeline(
            str(x),
            dataset_name=dataset_name,
            version=version,
            save_meta=True,
        )
    )

    print(f"Reproduce complete. Cleaned shape: {df_cleaned.shape}")


def reproduce_best(experiment_name: str, metric: str = "f1") -> None:
    """Reproduce run tốt nhất dựa trên metric."""
    manager = ExperimentManager(experiment_name=experiment_name)
    best = manager.get_best_run(metric=metric)
    if best is None:
        print(f"No runs found for experiment '{experiment_name}'")
        return

    run_id = best.get("run_id")
    if run_id:
        reproduce_run(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce MLflow experiments")
    parser.add_argument("--run-id", type=str, help="MLflow run ID to reproduce")
    parser.add_argument(
        "--experiment",
        type=str,
        default="toxic_comment_classification",
        help="Experiment name",
    )
    parser.add_argument(
        "--best", type=str, help="Reproduce best run by metric (e.g., f1)"
    )

    args = parser.parse_args()

    if args.run_id:
        reproduce_run(args.run_id)
    elif args.best:
        reproduce_best(args.experiment, args.best)
    else:
        print("Please provide --run-id or --best")
        parser.print_help()
