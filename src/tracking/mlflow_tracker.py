"""
mlflow_tracker.py

Wrapper cho MLflow tracking.
Cung cấp context manager để tự động log params, metrics, artifacts.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml

from config.path_config import default_path_config


class MLflowTracker:
    """
    Context manager cho MLflow experiment tracking.

    Usage:
        with MLflowTracker(experiment_name="toxic_comment_clf", run_name="lr_baseline") as tracker:
            tracker.log_params({"model": "logistic_regression", "C": 1.0})
            tracker.log_metrics({"accuracy": 0.85, "f1": 0.82})
            tracker.log_artifact("path/to/file.csv")
            # Hoặc log artifact từ dict
            tracker.log_dict_as_yaml({"key": "value"}, "report.yaml")
    """

    def __init__(
        self,
        experiment_name: str = "toxic_comment_classification",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """
        Parameters
        ----------
        experiment_name : str
            Tên experiment trên MLflow.
        run_name : str | None
            Tên run (mặc định: tự sinh).
        tracking_uri : str | None
            URI của MLflow tracking server.
            Mặc định: lấy từ env MLFLOW_TRACKING_URI hoặc 'mlruns' (local).
        tags : dict[str, str] | None
            Tags gắn với run.
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.tracking_uri: str = (
            tracking_uri
            or env_uri
            or str(Path(default_path_config.project_root) / "mlruns")
        )
        self.tags = tags or {}
        self._active_run: mlflow.ActiveRun | None = None

    def __enter__(self) -> MLflowTracker:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        self._active_run = mlflow.start_run(run_name=self.run_name)

        if self.tags:
            mlflow.set_tags(self.tags)

        return self

    def __exit__(self, *args: Any) -> None:
        mlflow.end_run()
        self._active_run = None

    @property
    def run_id(self) -> str:
        """ID của run hiện tại."""
        if self._active_run is None:
            raise RuntimeError("No active MLflow run")
        run_id: str = self._active_run.info.run_id
        return run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log tham số."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        """Log một file artifact."""
        mlflow.log_artifact(local_path)

    def log_dict_as_yaml(self, data: dict[str, Any], filename: str) -> None:
        """Log dict dưới dạng file YAML artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            mlflow.log_artifact(str(tmp_path))

    def log_dict_as_json(self, data: dict[str, Any], filename: str) -> None:
        """Log dict dưới dạng file JSON artifact."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(str(tmp_path))

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """
        Log model lên MLflow.
        Tự động chọn phương thức phù hợp dựa trên loại model:
        - sklearn: dùng mlflow.sklearn.log_model
        - Khác: dùng mlflow.pyfunc.log_model
        """
        try:
            import sklearn.base

            if isinstance(model, sklearn.base.BaseEstimator):
                mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)
                return
        except ImportError:
            pass

        # Fallback: dùng pyfunc
        mlflow.pyfunc.log_model(artifact_path=artifact_path, python_model=model)

    def set_tag(self, key: str, value: str) -> None:
        """Set tag cho run."""
        mlflow.set_tag(key, value)
