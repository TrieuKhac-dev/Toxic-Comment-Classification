"""
mlflow_tracking_config.py

Cấu hình MLflow tracking cho dataset pipeline.
Quyết định MLflow sẽ track những gì và track như thế nào.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.base_config import BaseConfig


@dataclass
class MLflowTrackingConfig(BaseConfig["MLflowTrackingConfig"]):
    """
    Cấu hình MLflow tracking cho dataset pipeline.

    Usage:
        cfg = MLflowTrackingConfig()
        if cfg.enabled:
            with MLflowTracker(experiment_name=cfg.experiment_name, ...) as tracker:
                ...
    """

    # Bật/tắt MLflow tracking (có thể override bằng --mlflow-track/--no-mlflow-track)
    enabled: bool = True

    # Tên experiment trên MLflow
    experiment_name: str = "dataset_pipeline"

    # Tự động track các file trong thư mục meta/ của dataset
    track_meta_files: bool = True

    # Tags mặc định gắn với mỗi run
    default_tags: dict[str, str] = field(
        default_factory=lambda: {
            "pipeline_type": "dataset",
            "tool": "dvc_pipeline",
        }
    )
