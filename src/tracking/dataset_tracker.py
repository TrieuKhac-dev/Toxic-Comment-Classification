"""
dataset_tracker.py

Module riêng để track dataset meta vào MLflow.
KHÔNG phụ thuộc vào src/pipeline/ - tracking là việc riêng, không phải của pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.tracking.mlflow_tracker import MLflowTracker


def get_dataset_hash(dataset_path: str) -> str | None:
    """
    Tính hash (md5) của thư mục dataset dựa trên nội dung các file bên trong.
    Trả về 8 ký tự đầu của hash.

    Parameters
    ----------
    dataset_path : str
        Đường dẫn đến thư mục version (VD: 'datasets/custom_dataset/v1').

    Returns
    -------
    str | None
        8 ký tự đầu của hash, hoặc None nếu thư mục không tồn tại.
    """
    from config.path_config import default_path_config

    full_path = Path(default_path_config.project_root) / dataset_path
    if not full_path.exists() or not full_path.is_dir():
        return None

    hash_md5 = hashlib.md5()
    # Duyệt tất cả file trong thư mục (bao gồm cả thư mục con)
    for f in sorted(full_path.rglob("*")):
        if f.is_file():
            # Thêm relative path vào hash
            rel_path = str(f.relative_to(full_path))
            hash_md5.update(rel_path.encode("utf-8"))
            # Thêm nội dung file vào hash
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(4096), b""):
                    hash_md5.update(chunk)

    return hash_md5.hexdigest()[:8]


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Làm phẳng dict lồng nhau thành key.path."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _extract_metrics(data: dict[str, Any]) -> dict[str, float]:
    """Trích xuất các giá trị số từ dict để log làm metrics."""
    flat = _flatten_dict(data)
    metrics = {}
    for k, v in flat.items():
        if isinstance(v, int | float) and not isinstance(v, bool):
            metrics[k] = float(v)
    return metrics


def _extract_params(data: dict[str, Any]) -> dict[str, str]:
    """Trích xuất các giá trị không phải số từ dict để log làm params."""
    flat = _flatten_dict(data)
    params = {}
    for k, v in flat.items():
        if not isinstance(v, int | float) or isinstance(v, bool):
            params[k] = str(v)
    return params


def track_dataset_meta(tracker: MLflowTracker, meta_dir: str) -> None:
    """
    Đọc tất cả file trong thư mục meta/ và log vào MLflow.

    Parameters
    ----------
    tracker : MLflowTracker
        Instance MLflowTracker đang active.
    meta_dir : str
        Đường dẫn đến thư mục meta (VD: 'datasets/custom_dataset/v1/meta').
    """
    meta_path = Path(meta_dir)
    if not meta_path.exists():
        print(f"  ⚠️  Không tìm thấy thư mục meta: {meta_dir}")
        return

    for f in sorted(meta_path.iterdir()):
        if not f.is_file():
            continue

        print(f"  📄 Tracking: {f.name}")
        data: dict[str, Any] | None = None

        if f.suffix in (".yaml", ".yml"):
            with open(f, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        elif f.suffix == ".json":
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)

        if not data or not isinstance(data, dict):
            # Log file gốc nếu không parse được
            tracker.log_artifact(str(f))
            continue

        # Log metrics (các giá trị số)
        metrics = _extract_metrics(data)
        if metrics:
            # Thêm prefix để phân biệt metrics từ các file khác nhau
            prefixed_metrics: dict[str, float] = {
                f"{f.stem}.{k}": v for k, v in metrics.items()
            }
            tracker.log_metrics(prefixed_metrics)

        # Log params (các giá trị không phải số)
        params = _extract_params(data)
        if params:
            prefixed_params: dict[str, str] = {
                f"{f.stem}.{k}": v for k, v in params.items()
            }
            tracker.log_params(prefixed_params)

        # Log file gốc làm artifact
        tracker.log_artifact(str(f))


def mlflow_run_exists(
    experiment_name: str,
    dataset_hash: str,
    tracking_uri: str | None = None,
) -> bool:
    """
    Kiểm tra xem đã có MLflow run nào với dataset hash này chưa.

    Parameters
    ----------
    experiment_name : str
        Tên experiment trên MLflow.
    dataset_hash : str
        Dataset hash cần kiểm tra.
    tracking_uri : str | None
        URI của MLflow tracking server.

    Returns
    -------
    bool
        True nếu đã có run, False nếu chưa.
    """
    import os

    import mlflow

    from config.path_config import default_path_config

    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    uri: str = (
        tracking_uri
        or env_uri
        or str(Path(default_path_config.project_root) / "mlruns")
    )
    mlflow.set_tracking_uri(uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return False

    runs: pd.DataFrame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.dataset_hash = '{dataset_hash}'",
        max_results=1,
    )
    return not runs.empty
