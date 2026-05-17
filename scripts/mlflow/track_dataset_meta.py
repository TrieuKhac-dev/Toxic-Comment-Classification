"""
track_dataset_meta.py

Script MLflow tracking riêng cho dataset pipeline.
Đọc các file trong thư mục meta/ của dataset và log vào MLflow.

Nguyên tắc:
- 1 dataset hash = 1 MLflow run (nếu hash không đổi thì không tạo run mới)
- Pipeline script (validation.py, cleaning.py) KHÔNG chứa MLflow code
- Script này là nơi duy nhất xử lý MLflow tracking cho dataset

Usage:
    python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1
    python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1 --no-mlflow-track
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.mlflow_tracking_config import MLflowTrackingConfig
from config.path_config import default_path_config
from src.tracking.dataset_tracker import (
    get_dataset_hash,
    mlflow_run_exists,
    track_dataset_meta,
)
from src.tracking.mlflow_tracker import MLflowTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track dataset meta vào MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1
  python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1 --no-mlflow-track
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Tên dataset (VD: custom_dataset)",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version của dataset (VD: v1)",
    )
    parser.add_argument(
        "--mlflow-track",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override MLflow tracking enabled từ config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Đọc config
    mlflow_cfg = MLflowTrackingConfig()

    # Override bằng argument nếu có
    if args.mlflow_track is not None:
        mlflow_cfg = mlflow_cfg.override(enabled=args.mlflow_track)

    if not mlflow_cfg.enabled:
        print("🔇 MLflow tracking disabled, skip")
        return

    dataset_path = f"datasets/{args.dataset}/{args.version}"
    meta_dir = f"{dataset_path}/meta"
    project_root = Path(default_path_config.project_root)

    # Kiểm tra thư mục meta có tồn tại không
    if not (project_root / meta_dir).exists():
        print(f"  ⚠️  Không tìm thấy thư mục meta: {meta_dir}")
        print("  Hãy chạy pipeline trước để sinh dữ liệu meta.")
        return

    # Lấy dataset hash
    dataset_hash = get_dataset_hash(dataset_path)
    if not dataset_hash:
        print(f"  ⚠️  Không thể tính hash cho {dataset_path}")
        return

    print(f"🔍 Dataset hash: {dataset_hash}")

    # Kiểm tra xem hash này đã được track chưa
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        (project_root / "mlruns").as_uri(),
    )

    if mlflow_run_exists(
        experiment_name=mlflow_cfg.experiment_name,
        dataset_hash=dataset_hash,
        tracking_uri=tracking_uri,
    ):
        print(
            f"  Dataset hash {dataset_hash} đã được track, bỏ qua (không tạo run mới)"
        )
        return

    # Tạo run name = dataset_version_hash
    run_name = f"{args.dataset}_{args.version}_{dataset_hash}"

    # Tạo tags
    tags = dict(mlflow_cfg.default_tags)
    tags.update(
        {
            "dataset_name": args.dataset,
            "dataset_version": args.version,
            "dataset_hash": dataset_hash,
        }
    )

    print(f"📊 Creating MLflow run: {run_name}")
    print(f"   Experiment: {mlflow_cfg.experiment_name}")

    with MLflowTracker(
        experiment_name=mlflow_cfg.experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags=tags,
    ) as tracker:
        # Log dataset hash làm param
        tracker.log_params({"dataset_hash": dataset_hash})

        # Track tất cả file trong meta/
        if mlflow_cfg.track_meta_files:
            track_dataset_meta(tracker, str(project_root / meta_dir))

    print(f"✅ Done! Run name: {run_name}")


if __name__ == "__main__":
    main()
