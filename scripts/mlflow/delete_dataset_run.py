"""
delete_dataset_run.py

Xoá MLflow run của một dataset version cụ thể.
Dùng khi bạn muốn track lại từ đầu cho version đó.

Usage:
    python scripts/mlflow/delete_dataset_run.py --dataset custom_dataset --version v2
    python scripts/mlflow/delete_dataset_run.py --dataset custom_dataset --version v2 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import mlflow

from config.mlflow_tracking_config import MLflowTrackingConfig
from config.path_config import default_path_config
from src.tracking.dataset_tracker import get_dataset_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Xoá MLflow run của một dataset version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/mlflow/delete_dataset_run.py --dataset custom_dataset --version v2
  python scripts/mlflow/delete_dataset_run.py --dataset custom_dataset --version v2 --dry-run
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
        help="Version của dataset (VD: v2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ liệt kê, không xoá",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow_cfg = MLflowTrackingConfig()
    project_root = Path(default_path_config.project_root)

    dataset_path = f"datasets/{args.dataset}/{args.version}"

    # Lấy dataset hash
    dataset_hash = get_dataset_hash(dataset_path)
    if not dataset_hash:
        print(f"  ⚠️  Không thể tính hash cho {dataset_path}")
        return

    run_name = f"{args.dataset}_{args.version}_{dataset_hash}"
    print(f"🔍 Dataset hash: {dataset_hash}")
    print(f"🔍 Run name cần xoá: {run_name}")

    # Set tracking URI (cần file:/// prefix cho đường dẫn local)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = (project_root / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    # Tìm experiment
    experiment = mlflow.get_experiment_by_name(mlflow_cfg.experiment_name)
    if not experiment:
        print(f"  ⚠️  Không tìm thấy experiment: {mlflow_cfg.experiment_name}")
        return

    # Tìm run theo tên
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )

    if runs.empty:
        print(f"  ⚠️  Không tìm thấy run nào có tên '{run_name}'")
        return

    run_id = runs.iloc[0]["run_id"]
    print(f"  📄 Tìm thấy run: {run_id}")

    if args.dry_run:
        print("\n🔍 Dry-run mode: không xoá gì cả.")
        print(
            f"  Chạy lệnh sau để xoá: python scripts/mlflow/delete_dataset_run.py --dataset {args.dataset} --version {args.version}"
        )
        return

    # Xác nhận
    confirm = input(f"\n⚠️  Xoá run '{run_name}' (ID: {run_id})? (y/N): ")
    if confirm.lower() != "y":
        print("❌ Đã huỷ.")
        return

    # Xoá run
    mlflow.delete_run(run_id)
    print(f"  ✅ Đã xoá run: {run_name} (ID: {run_id})")


if __name__ == "__main__":
    main()
