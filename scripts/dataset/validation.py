"""
validation.py

Script CLI để kiểm tra chất lượng dataset (validate).
Pipeline tự động lưu validation_report.json vào thư mục meta/.

Cách dùng:
    python scripts/dataset/validation.py --input <INPUT_CSV>

Tuỳ chỉnh validation:
    Sửa trực tiếp trong hàm main() để override config.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.dataset_config import default_dataset_config
from config.validation_config import default_validation_config
from src.dataset.loader import read_csv_with_columns
from src.pipeline.validation_pipeline import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng dataset.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn file CSV đầu vào.",
    )
    return parser.parse_args()


def _parse_dataset_info(input_path: Path) -> tuple[str, str]:
    """Parse dataset_name và version từ input path.
    Ví dụ: datasets/custom_dataset/v1/processed/processed_dataset.csv -> ("custom_dataset", "v1")
    """
    input_str = str(input_path).replace("\\", "/")
    parts = input_str.split("/")
    try:
        datasets_idx = parts.index("datasets")
        return parts[datasets_idx + 1], parts[datasets_idx + 2]
    except (ValueError, IndexError):
        return "custom_dataset", "v1"


def main() -> None:
    args = parse_args()

    # ==========================================================
    # [TUỲ CHỈNH] Cấu hình validation
    # ==========================================================
    dataset_config = default_dataset_config.override(
        # Ví dụ override:
        # comment_col="comment",
        # label_col="is_toxic",
    )
    validation_config = default_validation_config.override(
        # Ví dụ override:
        # required_cols=["comment", "is_toxic"],
        # enable_column_check=True,
        # enable_null_check=True,
        # enable_empty_or_no_letter_check=True,
        # enable_duplicate_check=True,
    )
    # ==========================================================

    input_path = Path(args.input)
    dataset_name, version = _parse_dataset_info(input_path)

    # Load dataset
    df = read_csv_with_columns(
        str(input_path),
        comment_col=dataset_config.comment_col,
        label_col=dataset_config.label_col,
        encoding=dataset_config.encoding,
    )

    # Validation pipeline (từ src/pipeline) — tự động lưu meta/
    report = validate_dataset(
        df,
        validation_config=validation_config,
        dataset_config=dataset_config,
        dataset_name=dataset_name,
        version=version,
    )

    # Console summary
    print("\n========== VALIDATION SUMMARY ==========")

    if "duplicate_check" in report:
        dup_info = report["duplicate_check"]
        print(f"Duplicated rows: {dup_info['total_duplicated_rows']}")
        print(f"Mixed-label duplicates: {dup_info['mixed_label_comments_count']}")

    if "null_check" in report:
        print(f"Null ratio: {report['null_check']['null_ratio']:.4f}")

    if "empty_or_no_letter_check" in report:
        print(
            f"Empty comments: {report['empty_or_no_letter_check'].get('empty_count', 0)}"
        )

    if "_meta_saved_to" in report:
        print(f"\n[INFO] Report saved: {report['_meta_saved_to']}")


if __name__ == "__main__":
    main()
