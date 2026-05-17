"""
validation.py

Script CLI để kiểm tra chất lượng dataset (validate).
Pipeline tự động lưu validation_report.json vào thư mục meta/.

Cách dùng CLI:
    python scripts/dataset/validation.py --input <INPUT_CSV>

Cách dùng trong code Python / Colab:
    from scripts.dataset.validation import validate_dataset_file

    report = validate_dataset_file(
        input_path="datasets/custom_dataset/v1/processed/processed_dataset.csv",
        comment_col="comment",
        label_col="is_toxic",
    )

Tuỳ chỉnh validation:
    Sửa trực tiếp trong hàm main() để override config.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.dataset_config import default_dataset_config
from config.validation_config import default_validation_config
from src.pipeline.validation_pipeline import validate_dataset
from src.utils.csv import read_csv_with_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng dataset.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn file CSV đầu vào.",
    )
    parser.add_argument(
        "--comment-col",
        type=str,
        default=None,
        help="Tên cột comment (mặc định: 'comment').",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Tên cột nhãn (mặc định: 'is_toxic').",
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


def validate_dataset_file(
    input_path: str,
    comment_col: str = "comment",
    label_col: str = "is_toxic",
    encoding: str = "utf-8",
    validation_config_override: dict[str, Any] | None = None,
    dataset_config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Kiểm tra chất lượng dataset và lưu báo cáo.

    Hàm này có thể gọi trực tiếp từ code Python hoặc Colab,
    không phụ thuộc argparse.

    Parameters
    ----------
    input_path : str
        Đường dẫn file CSV đầu vào.
    comment_col : str
        Tên cột comment (mặc định: "comment").
    label_col : str
        Tên cột nhãn (mặc định: "is_toxic").
    encoding : str
        Encoding file CSV (mặc định: "utf-8").
    validation_config_override : dict | None
        Các tham số ghi đè cho ValidationConfig.
    dataset_config_override : dict | None
        Các tham số ghi đè cho DatasetConfig.

    Returns
    -------
    dict[str, Any]
        Báo cáo validation.
    """
    input_path_obj = Path(input_path)
    dataset_name, version = _parse_dataset_info(input_path_obj)

    # Config
    dataset_config = default_dataset_config.override(
        comment_col=comment_col,
        label_col=label_col,
        encoding=encoding,
        **(dataset_config_override or {}),
    )
    validation_config = default_validation_config.override(
        **(validation_config_override or {}),
    )

    # Load dataset
    df = read_csv_with_columns(
        str(input_path_obj),
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

    return report


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
    validation_config = default_validation_config.override(  # noqa: F841
        # Ví dụ override:
        # required_cols=["comment", "is_toxic"],
        # enable_column_check=True,
        # enable_null_check=True,
        # enable_empty_or_no_letter_check=True,
        # enable_duplicate_check=True,
    )
    # ==========================================================

    # CLI args override config
    comment_col = args.comment_col or dataset_config.comment_col
    label_col = args.label_col or dataset_config.label_col

    validate_dataset_file(
        input_path=args.input,
        comment_col=comment_col,
        label_col=label_col,
        encoding=dataset_config.encoding,
    )


if __name__ == "__main__":
    main()
