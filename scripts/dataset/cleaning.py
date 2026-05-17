"""
cleaning.py

Script CLI để làm sạch dataset (cleaning).
Pipeline tự động lưu cleaning_log.json vào thư mục meta/.

Cách dùng:
    python scripts/dataset/cleaning.py --input <INPUT_CSV> [--output <OUTPUT_CSV>]

Nếu không truyền --output, mặc định:
    Input:  datasets/<dataset>/<version>/raw/<raw_filename>.csv
    Output: datasets/<dataset>/<version>/processed/processed_<raw_filename>.csv

Tuỳ chỉnh cleaning:
    Sửa trực tiếp trong hàm main() để override config.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.cleaning_config import default_cleaning_config
from config.dataset_config import default_dataset_config
from config.path_config import default_path_config
from src.pipeline.cleaning_pipeline import clean_text_pipeline
from src.utils.csv import read_csv_with_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Làm sạch dataset.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn file CSV đầu vào.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file CSV đầu ra (mặc định: tự sinh từ input path).",
    )
    return parser.parse_args()


def _parse_dataset_info(input_path: Path) -> tuple[str, str]:
    """Parse dataset_name và version từ input path.
    Ví dụ: datasets/custom_dataset/v1/raw/raw_dataset.csv -> ("custom_dataset", "v1")
    """
    input_str = str(input_path).replace("\\", "/")
    parts = input_str.split("/")
    try:
        datasets_idx = parts.index("datasets")
        return parts[datasets_idx + 1], parts[datasets_idx + 2]
    except (ValueError, IndexError):
        return "custom_dataset", "v1"


def _infer_output_path(input_path: Path) -> Path:
    """Tự sinh output path từ input path.
    Ví dụ:
        datasets/custom_dataset/v1/raw/raw_dataset.csv
        -> datasets/custom_dataset/v1/processed/processed_dataset.csv
    """
    dataset_name, version = _parse_dataset_info(input_path)
    path_cfg = default_path_config
    processed_dir = path_cfg.get_processed_dir(dataset_name, version)
    processed_filename = path_cfg.get_processed_filename(input_path.name)
    return Path(path_cfg.project_root) / processed_dir / processed_filename


def main() -> None:
    args = parse_args()

    # ==========================================================
    # [TUỲ CHỈNH] Cấu hình cleaning
    # ==========================================================
    dataset_config = default_dataset_config.override(
        # Ví dụ override:
        # comment_col="comment",
        # label_col="is_toxic",
    )
    cleaning_config = default_cleaning_config.override(
        # Ví dụ override:
        # keep_punctuation=r".,!?",
        # max_null_label_ratio=0.05,
        # outlier_enabled=False,
    )
    # ==========================================================

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _infer_output_path(input_path)
    dataset_name, version = _parse_dataset_info(input_path)

    # Load dataset
    df = read_csv_with_columns(
        str(input_path),
        comment_col=dataset_config.comment_col,
        label_col=dataset_config.label_col,
    )

    # Cleaning pipeline (từ src/pipeline) — tự động lưu meta/
    df_cleaned, report = clean_text_pipeline(
        df=df,
        dataset_config=dataset_config,
        cleaning_config=cleaning_config,
        dataset_name=dataset_name,
        version=version,
    )

    # Save output CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved cleaned dataset: {output_path}")
    print(f"Rows before: {len(df)}")
    print(f"Rows after : {len(df_cleaned)}")

    if "_meta_saved_to" in report:
        print(f"Cleaning report: {report['_meta_saved_to']}")


if __name__ == "__main__":
    main()
