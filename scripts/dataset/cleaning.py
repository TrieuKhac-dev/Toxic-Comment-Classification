"""
cleaning.py

Script CLI để làm sạch dataset (cleaning).
Pipeline tự động lưu cleaning_log.json vào thư mục meta/.

Cách dùng CLI:
    python scripts/dataset/cleaning.py --input <INPUT_CSV> [--output <OUTPUT_CSV>]

Nếu không truyền --output, mặc định:
    Input:  datasets/<dataset>/<version>/raw/<raw_filename>.csv
    Output: datasets/<dataset>/<version>/processed/processed_<raw_filename>.csv

Cách dùng trong code Python / Colab:
    from scripts.dataset.cleaning import clean_dataset

    df_cleaned, report = clean_dataset(
        input_path="datasets/custom_dataset/v1/raw/raw_dataset.csv",
        comment_col="comment",
        label_col="is_toxic",
    )

Tuỳ chỉnh cleaning:
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


def clean_dataset(
    input_path: str,
    output_path: str | None = None,
    comment_col: str = "comment",
    label_col: str = "is_toxic",
    encoding: str = "utf-8",
    cleaning_config_override: dict[str, Any] | None = None,
    dataset_config_override: dict[str, Any] | None = None,
) -> tuple[Any, dict]:
    """
    Làm sạch dataset và lưu kết quả.

    Hàm này có thể gọi trực tiếp từ code Python hoặc Colab,
    không phụ thuộc argparse.

    Parameters
    ----------
    input_path : str
        Đường dẫn file CSV đầu vào.
    output_path : str | None
        Đường dẫn file CSV đầu ra. Nếu None, tự sinh từ input path.
    comment_col : str
        Tên cột comment (mặc định: "comment").
    label_col : str
        Tên cột nhãn (mặc định: "is_toxic").
    encoding : str
        Encoding file CSV (mặc định: "utf-8").
    cleaning_config_override : dict | None
        Các tham số ghi đè cho CleaningConfig.
    dataset_config_override : dict | None
        Các tham số ghi đè cho DatasetConfig.

    Returns
    -------
    tuple[DataFrame, dict]
        (df_cleaned, cleaning_report)
    """
    input_path_obj = Path(input_path)
    output_path_obj = (
        Path(output_path) if output_path else _infer_output_path(input_path_obj)
    )
    dataset_name, version = _parse_dataset_info(input_path_obj)

    # Config
    dataset_config = default_dataset_config.override(
        comment_col=comment_col,
        label_col=label_col,
        encoding=encoding,
        **(dataset_config_override or {}),
    )
    cleaning_config = default_cleaning_config.override(
        **(cleaning_config_override or {}),
    )

    # Load dataset
    df = read_csv_with_columns(
        str(input_path_obj),
        comment_col=dataset_config.comment_col,
        label_col=dataset_config.label_col,
    )

    # Cleaning pipeline
    df_cleaned, report = clean_text_pipeline(
        df=df,
        dataset_config=dataset_config,
        cleaning_config=cleaning_config,
        dataset_name=dataset_name,
        version=version,
    )

    # Save output CSV
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_path_obj, index=False, encoding="utf-8-sig")

    print(f"Saved cleaned dataset: {output_path_obj}")
    print(f"Rows before: {len(df)}")
    print(f"Rows after : {len(df_cleaned)}")

    if "_meta_saved_to" in report:
        print(f"Cleaning report: {report['_meta_saved_to']}")

    return df_cleaned, report


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

    # CLI args override config
    comment_col = args.comment_col or dataset_config.comment_col
    label_col = args.label_col or dataset_config.label_col

    clean_dataset(
        input_path=args.input,
        output_path=args.output,
        comment_col=comment_col,
        label_col=label_col,
        encoding=dataset_config.encoding,
        cleaning_config_override=cleaning_config.to_dict(),
        dataset_config_override=dataset_config.to_dict(),
    )


if __name__ == "__main__":
    main()
