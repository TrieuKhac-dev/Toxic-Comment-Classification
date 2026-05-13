"""
preprocessing.py

Script CLI để tiền xử lý văn bản (preprocessing).

Cách dùng:
    python scripts/dataset/preprocessing.py --input <INPUT_CSV> --output <OUTPUT_CSV>

Tuỳ chỉnh preprocessing:
    Sửa trực tiếp trong hàm main() hoặc dùng config/preprocessing_config.py,
    config/dataset_config.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.dataset_config import default_dataset_config
from config.preprocessing_config import default_preprocessing_config
from config.validation_config import default_validation_config
from src.dataset.loader import read_csv_with_columns
from src.pipeline.preprocessing_pipeline import preprocess_text_pipeline
from src.pipeline.validation_pipeline import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiền xử lý văn bản dataset.")
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn file CSV đầu vào.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Đường dẫn file CSV đầu ra (đã tiền xử lý).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Đường dẫn file JSON báo cáo (mặc định: {output}_preprocess_report.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ==========================================================
    # [TUỲ CHỈNH] Cấu hình preprocessing
    # ==========================================================
    dataset_config = default_dataset_config.override(
        # Ví dụ override:
        # comment_col="comment",
        # label_col="is_toxic",
    )
    preprocess_config = default_preprocessing_config.override(
        # Ví dụ override:
        # normalize_lower=True,
        # normalize_strip_spaces=True,
        # return_tokens=False,
    )
    # ==========================================================

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = (
        Path(args.report)
        if args.report
        else output_path.with_name(f"{output_path.stem}_preprocess_report.json")
    )

    df = read_csv_with_columns(
        str(input_path),
        comment_col=dataset_config.comment_col,
        label_col=dataset_config.label_col,
    )

    # Kiểm tra cột bắt buộc bằng validation pipeline
    validation_config = default_validation_config.override(
        comment_col=dataset_config.comment_col,
        label_col=dataset_config.label_col,
        enable_null_check=False,
        enable_empty_or_no_letter_check=False,
        enable_duplicate_check=False,
    )
    validation_report = validate_dataset(df, validation_config=validation_config)
    col_check = validation_report.get("column_check", {})
    missing = [col for col, ok in col_check.items() if not ok]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Tiền xử lý từng comment bằng pipeline (từ src/pipeline)
    df = df.copy()
    df[dataset_config.comment_col] = (
        df[dataset_config.comment_col]
        .astype(str)
        .apply(lambda x: preprocess_text_pipeline(x, config=preprocess_config))
    )

    report = {
        "validation": validation_report,
        "final_rows": len(df),
        "final_columns": list(df.columns),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved preprocessed dataset: {output_path}")
    print(f"Saved preprocessing report: {report_path}")
    print(f"Rows before: {len(df)}")
    print(f"Rows after : {len(df)}")


if __name__ == "__main__":
    main()
