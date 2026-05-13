"""
cleaning.py

Script CLI để làm sạch dataset (cleaning).

Cách dùng:
    python scripts/dataset/cleaning.py --input <INPUT_CSV> --output <OUTPUT_CSV>

Tuỳ chỉnh cleaning:
    Sửa trực tiếp trong hàm main() để override config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.cleaning_config import default_cleaning_config
from config.dataset_config import default_dataset_config
from src.dataset.loader import read_csv_with_columns
from src.pipeline.cleaning_pipeline import clean_text_pipeline


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
        required=True,
        help="Đường dẫn file CSV đầu ra (đã làm sạch).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Đường dẫn file JSON báo cáo (mặc định: {output}_cleaning_report.json).",
    )
    return parser.parse_args()


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
    output_path = Path(args.output)
    report_path = (
        Path(args.report)
        if args.report
        else output_path.with_name(f"{output_path.stem}_cleaning_report.json")
    )

    # Load dataset
    df = read_csv_with_columns(
        str(input_path),
        comment_col=dataset_config.comment_col,
        label_col=dataset_config.label_col,
    )

    # Cleaning pipeline (từ src/pipeline)
    df_cleaned, report = clean_text_pipeline(
        df=df,
        dataset_config=dataset_config,
        cleaning_config=cleaning_config,
    )

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df_cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned dataset: {output_path}")
    print(f"Saved cleaning report: {report_path}")
    print(f"Rows before: {len(df)}")
    print(f"Rows after : {len(df_cleaned)}")


if __name__ == "__main__":
    main()
