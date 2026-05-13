"""
validation.py

Script CLI để kiểm tra chất lượng dataset (validate).

Cách dùng:
    python scripts/dataset/validation.py --input <INPUT_CSV>

Tuỳ chỉnh validation:
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/validation",
        help="Thư mục lưu báo cáo (mặc định: reports/validation).",
    )
    return parser.parse_args()


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def generate_markdown_report(report: dict) -> str:
    lines = []
    lines.append("# Dataset Validation Report\n")

    for section_name, section_data in report.items():
        lines.append(f"## {section_name}\n")

        if isinstance(section_data, dict):
            for key, value in section_data.items():
                lines.append(f"- **{key}**: `{value}`")
        else:
            lines.append(str(section_data))

        lines.append("")

    return "\n".join(lines)


def save_markdown(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    args = parse_args()

    # ==========================================================
    # [TUỲ CHỈNH] Cấu hình validation
    # ==========================================================
    validation_config = default_validation_config.override(
        # Ví dụ override:
        # comment_col="comment",
        # label_col="is_toxic",
        # required_cols=["comment", "is_toxic"],
        # enable_column_check=True,
        # enable_null_check=True,
        # enable_empty_or_no_letter_check=True,
        # enable_duplicate_check=True,
    )
    # ==========================================================

    # Load dataset
    df = read_csv_with_columns(
        args.input,
        comment_col=validation_config.comment_col,
        label_col=validation_config.label_col,
        encoding=validation_config.encoding,
    )

    # Validation pipeline (từ src/pipeline)
    report = validate_dataset(df, validation_config=validation_config)

    # Save outputs
    output_dir = Path(args.output_dir)
    dataset_name = Path(args.input).stem

    json_path = output_dir / f"{dataset_name}_validation.json"
    md_path = output_dir / f"{dataset_name}_validation.md"

    save_json(report, json_path)

    markdown_report = generate_markdown_report(report)
    save_markdown(markdown_report, md_path)

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

    print("\n[INFO] Reports saved:")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
