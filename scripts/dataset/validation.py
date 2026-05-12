import argparse
import json
from pathlib import Path

import pandas as pd

from src.dataset.validation import (
    check_column_names,
    check_duplicates,
    check_empty_or_no_letter,
    check_null,
)


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4,
        )


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


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality.")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV dataset.",
    )

    parser.add_argument(
        "--comment-col",
        type=str,
        default="comment",
    )

    parser.add_argument(
        "--label-col",
        type=str,
        default="is_toxic",
    )

    parser.add_argument(
        "--required-cols",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/validation",
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
    )

    args = parser.parse_args()

    # ==========================================================
    # Load dataset
    # ==========================================================
    df = pd.read_csv(
        args.input,
        encoding=args.encoding,
    )

    print(f"[INFO] Loaded dataset: {args.input}")
    print(f"[INFO] Shape: {df.shape}")

    # ==========================================================
    # Validation pipeline
    # ==========================================================
    report = {}

    # Column check
    if args.required_cols:
        report["column_check"] = check_column_names(
            df,
            args.required_cols,
        )

    # Null check
    report["null_check"] = check_null(df)

    # Empty/no-letter check
    report["empty_or_no_letter_check"] = check_empty_or_no_letter(
        df,
        col=args.comment_col,
    )

    # Duplicate check
    report["duplicate_check"] = check_duplicates(
        df,
        comment_col=args.comment_col,
        label_col=args.label_col,
    )

    # ==========================================================
    # Save outputs
    # ==========================================================
    output_dir = Path(args.output_dir)
    dataset_name = Path(args.input).stem

    json_path = output_dir / f"{dataset_name}_validation.json"
    md_path = output_dir / f"{dataset_name}_validation.md"

    save_json(report, json_path)

    markdown_report = generate_markdown_report(report)
    save_markdown(markdown_report, md_path)

    # ==========================================================
    # Console summary
    # ==========================================================
    print("\n========== VALIDATION SUMMARY ==========")

    dup_info = report["duplicate_check"]

    print(f"Duplicated rows: {dup_info['total_duplicated_rows']}")

    print(f"Mixed-label duplicates: {dup_info['mixed_label_comments_count']}")

    print(f"Null ratio: {report['null_check']['null_ratio']:.4f}")

    print(f"Empty comments: {report['empty_or_no_letter_check'].get('empty_count', 0)}")

    print("\n[INFO] Reports saved:")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
