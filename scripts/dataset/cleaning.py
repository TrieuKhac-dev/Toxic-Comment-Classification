"""
cleaning.py

Script CLI để làm sạch dataset toxic comment.

Cách dùng:
    python scripts/dataset/cleaning.py --input <INPUT_CSV> --output <OUTPUT_CSV>

Tuỳ chỉnh pipeline:
    Mở file này, sửa trực tiếp trong hàm main() để:
    - Bật/tắt các bước cleaning (outlier, duplicate, non-text filter, ...)
    - Thay đổi tham số (keep_punctuation, max_null_label_ratio, ...)
    - Cung cấp file stopwords riêng
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.cleaning_config import default_cleaning_config
from config.dataset_config import default_dataset_config
from src.dataset.cleaning import (
    remove_duplicate_comments,
    remove_emoji,
    remove_html_and_entities,
    remove_mentions,
    remove_non_text_comments,
    remove_null_or_empty,
    remove_outliers,
    remove_special_chars,
    remove_urls,
)
from src.dataset.loader import read_csv_with_columns
from src.dataset.preprocessing import filter_stopwords, normalize_text
from src.dataset.validation import check_column_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Làm sạch dataset toxic comment.")
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn file CSV đầu vào.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Đường dẫn file CSV đầu ra (đã làm sạch).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Đường dẫn file JSON báo cáo (mặc định: {output}_clean_report.json).",
    )
    return parser.parse_args()


def load_stopwords(path: str | None) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Stopwords file not found: {path}")
    with open(file_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def clean_comment(
    text: str,
    keep_punctuation: str,
    keep_emoji: bool,
    keep_stopwords: bool,
    stopwords: list[str],
) -> str:
    text = normalize_text(text)
    text = remove_html_and_entities(text)
    text = remove_urls(text)
    text = remove_mentions(text)

    if not keep_emoji:
        text = remove_emoji(text)

    text = remove_special_chars(text, keep_punctuation=keep_punctuation)
    text = normalize_text(text)

    if not keep_stopwords and stopwords:
        text = filter_stopwords(
            text.split(), stopwords=set(stopwords), return_tokens=False
        )

    text = normalize_text(text)
    return text


def clean_dataset(
    df: pd.DataFrame,
    comment_col: str,
    label_col: str,
    keep_punctuation: str,
    max_null_label_ratio: float,
    enable_outlier: bool,
    outlier_contamination: float,
    outlier_random_state: int,
    outlier_feature_cols: list[str] | None,
    keep_emoji: bool,
    keep_stopwords: bool,
    stopwords: list[str],
    skip_non_text_filter: bool,
    skip_duplicate_filter: bool,
) -> tuple[pd.DataFrame, dict]:
    report: dict = {}

    required_cols = [comment_col, label_col]
    col_check = check_column_names(df, required_cols)
    report["column_check"] = col_check

    missing = [col for col, ok in col_check.items() if not ok]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df, rep = remove_null_or_empty(
        df,
        comment_col=comment_col,
        label_col=label_col,
        max_null_label_ratio=max_null_label_ratio,
    )
    report["remove_null_or_empty"] = rep

    df = df.copy()
    df[comment_col] = (
        df[comment_col]
        .astype(str)
        .apply(
            lambda x: clean_comment(
                x,
                keep_punctuation=keep_punctuation,
                keep_emoji=keep_emoji,
                keep_stopwords=keep_stopwords,
                stopwords=stopwords,
            )
        )
    )

    if not skip_non_text_filter:
        df, rep = remove_non_text_comments(df, comment_col=comment_col)
        report["remove_non_text_comments"] = rep

    if not skip_duplicate_filter:
        df, rep = remove_duplicate_comments(
            df,
            comment_col=comment_col,
            label_col=label_col,
        )
        report["remove_duplicate_comments"] = rep

    if enable_outlier:
        df, rep = remove_outliers(
            df,
            feature_cols=outlier_feature_cols,
            contamination=outlier_contamination,
            random_state=outlier_random_state,
            label_col=label_col,
        )
        report["remove_outliers"] = rep

    report["final_rows"] = len(df)
    report["final_columns"] = list(df.columns)
    return df, report


def main() -> None:
    args = parse_args()

    # ==========================================================
    # [TUỲ CHỈNH] Cấu hình cleaning
    # ==========================================================
    # Tên cột
    comment_col = default_dataset_config.comment_col
    label_col = default_dataset_config.label_col

    # Cleaning text
    keep_punctuation = default_cleaning_config.keep_punctuation
    keep_emoji = False  # True: giữ emoji, False: xóa emoji
    keep_stopwords = False  # True: giữ stopwords, False: xóa stopwords
    stopwords_file = None  # Đường dẫn file stopwords (None: dùng mặc định)

    # Xử lý null
    max_null_label_ratio = default_cleaning_config.max_null_label_ratio

    # Outlier detection (IsolationForest)
    enable_outlier = default_cleaning_config.outlier_enabled
    outlier_contamination = default_cleaning_config.outlier_contamination
    outlier_random_state = default_cleaning_config.outlier_random_state
    outlier_feature_cols = default_cleaning_config.outlier_feature_cols

    # Bật/tắt các bước lọc
    skip_non_text_filter = False  # True: bỏ qua lọc comment không có chữ
    skip_duplicate_filter = False  # True: bỏ qua lọc duplicate
    # ==========================================================

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = (
        Path(args.report)
        if args.report
        else output_path.with_name(f"{output_path.stem}_clean_report.json")
    )

    df = read_csv_with_columns(
        str(input_path),
        comment_col=comment_col,
        label_col=label_col,
    )
    stopwords = load_stopwords(stopwords_file)

    df_clean, report = clean_dataset(
        df=df,
        comment_col=comment_col,
        label_col=label_col,
        keep_punctuation=keep_punctuation,
        max_null_label_ratio=max_null_label_ratio,
        enable_outlier=enable_outlier,
        outlier_contamination=outlier_contamination,
        outlier_random_state=outlier_random_state,
        outlier_feature_cols=outlier_feature_cols,
        keep_emoji=keep_emoji,
        keep_stopwords=keep_stopwords,
        stopwords=stopwords,
        skip_non_text_filter=skip_non_text_filter,
        skip_duplicate_filter=skip_duplicate_filter,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned dataset: {output_path}")
    print(f"Saved cleaning report: {report_path}")
    print(f"Rows before: {len(df)}")
    print(f"Rows after : {len(df_clean)}")


if __name__ == "__main__":
    main()
