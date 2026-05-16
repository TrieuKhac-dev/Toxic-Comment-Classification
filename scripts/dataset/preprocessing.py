"""
preprocessing.py

Script CLI để tiền xử lý văn bản (preprocessing).
Pipeline tự động lưu preprocessing_params.json vào thư mục meta/.

Cách dùng:
    python scripts/dataset/preprocessing.py --input <INPUT_CSV> [--output <OUTPUT_CSV>]

Nếu không truyền --output, mặc định:
    Input:  datasets/<dataset>/<version>/processed/<filename>.csv
    Output: datasets/<dataset>/<version>/processed/<filename>.csv (ghi đè)

Tuỳ chỉnh preprocessing:
    Sửa trực tiếp trong hàm main() hoặc dùng config/preprocessing_config.py,
    config/dataset_config.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.dataset_config import default_dataset_config
from config.path_config import default_path_config
from config.preprocessing_config import default_preprocessing_config
from src.dataset.loader import read_csv_with_columns
from src.pipeline.preprocessing_pipeline import preprocess_text_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiền xử lý văn bản dataset.")
    parser.add_argument(
        "--input",
        required=True,
        help="Đường dẫn file CSV đầu vào.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Đường dẫn file CSV đầu ra (mặc định: ghi đè lên input).",
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


def _infer_output_path(input_path: Path) -> Path:
    """Tự sinh output path từ input path.
    - Nếu input đã ở processed/ thì giữ nguyên (ghi đè)
    - Nếu input từ raw/ thì chuyển sang processed/ + thêm prefix processed_
    """
    dataset_name, version = _parse_dataset_info(input_path)
    path_cfg = default_path_config
    processed_dir = path_cfg.get_processed_dir(dataset_name, version)

    # Nếu input đã ở processed/ thì giữ nguyên tên file
    input_str = str(input_path).replace("\\", "/")
    if "processed" in input_str.split("/"):
        return Path(path_cfg.project_root) / processed_dir / input_path.name

    # Nếu input từ raw/ thì thêm prefix processed_
    processed_filename = path_cfg.get_processed_filename(input_path.name)
    return Path(path_cfg.project_root) / processed_dir / processed_filename


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
    output_path = Path(args.output) if args.output else _infer_output_path(input_path)
    dataset_name, version = _parse_dataset_info(input_path)

    df = read_csv_with_columns(
        str(input_path),
        comment_col=dataset_config.comment_col,
        label_col=dataset_config.label_col,
    )

    # Kiểm tra cột bắt buộc
    required_cols = [dataset_config.comment_col, dataset_config.label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Tiền xử lý từng comment bằng pipeline (từ src/pipeline) — tự động lưu meta/
    df = df.copy()
    df[dataset_config.comment_col] = (
        df[dataset_config.comment_col]
        .astype(str)
        .apply(
            lambda x: preprocess_text_pipeline(
                x,
                config=preprocess_config,
                dataset_name=dataset_name,
                version=version,
            )
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved preprocessed dataset: {output_path}")
    print(f"Rows before: {len(df)}")
    print(f"Rows after : {len(df)}")


if __name__ == "__main__":
    main()
