"""
preprocessing.py

Script CLI để tiền xử lý văn bản (preprocessing).
Pipeline tự động lưu preprocessing_params.json vào thư mục meta/.

Cách dùng CLI:
    python scripts/dataset/preprocessing.py --input <INPUT_CSV> [--output <OUTPUT_CSV>]

Nếu không truyền --output, mặc định:
    Input:  datasets/<dataset>/<version>/processed/<filename>.csv
    Output: datasets/<dataset>/<version>/processed/<filename>.csv (ghi đè)

Cách dùng trong code Python / Colab:
    from scripts.dataset.preprocessing import preprocess_dataset

    df_preprocessed = preprocess_dataset(
        input_path="datasets/custom_dataset/v1/processed/processed_dataset.csv",
        comment_col="comment",
        label_col="is_toxic",
    )

Tuỳ chỉnh preprocessing:
    Sửa trực tiếp trong hàm main() hoặc dùng config/preprocessing_config.py,
    config/dataset_config.py
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
from config.path_config import default_path_config
from config.preprocessing_config import default_preprocessing_config
from src.pipeline.preprocessing_pipeline import preprocess_text_pipeline
from src.utils.csv import read_csv_with_columns


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


def preprocess_dataset(
    input_path: str,
    output_path: str | None = None,
    comment_col: str = "comment",
    label_col: str = "is_toxic",
    encoding: str = "utf-8",
    preprocessing_config_override: dict[str, Any] | None = None,
    dataset_config_override: dict[str, Any] | None = None,
) -> Any:
    """
    Tiền xử lý dataset và lưu kết quả.

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
    preprocessing_config_override : dict | None
        Các tham số ghi đè cho PreprocessingConfig.
    dataset_config_override : dict | None
        Các tham số ghi đè cho DatasetConfig.

    Returns
    -------
    DataFrame
        DataFrame đã được tiền xử lý.
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
    preprocess_config = default_preprocessing_config.override(
        **(preprocessing_config_override or {}),
    )

    df = read_csv_with_columns(
        str(input_path_obj),
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

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path_obj, index=False, encoding="utf-8-sig")

    print(f"Saved preprocessed dataset: {output_path_obj}")
    print(f"Rows before: {len(df)}")
    print(f"Rows after : {len(df)}")

    return df


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

    # CLI args override config
    comment_col = args.comment_col or dataset_config.comment_col
    label_col = args.label_col or dataset_config.label_col

    preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        comment_col=comment_col,
        label_col=label_col,
        encoding=dataset_config.encoding,
        preprocessing_config_override=preprocess_config.to_dict(),
        dataset_config_override=dataset_config.to_dict(),
    )


if __name__ == "__main__":
    main()
