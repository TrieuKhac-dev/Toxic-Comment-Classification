"""
split.py

Script CLI để chia dữ liệu (train/val/test split).
Tự động lưu các file CSV vào thư mục split/ và lưu split_report.json vào meta/.

Cách dùng CLI:
    python scripts/dataset/split.py --input <INPUT_CSV>

    # Với tuỳ chỉnh tỷ lệ
    python scripts/dataset/split.py --input <INPUT_CSV> --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

    # Với tuỳ chỉnh tên cột
    python scripts/dataset/split.py --input <INPUT_CSV> --comment-col comment --label-col is_toxic

Cách dùng trong code Python / Colab:
    from scripts.dataset.split import split_dataset

    result = split_dataset(
        input_path="datasets/custom_dataset/v1/processed/processed_dataset.csv",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        comment_col="comment",
        label_col="is_toxic",
    )
    # result["train_df"], result["val_df"], result["test_df"]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.dataset_config import default_dataset_config
from config.path_config import default_path_config
from config.split_config import default_split_config
from src.dataset.split import split_dataframe
from src.utils.csv import read_csv_with_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chia dữ liệu thành train/val/test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chia với tỷ lệ mặc định (70/15/15)
  python scripts/dataset/split.py --input datasets/custom_dataset/v1/processed/processed_dataset.csv

  # Chia với tỷ lệ tuỳ chỉnh
  python scripts/dataset/split.py --input datasets/custom_dataset/v1/processed/processed_dataset.csv --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

  # Chia với tên cột tuỳ chỉnh
  python scripts/dataset/split.py --input datasets/custom_dataset/v1/processed/processed_dataset.csv --comment-col comment --label-col is_toxic
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn file CSV đầu vào.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Tỷ lệ tập train (mặc định: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Tỷ lệ tập validation (mặc định: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Tỷ lệ tập test (mặc định: 0.15).",
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
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random state (mặc định: 42).",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Tắt stratified split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thư mục đầu ra (mặc định: tự sinh từ input path).",
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


def _infer_output_dir(input_path: Path) -> Path:
    """Tự sinh output dir từ input path.
    Ví dụ:
        datasets/custom_dataset/v1/processed/processed_dataset.csv
        -> datasets/custom_dataset/v1/split
    """
    dataset_name, version = _parse_dataset_info(input_path)
    path_cfg = default_path_config
    split_dir = path_cfg.get_split_dir(dataset_name, version)
    return Path(path_cfg.project_root) / split_dir


def _save_split_report(
    result: dict[str, Any],
    output_dir: Path,
    dataset_name: str,
    version: str,
) -> dict[str, Any]:
    """Lưu split report vào thư mục meta/."""
    from config.path_config import default_path_config

    meta_dir = Path(
        default_path_config.project_root
    ) / default_path_config.get_meta_dir(dataset_name, version)
    meta_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_name": dataset_name,
        "version": version,
        "train_ratio": result["train_ratio"],
        "val_ratio": result["val_ratio"],
        "test_ratio": result["test_ratio"],
        "n_train": len(result["train_df"]),
        "n_val": len(result["val_df"]),
        "n_test": len(result["test_df"]),
        "n_total": len(result["train_df"])
        + len(result["val_df"])
        + len(result["test_df"]),
        "output_dir": str(output_dir),
        "train_file": str(output_dir / "train.csv"),
        "val_file": str(output_dir / "val.csv"),
        "test_file": str(output_dir / "test.csv"),
    }

    report_path = meta_dir / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    report["_meta_saved_to"] = str(report_path)
    return report


def split_dataset(
    input_path: str,
    output_dir: str | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    comment_col: str = "comment",
    label_col: str = "is_toxic",
    encoding: str = "utf-8",
    stratify: bool = True,
    random_state: int = 42,
    shuffle: bool = True,
    save_csv: bool = True,
    save_meta: bool = True,
    train_filename: str = "train.csv",
    val_filename: str = "val.csv",
    test_filename: str = "test.csv",
) -> dict[str, Any]:
    """
    Chia dataset thành train/val/test và lưu kết quả.

    Hàm này có thể gọi trực tiếp từ code Python hoặc Colab,
    không phụ thuộc argparse.

    Parameters
    ----------
    input_path : str
        Đường dẫn file CSV đầu vào.
    output_dir : str | None
        Thư mục đầu ra. Nếu None, tự sinh từ input path.
    train_ratio : float
        Tỷ lệ tập train (mặc định: 0.70).
    val_ratio : float
        Tỷ lệ tập validation (mặc định: 0.15).
    test_ratio : float
        Tỷ lệ tập test (mặc định: 0.15).
    comment_col : str
        Tên cột comment (mặc định: "comment").
    label_col : str
        Tên cột nhãn (mặc định: "is_toxic").
    encoding : str
        Encoding file CSV (mặc định: "utf-8").
    stratify : bool
        Có stratified split hay không (mặc định: True).
    random_state : int
        Random state (mặc định: 42).
    shuffle : bool
        Có shuffle dữ liệu trước khi split hay không (mặc định: True).
    save_csv : bool
        Có lưu file CSV hay không (mặc định: True).
    save_meta : bool
        Có lưu split_report.json vào meta/ hay không (mặc định: True).
    train_filename : str
        Tên file train đầu ra (mặc định: "train.csv").
    val_filename : str
        Tên file val đầu ra (mặc định: "val.csv").
    test_filename : str
        Tên file test đầu ra (mặc định: "test.csv").

    Returns
    -------
    dict[str, Any]
        {
            "train_df": pd.DataFrame,
            "val_df": pd.DataFrame,
            "test_df": pd.DataFrame,
            "train_indices": np.ndarray,
            "val_indices": np.ndarray,
            "test_indices": np.ndarray,
            "train_comments": np.ndarray,
            "val_comments": np.ndarray,
            "test_comments": np.ndarray,
            "y_train": np.ndarray,
            "y_val": np.ndarray,
            "y_test": np.ndarray,
            "train_ratio": float,
            "val_ratio": float,
            "test_ratio": float,
            "_meta_saved_to": str | None,
        }
    """
    input_path_obj = Path(input_path)
    dataset_name, version = _parse_dataset_info(input_path_obj)

    # Xác định output_dir
    if output_dir is None:
        output_dir_obj = _infer_output_dir(input_path_obj)
    else:
        output_dir_obj = Path(output_dir)

    # Load dataset
    df = read_csv_with_columns(
        str(input_path_obj),
        comment_col=comment_col,
        label_col=label_col,
        encoding=encoding,
    )

    # Kiểm tra cột bắt buộc
    required_cols = [comment_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Split: lấy indices trước, rồi build result dict
    train_idx, val_idx, test_idx = split_dataframe(
        df=df,
        label_col=label_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=stratify,
        random_state=random_state,
        shuffle=shuffle,
        return_indices=True,
    )

    y = df[label_col].values
    original_comments = df[comment_col].values

    result = {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "train_df": df.iloc[train_idx].reset_index(drop=True),
        "val_df": df.iloc[val_idx].reset_index(drop=True),
        "test_df": df.iloc[test_idx].reset_index(drop=True),
        "train_comments": original_comments[train_idx],
        "val_comments": original_comments[val_idx],
        "test_comments": original_comments[test_idx],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "y_test": y[test_idx],
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

    # In summary
    print(f"\n{'=' * 50}")
    print("SPLIT SUMMARY")
    print(f"{'=' * 50}")
    for name, y_data in [
        ("Train", result["y_train"]),
        ("Val", result["y_val"]),
        ("Test", result["y_test"]),
    ]:
        print(f"\n--- {name} ({len(y_data)} samples) ---")
        unique, counts = np.unique(y_data, return_counts=True)
        for cls, cnt in zip(unique, counts, strict=True):
            print(f"  Class {cls}: {cnt} ({cnt / len(y_data):.2%})")

    # Lưu CSV
    if save_csv:
        output_dir_obj.mkdir(parents=True, exist_ok=True)

        train_path = output_dir_obj / train_filename
        val_path = output_dir_obj / val_filename
        test_path = output_dir_obj / test_filename

        result["train_df"].to_csv(train_path, index=False, encoding="utf-8-sig")
        result["val_df"].to_csv(val_path, index=False, encoding="utf-8-sig")
        result["test_df"].to_csv(test_path, index=False, encoding="utf-8-sig")

        print(f"\nSaved train: {train_path} ({len(result['train_df'])} rows)")
        print(f"Saved val  : {val_path} ({len(result['val_df'])} rows)")
        print(f"Saved test : {test_path} ({len(result['test_df'])} rows)")

    # Lưu meta
    if save_meta:
        report = _save_split_report(result, output_dir_obj, dataset_name, version)
        result["_meta_saved_to"] = report["_meta_saved_to"]
        print(f"\nSplit report saved: {result['_meta_saved_to']}")
    else:
        result["_meta_saved_to"] = None

    return result


def main() -> None:
    args = parse_args()

    # ==========================================================
    # [TUỲ CHỈNH] Cấu hình split
    # ==========================================================
    dataset_config = default_dataset_config.override(
        # Ví dụ override:
        # comment_col="comment",
        # label_col="is_toxic",
    )
    split_config = default_split_config.override(
        # Ví dụ override:
        # train_ratio=0.8,
        # val_ratio=0.1,
        # test_ratio=0.1,
    )
    # ==========================================================

    # CLI args override config
    comment_col = args.comment_col or dataset_config.comment_col
    label_col = args.label_col or dataset_config.label_col
    train_ratio = (
        args.train_ratio if args.train_ratio is not None else split_config.train_ratio
    )
    val_ratio = args.val_ratio if args.val_ratio is not None else split_config.val_ratio
    test_ratio = (
        args.test_ratio if args.test_ratio is not None else split_config.test_ratio
    )
    random_state = (
        args.random_state
        if args.random_state is not None
        else split_config.random_state
    )
    stratify = not args.no_stratify if args.no_stratify else split_config.stratify

    split_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        comment_col=comment_col,
        label_col=label_col,
        encoding=dataset_config.encoding,
        stratify=stratify,
        random_state=random_state,
        shuffle=split_config.shuffle,
        save_csv=True,
        save_meta=True,
        train_filename=split_config.train_filename,
        val_filename=split_config.val_filename,
        test_filename=split_config.test_filename,
    )


if __name__ == "__main__":
    main()
