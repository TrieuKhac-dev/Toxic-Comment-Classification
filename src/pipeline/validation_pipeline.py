"""
validation_pipeline.py

Pipeline kiểm tra chất lượng dataset (validation).
Sử dụng BasePipeline.run_steps() để chạy các bước theo config.
Tự động lưu validation_report.yaml vào thư mục meta/ sau khi chạy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.dataset_config import DatasetConfig, default_dataset_config
from config.path_config import default_path_config
from config.validation_config import ValidationConfig, default_validation_config
from src.dataset.validation import (
    check_column_names,
    check_duplicates,
    check_empty_or_no_letter,
    check_null,
)
from src.pipeline.base_pipeline import BasePipeline, PipelineStep

# --- Hàm wrapper: chuẩn hóa interface (data, **kwargs) -> (data, dict) ---


def _check_column_names_wrapper(
    df: pd.DataFrame, required_cols: list[str] | None = None, **kwargs: Any
) -> tuple[pd.DataFrame, dict]:
    """Wrapper cho check_column_names: trả về (data, report) thay vì chỉ report."""
    cols = required_cols or []
    report = check_column_names(df, cols)
    return df, report


def _check_null_wrapper(
    df: pd.DataFrame, cols: list[str] | None = None, **kwargs: Any
) -> tuple[pd.DataFrame, dict]:
    """Wrapper cho check_null: trả về (data, report) thay vì chỉ report."""
    report = check_null(df, cols)
    return df, report


def _check_empty_or_no_letter_wrapper(
    df: pd.DataFrame, col: str | None = None, **kwargs: Any
) -> tuple[pd.DataFrame, dict]:
    """Wrapper cho check_empty_or_no_letter: trả về (data, report) thay vì chỉ report."""
    report = check_empty_or_no_letter(df, col or "")
    return df, report


def _check_duplicates_wrapper(
    df: pd.DataFrame,
    comment_col: str = "",
    label_col: str = "",
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict]:
    """Wrapper cho check_duplicates: trả về (data, report) thay vì chỉ report."""
    report = check_duplicates(df, comment_col, label_col)
    return df, report


# --- Định nghĩa các bước validation ---

VALIDATION_STEPS = [
    PipelineStep(
        name="column_check",
        enabled_flag="enable_column_check",
        func=_check_column_names_wrapper,
    ),
    PipelineStep(
        name="null_check",
        enabled_flag="enable_null_check",
        func=_check_null_wrapper,
    ),
    PipelineStep(
        name="empty_or_no_letter_check",
        enabled_flag="enable_empty_or_no_letter_check",
        func=_check_empty_or_no_letter_wrapper,
    ),
    PipelineStep(
        name="duplicate_check",
        enabled_flag="enable_duplicate_check",
        func=_check_duplicates_wrapper,
    ),
]


def validate_dataset(
    df: pd.DataFrame,
    validation_config: ValidationConfig | None = None,
    dataset_config: DatasetConfig | None = None,
    dataset_name: str = "custom_dataset",
    version: str = "v1",
    save_meta: bool = True,
) -> dict:
    """
    Pipeline validation: chạy tất cả các bước kiểm tra dựa trên config.

    Parameters
    ----------
    df : DataFrame đầu vào
    validation_config : Đối tượng ValidationConfig (mặc định: default_validation_config)
    dataset_config : Đối tượng DatasetConfig (mặc định: default_dataset_config)
    dataset_name : Tên dataset (mặc định: 'custom_dataset')
    version : Version dataset (mặc định: 'v1')
    save_meta : Tự động lưu validation_report.yaml vào meta/ (mặc định: True)

    Returns
    -------
    dict
        Báo cáo validation tổng hợp.
    """
    vcfg = validation_config or default_validation_config
    dcfg = dataset_config or default_dataset_config

    # Cập nhật kwargs cho các steps dựa trên config
    for step in VALIDATION_STEPS:
        if step.name == "column_check":
            step.kwargs["required_cols"] = vcfg.required_cols
        elif step.name == "null_check":
            step.kwargs["cols"] = None  # None = tất cả cột
        elif step.name == "empty_or_no_letter_check":
            step.kwargs["col"] = dcfg.comment_col
        elif step.name == "duplicate_check":
            step.kwargs["comment_col"] = dcfg.comment_col
            step.kwargs["label_col"] = dcfg.label_col

    # Chạy pipeline
    _, report = BasePipeline.run_steps(df, vcfg, VALIDATION_STEPS)

    # Bổ sung config vào report
    report["_config"] = {
        "validation": {
            "required_cols": vcfg.required_cols,
            "enable_column_check": vcfg.enable_column_check,
            "enable_null_check": vcfg.enable_null_check,
            "enable_empty_or_no_letter_check": vcfg.enable_empty_or_no_letter_check,
            "enable_duplicate_check": vcfg.enable_duplicate_check,
        },
        "dataset": {
            "comment_col": dcfg.comment_col,
            "label_col": dcfg.label_col,
        },
    }

    # Tự động lưu meta
    if save_meta:
        path_cfg = default_path_config
        meta_dir = Path(path_cfg.project_root) / path_cfg.get_meta_dir(
            dataset_name, version
        )
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "validation_report.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["_meta_saved_to"] = str(meta_path)

    return report
