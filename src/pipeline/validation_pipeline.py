"""
validation_pipeline.py

Pipeline kiểm tra chất lượng dataset (validation).
Chứa luồng xử lý (pipeline) cho validation, sử dụng các hàm từ src/dataset/validation.
Có thể nhận đối tượng config để cấu hình.
"""

from __future__ import annotations

import pandas as pd

from config.validation_config import ValidationConfig, default_validation_config
from src.dataset.validation import (
    check_column_names,
    check_duplicates,
    check_empty_or_no_letter,
    check_null,
)


def validate_dataset(
    df: pd.DataFrame,
    validation_config: ValidationConfig | None = None,
) -> dict:
    """
    Pipeline validation: chạy tất cả các bước kiểm tra dựa trên config.

    Parameters
    ----------
    df : DataFrame đầu vào
    validation_config : Đối tượng ValidationConfig (mặc định: default_validation_config)

    Returns
    -------
    dict
        Báo cáo validation tổng hợp.
    """
    cfg = validation_config or default_validation_config
    report: dict = {}

    if cfg.enable_column_check and cfg.required_cols:
        report["column_check"] = check_column_names(df, cfg.required_cols)

    if cfg.enable_null_check:
        report["null_check"] = check_null(df)

    if cfg.enable_empty_or_no_letter_check:
        report["empty_or_no_letter_check"] = check_empty_or_no_letter(
            df, col=cfg.comment_col
        )

    if cfg.enable_duplicate_check:
        report["duplicate_check"] = check_duplicates(
            df,
            comment_col=cfg.comment_col,
            label_col=cfg.label_col,
        )

    return report
