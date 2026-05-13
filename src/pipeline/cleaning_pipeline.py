"""
cleaning_pipeline.py

Pipeline làm sạch dataset (cleaning).
Chứa luồng xử lý (pipeline) cho cleaning, sử dụng các hàm từ src/dataset/cleaning.
Có thể nhận đối tượng config để cấu hình.
"""

from __future__ import annotations

import pandas as pd

from config.cleaning_config import CleaningConfig, default_cleaning_config
from config.dataset_config import DatasetConfig, default_dataset_config
from src.dataset.cleaning import (
    remove_duplicate_comments,
    remove_non_text_comments,
    remove_null_or_empty,
    remove_outliers,
)


def clean_text_pipeline(
    df: pd.DataFrame,
    dataset_config: DatasetConfig | None = None,
    cleaning_config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline làm sạch text: áp dụng các bước cleaning lên cột comment.

    Parameters
    ----------
    df : DataFrame đầu vào
    dataset_config : Đối tượng DatasetConfig (mặc định: default_dataset_config)
    cleaning_config : Đối tượng CleaningConfig (mặc định: default_cleaning_config)

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame đã làm sạch và báo cáo.
    """
    dcfg = dataset_config or default_dataset_config
    ccfg = cleaning_config or default_cleaning_config
    report: dict = {}

    comment_col = dcfg.comment_col
    label_col = dcfg.label_col

    # 1. Xóa null/empty
    df, rep = remove_null_or_empty(
        df,
        comment_col=comment_col,
        label_col=label_col,
        max_null_label_ratio=ccfg.max_null_label_ratio,
    )
    report["remove_null_or_empty"] = rep

    # 2. Xóa comment không chứa chữ cái
    df, rep = remove_non_text_comments(df, comment_col=comment_col)
    report["remove_non_text_comments"] = rep

    # 3. Xóa duplicate
    df, rep = remove_duplicate_comments(
        df,
        comment_col=comment_col,
        label_col=label_col,
    )
    report["remove_duplicate_comments"] = rep

    # 4. Xóa outlier (nếu được bật)
    if ccfg.outlier_enabled:
        df, rep = remove_outliers(
            df,
            feature_cols=ccfg.outlier_feature_cols,
            contamination=ccfg.outlier_contamination,
            random_state=ccfg.outlier_random_state,
            label_col=label_col,
        )
        report["remove_outliers"] = rep

    report["final_rows"] = len(df)
    report["final_columns"] = list(df.columns)
    return df, report
