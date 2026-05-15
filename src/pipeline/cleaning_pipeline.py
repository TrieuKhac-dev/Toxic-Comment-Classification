"""
cleaning_pipeline.py

Pipeline làm sạch dataset (cleaning).
Chứa luồng xử lý (pipeline) cho cleaning, sử dụng các hàm từ src/dataset/cleaning.
Có thể nhận đối tượng config để cấu hình.

Pipeline order (giống notebook Method 10 - ALL cleaning methods combined):
  1. Text-level: html -> urls -> mentions -> emoji -> special_chars
  2. Dataset-level: null/empty -> non-text -> duplicates -> outliers
"""

from __future__ import annotations

import pandas as pd

from config.cleaning_config import CleaningConfig, default_cleaning_config
from config.dataset_config import DatasetConfig, default_dataset_config
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
from src.dataset.feature_enginering import (
    add_length_features,
    add_punctuation_emoji_features,
)
from src.dataset.preprocessing import normalize_text


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

    # ==========================================================
    # Step 1: Text-level transformations
    # ==========================================================
    df[comment_col] = df[comment_col].astype(str).apply(remove_html_and_entities)
    df[comment_col] = df[comment_col].apply(remove_urls)
    df[comment_col] = df[comment_col].apply(remove_mentions)
    df[comment_col] = df[comment_col].apply(remove_emoji)
    df[comment_col] = df[comment_col].apply(
        lambda x: remove_special_chars(x, keep_punctuation=ccfg.keep_punctuation)
    )

    # ==========================================================
    # Step 2: Xóa null/empty
    # ==========================================================
    df, rep = remove_null_or_empty(
        df,
        comment_col=comment_col,
        label_col=label_col,
        max_null_label_ratio=ccfg.max_null_label_ratio,
    )
    report["remove_null_or_empty"] = rep

    # ==========================================================
    # Step 3: Xóa comment không chứa chữ cái
    # ==========================================================
    df, rep = remove_non_text_comments(df, comment_col=comment_col)
    report["remove_non_text_comments"] = rep

    # ==========================================================
    # Step 4: Normalize text trước khi xóa duplicate
    # ==========================================================
    df[comment_col] = df[comment_col].astype(str).apply(normalize_text)

    # ==========================================================
    # Step 5: Xóa duplicate
    # ==========================================================
    df, rep = remove_duplicate_comments(
        df,
        comment_col=comment_col,
        label_col=label_col,
    )
    report["remove_duplicate_comments"] = rep

    # ==========================================================
    # Step 6: Xóa outlier (nếu được bật)
    # ==========================================================
    if ccfg.outlier_enabled:
        # Tạo feature columns tạm thời (giống notebook)
        df_temp = add_length_features(df, comment_col=comment_col)
        df_temp = add_punctuation_emoji_features(df_temp, comment_col=comment_col)

        df, rep = remove_outliers(
            df_temp,
            feature_cols=ccfg.outlier_feature_cols,
            contamination=ccfg.outlier_contamination,
            random_state=ccfg.outlier_random_state,
            label_col=label_col,
        )

        # Drop temporary feature columns (giống notebook)
        df = df.drop(
            columns=[
                "word_len",
                "char_len",
                "num_exclamation",
                "num_question",
                "num_upper",
                "num_emoji",
            ],
            errors="ignore",
        )
        report["remove_outliers"] = rep

    report["final_rows"] = len(df)
    report["final_columns"] = list(df.columns)
    return df, report
