"""
cleaning_pipeline.py

Pipeline làm sạch dataset (cleaning).
Sử dụng BasePipeline.run_steps() để chạy các bước theo config.
Tự động lưu cleaning_log.yaml vào thư mục meta/ sau khi chạy.

Pipeline order (giống notebook Method 10 - ALL cleaning methods combined):
  1. Text-level: html -> urls -> mentions -> emoji -> special_chars
  2. Dataset-level: null/empty -> non-text -> duplicates -> outliers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.cleaning_config import CleaningConfig, default_cleaning_config
from config.dataset_config import DatasetConfig, default_dataset_config
from config.path_config import default_path_config
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
from src.dataset.feature_engineering import (
    add_length_features,
    add_punctuation_emoji_features,
)
from src.dataset.preprocessing import normalize_text
from src.pipeline.base_pipeline import BasePipeline, PipelineStep

# --- Hàm wrapper: chuẩn hóa interface (data, **kwargs) -> (data, dict) ---


def _apply_to_comment_col(
    df: pd.DataFrame, func: Any, comment_col: str, **kwargs: Any
) -> tuple[pd.DataFrame, dict]:
    """Áp dụng hàm text-level lên cột comment."""
    df = df.copy()
    df[comment_col] = df[comment_col].astype(str).apply(func)
    return df, {"applied": True}


def _remove_special_chars_wrapper(
    df: pd.DataFrame, comment_col: str, keep_punctuation: str = r".,!?", **kwargs: Any
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df[comment_col] = df[comment_col].apply(
        lambda x: remove_special_chars(x, keep_punctuation=keep_punctuation)
    )
    return df, {"applied": True}


def _remove_outliers_wrapper(
    df: pd.DataFrame,
    comment_col: str,
    label_col: str,
    feature_cols: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict]:
    if feature_cols is None:
        feature_cols = [
            "word_len",
            "char_len",
            "num_exclamation",
            "num_question",
            "num_upper",
            "num_emoji",
        ]

    # Tạo feature columns tạm thời
    df_temp = add_length_features(df, comment_col=comment_col)
    df_temp = add_punctuation_emoji_features(df_temp, comment_col=comment_col)

    df_result, report = remove_outliers(
        df_temp,
        feature_cols=feature_cols,
        contamination=contamination,
        random_state=random_state,
        label_col=label_col,
    )

    # Drop temporary feature columns
    df_result = df_result.drop(columns=feature_cols, errors="ignore")
    return df_result, report


def _normalize_before_dedup(
    df: pd.DataFrame, comment_col: str, **kwargs: Any
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df[comment_col] = df[comment_col].astype(str).apply(normalize_text)
    return df, {"applied": True}


def _remove_non_text_wrapper(
    df: pd.DataFrame, comment_col: str = "comment", **kwargs: Any
) -> tuple[pd.DataFrame, dict]:
    """Wrapper cho remove_non_text_comments: bỏ qua các kwargs không cần thiết (vd: label_col)."""
    return remove_non_text_comments(df, comment_col=comment_col)


# --- Định nghĩa các bước cleaning ---

CLEANING_STEPS = [
    PipelineStep(
        name="remove_html",
        enabled_flag="enable_html_removal",
        func=_apply_to_comment_col,
        kwargs={"func": remove_html_and_entities},
    ),
    PipelineStep(
        name="remove_urls",
        enabled_flag="enable_url_removal",
        func=_apply_to_comment_col,
        kwargs={"func": remove_urls},
    ),
    PipelineStep(
        name="remove_mentions",
        enabled_flag="enable_mention_removal",
        func=_apply_to_comment_col,
        kwargs={"func": remove_mentions},
    ),
    PipelineStep(
        name="remove_emoji",
        enabled_flag="enable_emoji_removal",
        func=_apply_to_comment_col,
        kwargs={"func": remove_emoji},
    ),
    PipelineStep(
        name="remove_special_chars",
        enabled_flag="enable_special_chars_removal",
        func=_remove_special_chars_wrapper,
    ),
    PipelineStep(
        name="remove_null_empty",
        enabled_flag="enable_null_empty_removal",
        func=remove_null_or_empty,
    ),
    PipelineStep(
        name="remove_non_text",
        enabled_flag="enable_non_text_removal",
        func=_remove_non_text_wrapper,
    ),
    PipelineStep(
        name="normalize_before_dedup",
        enabled_flag="enable_duplicate_removal",
        func=_normalize_before_dedup,
    ),
    PipelineStep(
        name="remove_duplicates",
        enabled_flag="enable_duplicate_removal",
        func=remove_duplicate_comments,
    ),
    PipelineStep(
        name="remove_outliers",
        enabled_flag="enable_outlier_removal",
        func=_remove_outliers_wrapper,
    ),
]


def clean_text_pipeline(
    df: pd.DataFrame,
    dataset_config: DatasetConfig | None = None,
    cleaning_config: CleaningConfig | None = None,
    dataset_name: str = "custom_dataset",
    version: str = "v1",
    save_meta: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline làm sạch text: áp dụng các bước cleaning lên cột comment.

    Parameters
    ----------
    df : DataFrame đầu vào
    dataset_config : Đối tượng DatasetConfig (mặc định: default_dataset_config)
    cleaning_config : Đối tượng CleaningConfig (mặc định: default_cleaning_config)
    dataset_name : Tên dataset (mặc định: 'custom_dataset')
    version : Version dataset (mặc định: 'v1')
    save_meta : Tự động lưu cleaning_log.yaml vào meta/ (mặc định: True)

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame đã làm sạch và báo cáo.
    """
    dcfg = dataset_config or default_dataset_config
    ccfg = cleaning_config or default_cleaning_config

    # Cập nhật kwargs cho các steps dựa trên config
    for step in CLEANING_STEPS:
        step.kwargs["comment_col"] = dcfg.comment_col
        step.kwargs["label_col"] = dcfg.label_col

        if step.name == "remove_special_chars":
            step.kwargs["keep_punctuation"] = ccfg.keep_punctuation
        elif step.name == "remove_null_empty":
            step.kwargs["max_null_label_ratio"] = ccfg.max_null_label_ratio
        elif step.name == "remove_outliers":
            step.kwargs["feature_cols"] = ccfg.outlier_feature_cols
            step.kwargs["contamination"] = ccfg.outlier_contamination
            step.kwargs["random_state"] = ccfg.outlier_random_state

    # Chạy pipeline
    df_cleaned, report = BasePipeline.run_steps(df, ccfg, CLEANING_STEPS)

    # Thêm thông tin tổng quan
    report["final_rows"] = len(df_cleaned)
    report["final_columns"] = list(df_cleaned.columns)

    # Tự động lưu meta
    if save_meta:
        path_cfg = default_path_config
        meta_dir = Path(path_cfg.project_root) / path_cfg.get_meta_dir(
            dataset_name, version
        )
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "cleaning_log.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["_meta_saved_to"] = str(meta_path)

    return df_cleaned, report
