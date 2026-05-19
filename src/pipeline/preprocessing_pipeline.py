"""
preprocessing_pipeline.py

Pipeline tiền xử lý văn bản (preprocessing).
Sử dụng BasePipeline.run_steps() để chạy các bước theo config.
Tự động lưu preprocessing_params.yaml vào thư mục meta/ sau khi chạy.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from config.path_config import default_path_config
from config.preprocessing_config import (
    PreprocessingConfig,
    default_preprocessing_config,
)
from src.dataset.preprocessing import (
    filter_stopwords,
    get_stopwords,
    get_tokenizer,
    normalize_text,
)
from src.pipeline.base_pipeline import BasePipeline, PipelineStep


def get_tokenizer_from_config(
    config: PreprocessingConfig | None = None,
) -> Callable[[str], list[str]]:
    """Lấy tokenizer từ config hoặc mặc định."""
    cfg = config or default_preprocessing_config
    return get_tokenizer(cfg.tokenizer)


def get_stopwords_from_config(config: PreprocessingConfig | None = None) -> set[str]:
    """Lấy stopwords từ config hoặc mặc định."""
    cfg = config or default_preprocessing_config
    return get_stopwords(cfg.stopwords)


# --- Hàm wrapper cho các bước preprocessing ---


def _normalize_step(
    text: str, lower: bool = True, strip_spaces: bool = True, **kwargs: Any
) -> tuple[str, dict]:
    result = normalize_text(text, lower=lower, strip_spaces=strip_spaces)
    return result, {"applied": True, "lower": lower, "strip_spaces": strip_spaces}


def _tokenize_step(
    text: str, tokenizer: Callable[[str], list[str]] | None = None, **kwargs: Any
) -> tuple[list[str], dict]:
    tok = get_tokenizer(tokenizer)
    tokens = tok(text)
    return tokens, {"applied": True, "num_tokens": len(tokens)}


def _filter_stopwords_step(
    tokens: list[str], stopwords: set[str] | None = None, **kwargs: Any
) -> tuple[list[str], dict]:
    sw = get_stopwords(stopwords)
    filtered = filter_stopwords(tokens, stopwords=sw, return_tokens=True)
    if isinstance(filtered, list):
        return filtered, {
            "applied": True,
            "num_before": len(tokens),
            "num_after": len(filtered),
        }
    return list(filtered), {
        "applied": True,
        "num_before": len(tokens),
        "num_after": len(filtered),
    }


# --- Định nghĩa các bước preprocessing ---

PREPROCESSING_STEPS = [
    PipelineStep(
        name="normalize",
        enabled_flag="enable_normalize",
        func=_normalize_step,
    ),
    PipelineStep(
        name="tokenize",
        enabled_flag="enable_tokenize",
        func=_tokenize_step,
    ),
    PipelineStep(
        name="filter_stopwords",
        enabled_flag="enable_stopword_filter",
        func=_filter_stopwords_step,
    ),
]


def preprocess_text_pipeline(
    text: str,
    config: PreprocessingConfig | None = None,
    dataset_name: str = "custom_dataset",
    version: str = "v1",
    save_meta: bool = True,
) -> str | list[str]:
    """
    Pipeline tiền xử lý text hoàn chỉnh: normalize -> tokenize -> filter_stopwords -> join.

    Parameters
    ----------
    text : str
        Văn bản đầu vào.
    config : PreprocessingConfig | None
        Đối tượng PreprocessingConfig (mặc định: default_preprocessing_config).
    dataset_name : Tên dataset (mặc định: 'custom_dataset')
    version : Version dataset (mặc định: 'v1')
    save_meta : Tự động lưu preprocessing_params.yaml vào meta/ (mặc định: True)

    Returns
    -------
    str
        Văn bản đã tiền xử lý.
    """
    cfg = config or default_preprocessing_config

    # Cập nhật kwargs cho các steps dựa trên config
    for step in PREPROCESSING_STEPS:
        if step.name == "normalize":
            step.kwargs["lower"] = cfg.normalize_lower
            step.kwargs["strip_spaces"] = cfg.normalize_strip_spaces
        elif step.name == "tokenize":
            step.kwargs["tokenizer"] = cfg.tokenizer
        elif step.name == "filter_stopwords":
            step.kwargs["stopwords"] = cfg.stopwords

    # Chạy pipeline
    result, report = BasePipeline.run_steps(text, cfg, PREPROCESSING_STEPS)

    # Xử lý kết quả dựa trên return_tokens
    if isinstance(result, list):
        if cfg.return_tokens:
            return result  # Trả về list tokens
        result_str: str = " ".join(result)
    else:
        result_str = result if isinstance(result, str) else str(result)

    # Tự động lưu meta
    if save_meta:
        path_cfg = default_path_config
        meta_dir = Path(path_cfg.project_root) / path_cfg.get_meta_dir(
            dataset_name, version
        )
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "preprocessing_params.json"
        meta_data = {
            "steps_report": report,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

    return result_str
