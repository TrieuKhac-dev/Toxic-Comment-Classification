"""
preprocessing_pipeline.py

Pipeline tiền xử lý văn bản (preprocessing).
Chứa luồng xử lý (pipeline) cho preprocessing, sử dụng các hàm từ src/dataset/preprocessing.
Có thể nhận đối tượng config để cấu hình.
"""

from collections.abc import Callable

from config.preprocessing_config import (
    PreprocessingConfig,
    default_preprocessing_config,
)
from src.dataset.preprocessing import (
    filter_stopwords,
    normalize_text,
)
from src.dataset.preprocessing import (
    get_stopwords as _get_stopwords,
)
from src.dataset.preprocessing import (
    get_tokenizer as _get_tokenizer,
)


def get_tokenizer_from_config(
    config: PreprocessingConfig | None = None,
) -> Callable[[str], list[str]]:
    """Lấy tokenizer từ config hoặc mặc định."""
    cfg = config or default_preprocessing_config
    return _get_tokenizer(cfg.tokenizer)


def get_stopwords_from_config(config: PreprocessingConfig | None = None) -> set[str]:
    """Lấy stopwords từ config hoặc mặc định."""
    cfg = config or default_preprocessing_config
    return _get_stopwords(cfg.stopwords)


def preprocess_text_pipeline(
    text: str,
    config: PreprocessingConfig | None = None,
) -> str:
    """
    Pipeline tiền xử lý text hoàn chỉnh: normalize -> tokenize -> filter_stopwords -> join.

    Parameters
    ----------
    text : str
        Văn bản đầu vào.
    config : PreprocessingConfig | None
        Đối tượng PreprocessingConfig (mặc định: default_preprocessing_config).

    Returns
    -------
    str
        Văn bản đã tiền xử lý.
    """
    cfg = config or default_preprocessing_config

    # Chuẩn hóa text
    text = normalize_text(
        text,
        lower=cfg.normalize_lower,
        strip_spaces=cfg.normalize_strip_spaces,
    )

    # Tokenize
    tokenizer = get_tokenizer_from_config(cfg)
    tokens = tokenizer(text)

    # Lọc stopwords
    stopwords = get_stopwords_from_config(cfg)
    filtered_tokens = filter_stopwords(
        tokens,
        stopwords=stopwords,
        return_tokens=True,
    )

    # Ghép lại thành text
    return " ".join(filtered_tokens)
