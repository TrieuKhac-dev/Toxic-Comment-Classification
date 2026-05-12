import re
import unicodedata
from collections.abc import Callable
from typing import cast

from config.preprocessing_config import (
    PreprocessingConfig,
    default_preprocessing_config,
)


def _get_default_tokenizer() -> Callable[[str], list[str]]:
    """Lazy load tokenizer mặc định (underthesea)."""
    try:
        from underthesea import word_tokenize

        return cast(Callable[[str], list[str]], word_tokenize)
    except ImportError as err:
        raise ImportError(
            "Thiếu underthesea. Cài đặt: pip install underthesea, hoặc override tokenizer trong config."
        ) from err


def _get_default_stopwords() -> set[str]:
    """Lazy load stopwords mặc định (stopwordsiso)."""
    try:
        import stopwordsiso

        return set(stopwordsiso.stopwords("vi"))
    except ImportError as err:
        raise ImportError(
            "Thiếu stopwordsiso. Cài đặt: pip install stopwordsiso, hoặc override stopwords trong config."
        ) from err


def get_tokenizer(
    config: PreprocessingConfig | None = None,
) -> Callable[[str], list[str]]:
    cfg = config or default_preprocessing_config
    return cfg.tokenizer or _get_default_tokenizer()


def get_stopwords(config: PreprocessingConfig | None = None) -> set[str]:
    cfg = config or default_preprocessing_config
    return cfg.stopwords or _get_default_stopwords()


def normalize_text(
    text: str,
    lower: bool | None = None,
    strip_spaces: bool | None = None,
    config: PreprocessingConfig | None = None,
) -> str:
    cfg = config or default_preprocessing_config
    lower = cfg.normalize_lower if lower is None else lower
    strip_spaces = cfg.normalize_strip_spaces if strip_spaces is None else strip_spaces

    text = unicodedata.normalize("NFC", str(text))
    if lower:
        text = text.lower()
    if strip_spaces:
        text = re.sub(r"\s+", " ", text.strip())
    return text


def filter_stopwords(
    tokens: list[str],
    stopwords: set[str] | None = None,
    return_tokens: bool | None = None,
    config: PreprocessingConfig | None = None,
) -> list[str] | str:
    cfg = config or default_preprocessing_config
    if stopwords is None:
        stopwords = get_stopwords(cfg)
    return_tokens = return_tokens if return_tokens is not None else cfg.return_tokens

    filtered = [t for t in tokens if t not in stopwords]
    return filtered if return_tokens else " ".join(filtered)
