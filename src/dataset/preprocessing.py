"""
preprocessing.py

Các hàm tiền xử lý văn bản (text preprocessing).
Bao gồm: chuẩn hóa (lower, strip), tokenize, loại bỏ stopwords,
và các hàm tiện ích liên quan.
"""

import re
import unicodedata
from collections.abc import Callable
from typing import cast


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
    tokenizer: Callable[[str], list[str]] | None = None,
) -> Callable[[str], list[str]]:
    """
    Lấy tokenizer. Nếu tokenizer được truyền thì dùng, nếu không dùng default.

    Parameters
    ----------
    tokenizer : Callable | None
        Tokenizer tuỳ chỉnh (mặc định: None -> dùng underthesea).

    Returns
    -------
    Callable[[str], list[str]]
        Hàm tokenize.
    """
    if tokenizer is not None:
        return tokenizer
    return _get_default_tokenizer()


def get_stopwords(stopwords: set[str] | None = None) -> set[str]:
    """
    Lấy stopwords. Nếu stopwords được truyền thì dùng, nếu không dùng default.

    Parameters
    ----------
    stopwords : set[str] | None
        Tập stopwords tuỳ chỉnh (mặc định: None -> dùng stopwordsiso).

    Returns
    -------
    set[str]
        Tập stopwords.
    """
    if stopwords is not None:
        return stopwords
    return _get_default_stopwords()


def normalize_text(
    text: str,
    lower: bool = True,
    strip_spaces: bool = True,
) -> str:
    """Chuẩn hóa văn bản: NFC normalize, lowercase, strip spaces."""
    text = unicodedata.normalize("NFC", str(text))
    if lower:
        text = text.lower()
    if strip_spaces:
        text = re.sub(r"\s+", " ", text.strip())
    return text


def filter_stopwords(
    tokens: list[str],
    stopwords: set[str] | None = None,
    return_tokens: bool = True,
) -> list[str] | str:
    """Lọc stopwords khỏi danh sách token."""
    if stopwords is None:
        stopwords = _get_default_stopwords()
    filtered = [t for t in tokens if t not in stopwords]
    return filtered if return_tokens else " ".join(filtered)
