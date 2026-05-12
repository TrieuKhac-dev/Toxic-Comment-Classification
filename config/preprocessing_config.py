from collections.abc import Callable
from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class PreprocessingConfig(BaseConfig["PreprocessingConfig"]):
    """Cấu hình tiền xử lý văn bản."""

    normalize_lower: bool = True
    normalize_strip_spaces: bool = True

    tokenizer: Callable[[str], list[str]] | None = None
    stopwords: set[str] | None = None

    return_tokens: bool = True  # True: danh sách token của text, False: text


default_preprocessing_config = PreprocessingConfig()
