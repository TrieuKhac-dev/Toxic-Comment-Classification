from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class CleaningConfig(BaseConfig["CleaningConfig"]):
    """Cấu hình cho các bước làm sạch dữ liệu."""

    # --- Tham số normalize_text ---
    normalize_lower: bool = True
    normalize_strip_spaces: bool = True

    # --- Tham số khác ---
    keep_punctuation: str = r".,!?"
    max_null_label_ratio: float = 0.05


# Bản mặc định
default_cleaning_config = CleaningConfig()
