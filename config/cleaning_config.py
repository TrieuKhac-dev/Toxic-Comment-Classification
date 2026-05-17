from dataclasses import dataclass, field

from config.base_config import BaseConfig


@dataclass
class CleaningConfig(BaseConfig["CleaningConfig"]):
    """Cấu hình cho các bước làm sạch dữ liệu."""

    # --- Bật/tắt các bước cleaning ---
    enable_html_removal: bool = True
    enable_url_removal: bool = True
    enable_mention_removal: bool = True
    enable_emoji_removal: bool = True
    enable_special_chars_removal: bool = True
    enable_null_empty_removal: bool = True
    enable_non_text_removal: bool = True
    enable_duplicate_removal: bool = True
    enable_outlier_removal: bool = True

    # --- Tham số cleaning text ---
    keep_punctuation: str = r".,!?"
    max_null_label_ratio: float = 0.05

    # --- Outlier detection (Isolation Forest) ---
    outlier_contamination: float = 0.05  # Tỷ lệ outlier kỳ vọng
    outlier_random_state: int = 42
    outlier_feature_cols: list[str] = field(
        default_factory=lambda: [
            "word_len",
            "char_len",
            "num_exclamation",
            "num_question",
            "num_upper",
            "num_emoji",
        ]
    )


# Bản mặc định
default_cleaning_config = CleaningConfig()
