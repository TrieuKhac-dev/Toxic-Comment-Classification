from dataclasses import dataclass, field

from config.base_config import BaseConfig


@dataclass
class CleaningConfig(BaseConfig["CleaningConfig"]):
    """Cấu hình cho các bước làm sạch dữ liệu."""

    # --- Tham số cleaning text ---
    keep_punctuation: str = r".,!?"
    max_null_label_ratio: float = 0.05

    # --- Outlier detection (Isolation Forest) ---
    outlier_enabled: bool = True
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
