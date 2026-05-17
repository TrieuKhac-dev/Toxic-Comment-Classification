from dataclasses import dataclass, field

from config.base_config import BaseConfig


@dataclass
class ValidationConfig(BaseConfig["ValidationConfig"]):
    """Cấu hình cho các bước kiểm tra chất lượng dữ liệu."""

    # --- Các cột bắt buộc (None = bỏ qua kiểm tra) ---
    required_cols: list[str] | None = field(
        default_factory=lambda: ["comment", "is_toxic"]
    )

    # --- Bật/tắt các bước kiểm tra ---
    enable_column_check: bool = True
    enable_null_check: bool = True
    enable_empty_or_no_letter_check: bool = True
    enable_duplicate_check: bool = True


# Bản mặc định
default_validation_config = ValidationConfig()
