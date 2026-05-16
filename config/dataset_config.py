# dataset_config.py
from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class DatasetConfig(BaseConfig["DatasetConfig"]):
    """Cấu hình dữ liệu cho một dataset cụ thể (có thể ghi đè)."""

    # --- Tên cột ---
    comment_col: str = "comment"
    label_col: str = "is_toxic"

    # --- Encoding file CSV ---
    encoding: str = "utf-8"

    # --- Metadata ---
    description: str = ""


# Bản mặc định
default_dataset_config = DatasetConfig()
default_dataset_config._export_to_globals()
