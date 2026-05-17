"""
split_config.py

Cấu hình cho việc chia dữ liệu (train/val/test split).
Kế thừa BaseConfig để có thể override linh hoạt.
"""

from dataclasses import dataclass

from config.base_config import BaseConfig


@dataclass
class SplitConfig(BaseConfig["SplitConfig"]):
    """Cấu hình chia dữ liệu train/val/test."""

    # --- Tỷ lệ split ---
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # --- Stratified split ---
    stratify: bool = True

    # --- Random state ---
    random_state: int = 42

    # --- Shuffle ---
    shuffle: bool = True

    # --- Tên file đầu ra ---
    train_filename: str = "train.csv"
    val_filename: str = "val.csv"
    test_filename: str = "test.csv"


# Bản mặc định
default_split_config = SplitConfig()
