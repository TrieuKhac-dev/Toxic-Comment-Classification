"""
path_config.py

Cấu hình đường dẫn chuẩn cho project, hỗ trợ cả local và Google Drive.
Kế thừa BaseConfig để có thể override linh hoạt.
"""

from dataclasses import dataclass, field
from pathlib import Path

from config.base_config import BaseConfig


@dataclass
class PathConfig(BaseConfig["PathConfig"]):
    """Cấu hình đường dẫn chuẩn cho project."""

    # --- Thư mục gốc ---
    project_root: str = field(
        default_factory=lambda: str(Path(__file__).resolve().parent.parent)
    )

    # --- Dataset paths ---
    raw_data_dir: str = "dataset/raw"
    preprocessed_data_dir: str = "dataset/preprocess"

    # --- Google Drive paths (dùng trong Colab) ---
    drive_mount_path: str = "/content/drive"
    drive_data_dir: str = "/content/drive/MyDrive/CommentClassificationDataset"

    # --- File names ---
    raw_dataset_filename: str = "raw_dataset.csv"

    # --- Derived paths (computed) ---
    @property
    def raw_dataset_path(self) -> str:
        return str(
            Path(self.project_root) / self.raw_data_dir / self.raw_dataset_filename
        )

    @property
    def drive_raw_dataset_path(self) -> str:
        return f"{self.drive_data_dir}/{self.raw_dataset_filename}"

    def get_preprocessed_path(self, filename: str) -> str:
        return str(Path(self.project_root) / self.preprocessed_data_dir / filename)


# Bản mặc định
default_path_config = PathConfig()
