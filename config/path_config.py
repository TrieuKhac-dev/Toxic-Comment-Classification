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

    # --- Data paths ---
    data_dir: str = "datasets"

    # --- Google Drive paths (dùng trong Colab) ---
    drive_mount_path: str = "/content/drive"
    drive_data_dir: str = "/content/drive/MyDrive/CommentClassificationDataset"

    # --- File names ---
    raw_dataset_filename: str = "raw_dataset.csv"

    # --- Helper methods (dùng chung cho mọi dataset) ---
    def get_dataset_dir(self, dataset_name: str) -> str:
        """Lấy thư mục dataset: datasets/<dataset_name>"""
        return f"{self.data_dir}/{dataset_name}"

    def get_drive_dataset_dir(self, dataset_name: str) -> str:
        """Lấy thư mục dataset trên Drive: drive_data_dir/<dataset_name>"""
        return f"{self.drive_data_dir}/{dataset_name}"

    def get_version_dir(self, dataset_name: str, version: str = "v1") -> str:
        return f"{self.get_dataset_dir(dataset_name)}/{version}"

    def get_raw_dir(self, dataset_name: str, version: str = "v1") -> str:
        return f"{self.get_version_dir(dataset_name, version)}/raw"

    def get_processed_dir(self, dataset_name: str, version: str = "v1") -> str:
        return f"{self.get_version_dir(dataset_name, version)}/processed"

    def get_processed_filename(self, raw_filename: str) -> str:
        """Sinh tên file processed từ tên file raw.
        Ví dụ: raw_dataset.csv -> processed_dataset.csv
                raw_data.csv -> processed_data.csv
                dataset.csv -> processed_dataset.csv
        """
        stem = Path(raw_filename).stem
        # Nếu tên bắt đầu bằng "raw_" thì bỏ "raw_"
        if stem.startswith("raw_"):
            stem = stem[4:]
        return f"processed_{stem}.csv"

    def get_split_dir(self, dataset_name: str, version: str = "v1") -> str:
        return f"{self.get_version_dir(dataset_name, version)}/split"

    def get_meta_dir(self, dataset_name: str, version: str = "v1") -> str:
        return f"{self.get_version_dir(dataset_name, version)}/meta"


# Bản mặc định
default_path_config = PathConfig()
