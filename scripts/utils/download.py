"""
download.py

Script CLI để tải dữ liệu từ Google Drive về local.

Hỗ trợ 2 chế độ:

1. Tải dataset theo cấu trúc project (dùng path_config):
    python scripts/utils/download.py ^
        --url <folder_url> ^
        --dataset-name custom_dataset ^
        --version v1 ^
        --subfolder raw ^
        --filename raw_dataset.csv

2. Tải file/folder bất kỳ (tự chỉ định đường dẫn):
    python scripts/utils/download.py ^
        --url <folder_url> ^
        --dest-dir some/path ^
        --filename some_file.csv

Nếu có --filename: tải 1 file cụ thể.
Nếu không có --filename: tải toàn bộ folder.
"""

from __future__ import annotations

import argparse
import os
import sys

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.downloader import download


def download_from_config(
    folder_url: str,
    dataset_name: str,
    version: str,
    filename: str | None = None,
    subfolder: str = "raw",
) -> None:
    """
    Tải dữ liệu từ Google Drive, dùng config để lấy đường dẫn đích.

    Parameters
    ----------
    folder_url : str
        URL của folder Google Drive.
    dataset_name : str
        Tên dataset (vd: custom_dataset).
    version : str
        Phiên bản dataset (vd: v1).
    filename : str | None
        Tên file cần tải. Nếu None, tải toàn bộ folder.
    subfolder : str
        Thư mục con (raw, processed, split, meta). Mặc định: raw.
    """
    from config.path_config import default_path_config

    if subfolder == "raw":
        dest_dir = default_path_config.get_raw_dir(dataset_name, version)
    elif subfolder == "processed":
        dest_dir = default_path_config.get_processed_dir(dataset_name, version)
    elif subfolder == "split":
        dest_dir = default_path_config.get_split_dir(dataset_name, version)
    elif subfolder == "meta":
        dest_dir = default_path_config.get_meta_dir(dataset_name, version)
    else:
        dest_dir = (
            f"{default_path_config.get_version_dir(dataset_name, version)}/{subfolder}"
        )

    download(folder_url=folder_url, dest_dir=dest_dir, filename=filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải dữ liệu từ Google Drive về local.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Tải dataset theo cấu trúc project
  python scripts/utils/download.py --url <URL> --dataset-name custom_dataset --version v1 --subfolder raw --filename raw_dataset.csv

  # Tải toàn bộ folder dataset
  python scripts/utils/download.py --url <URL> --dataset-name my_dataset --version v2 --subfolder processed

  # Tải file bất kỳ
  python scripts/utils/download.py --url <URL> --dest-dir some/path --filename some_file.csv
        """,
    )

    # Nhóm 1: Tải dataset (dùng path_config)
    dataset_group = parser.add_argument_group("Dataset mode (dùng path_config)")
    dataset_group.add_argument(
        "--dataset-name",
        help="Tên dataset (vd: custom_dataset). Kết hợp với --version và --subfolder để tạo đường dẫn.",
    )
    dataset_group.add_argument(
        "--version",
        default="v1",
        help="Phiên bản dataset (mặc định: v1).",
    )
    dataset_group.add_argument(
        "--subfolder",
        default="raw",
        help="Thư mục con (vd: raw, processed, split, meta). Mặc định: raw.",
    )

    # Nhóm 2: Tải file/folder bất kỳ
    custom_group = parser.add_argument_group("Custom mode (tự chỉ định đường dẫn)")
    custom_group.add_argument(
        "--dest-dir",
        help="Đường dẫn đích (vd: datasets/custom_dataset/v1/raw). Dùng khi không có --dataset-name.",
    )

    # Chung
    parser.add_argument(
        "--url",
        required=True,
        help="URL của folder Google Drive (vd: 'https://drive.google.com/drive/folders/...').",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Tên file cần tải. Nếu không có, tải toàn bộ folder.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset_name:
        # Chế độ dataset: dùng download_from_config
        download_from_config(
            folder_url=args.url,
            dataset_name=args.dataset_name,
            version=args.version,
            filename=args.filename,
            subfolder=args.subfolder,
        )
    elif args.dest_dir:
        # Chế độ custom
        try:
            download(
                folder_url=args.url,
                dest_dir=args.dest_dir,
                filename=args.filename,
            )
        except Exception as e:
            print(f"\nLỗi: {e}")
            print("Gợi ý:")
            print("  - Kiểm tra URL có đúng không")
            print("  - Đảm bảo folder trên Drive đã được chia sẻ public")
            print("  - Kiểm tra --filename có tồn tại trong folder không")
            sys.exit(1)
    else:
        print("Lỗi: Cần cung cấp --dataset-name hoặc --dest-dir.")
        print(
            "  - Nếu là dataset: --dataset-name <name> [--version v1] [--subfolder raw]"
        )
        print("  - Nếu là file/folder bất kỳ: --dest-dir <path>")
        sys.exit(1)


if __name__ == "__main__":
    main()
