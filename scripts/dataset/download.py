"""
download.py

Script CLI để tải dataset từ Google Drive về local.
Gọi hàm download_local() từ src.dataset.loader.

Cách dùng:
    python scripts/dataset/download.py --file-id <FILE_ID> --dest <DEST_PATH>

Ví dụ:
    python scripts/dataset/download.py ^
        --file-id 1ABCxyz123 ^
        --dest dataset/raw/raw_dataset.csv
"""

from __future__ import annotations

import argparse

from src.dataset.loader import download_local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải dataset từ Google Drive về local."
    )
    parser.add_argument(
        "--file-id",
        required=True,
        help="ID của file trên Google Drive (vd: '1ABCxyz...').",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Đường dẫn đích để lưu file (vd: 'dataset/raw/raw_dataset.csv').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_local(file_id=args.file_id, dest=args.dest)


if __name__ == "__main__":
    main()
