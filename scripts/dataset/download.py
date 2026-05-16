"""
download.py

Script CLI để tải dataset từ Google Drive về local.
Gọi hàm download_local() từ src.dataset.loader.

Cách dùng:
    python scripts/dataset/download.py --file-id <FILE_ID> --dest <DEST_PATH>

Ví dụ:
    python scripts/dataset/download.py ^
        --file-id 1ABCxyz123 ^
        --dest data/custom_dataset/v1/raw/raw_dataset.csv
"""

from __future__ import annotations

import argparse
import os
import sys

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

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
        help="Đường dẫn đích để lưu file (vd: 'data/custom_dataset/v1/raw/raw_dataset.csv').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        download_local(file_id=args.file_id, dest=args.dest)
    except Exception as e:
        print(f"\nLỗi: {e}")
        print("Gợi ý:")
        print(
            "  - Kiểm tra --dest phải bao gồm tên file (vd: data/custom_dataset/v1/raw/raw_dataset.csv), không chỉ thư mục"
        )
        print("  - Kiểm tra file_id có đúng không")
        print("  - Đảm bảo file trên Drive đã được chia sẻ public")
        sys.exit(1)


if __name__ == "__main__":
    main()
