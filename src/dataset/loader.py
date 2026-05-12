"""
loader.py

Các hàm tiện ích để tải dataset từ Google Drive và đọc CSV.
Hỗ trợ cả Google Colab (mount Drive) và local (dùng gdown).
Có thể tái sử dụng ở nhiều notebook khác nhau.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def download_from_drive(
    source: str,
    dest: str,
    drive_path: str = "/content/drive",
) -> None:
    """
    [Dành cho Colab] Mount Google Drive và copy file/folder về đúng vị trí.

    Parameters
    ----------
    source : str
        Đường dẫn file/folder trên Drive (ví dụ:
        '/content/drive/MyDrive/CommentClassificationDataset/raw_dataset.csv').
    dest : str
        Đường dẫn đích (ví dụ: 'dataset/raw/raw_dataset.csv').
    drive_path : str
        Đường dẫn mount Drive (mặc định: '/content/drive').
    """
    from google.colab import drive

    drive.mount(drive_path)
    print(f"✅ Đã mount Google Drive vào '{drive_path}'")

    _copy(source, dest)


def download_local(
    file_id: str,
    dest: str,
) -> None:
    """
    [Dành cho local] Tải file từ Google Drive bằng file_id.

    Thử dùng gdown trước, nếu lỗi thì fallback sang requests.

    Parameters
    ----------
    file_id : str
        ID của file trên Google Drive (ví dụ: '1ABCxyz...').
    dest : str
        Đường dẫn đích để lưu file (ví dụ: 'dataset/raw/raw_dataset.csv').
    """
    dst_path = Path(dest)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _download_with_gdown(file_id, dst_path)
    except Exception:
        _download_with_requests(file_id, dst_path)

    print(f"Đã tải file từ Google Drive về '{dest}'")


def _download_with_gdown(file_id: str, dst_path: Path) -> None:
    """Tải file bằng gdown."""
    try:
        import gdown
    except ImportError as err:
        raise ImportError(
            "Thiếu thư viện 'gdown'. Vui lòng cài bằng: pip install gdown"
        ) from err

    url = f"https://drive.google.com/uc?id={file_id}"
    output = gdown.download(url, str(dst_path), quiet=False, use_cookies=False)
    if output is None:
        raise RuntimeError(f"gdown không thể tải file ID '{file_id}'.")


def _download_with_requests(file_id: str, dst_path: Path) -> None:
    """Tải file bằng requests (fallback khi gdown lỗi)."""
    import re

    import requests

    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = session.get(url, stream=True)
    response.raise_for_status()

    # Tìm confirmation token nếu có warning page
    confirm_token = None
    for line in response.text.split("\n"):
        match = re.search(r"confirm=([0-9A-Za-z_-]+)", line)
        if match:
            confirm_token = match.group(1)
            break

    if confirm_token:
        url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
        response = session.get(url, stream=True)
        response.raise_for_status()

    with open(dst_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def _copy(source: str, dest: str) -> None:
    """Copy file hoặc folder từ source tới dest."""
    src_path = Path(source)
    dst_path = Path(dest)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {source}")

    if src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        print(f"Đã copy thư mục '{source}' -> '{dest}'")
    else:
        shutil.copy2(src_path, dst_path)
        print(f"Đã copy file '{source}' -> '{dest}'")


def read_csv_with_columns(
    data_path: str,
    comment_col: str = "comment",
    label_col: str = "is_toxic",
    **pd_kwargs: Any,
) -> pd.DataFrame:
    """
    Đọc file CSV và kiểm tra sự tồn tại của các cột bắt buộc.

    Parameters
    ----------
    data_path : str
        Đường dẫn tới file CSV.
    comment_col : str
        Tên cột chứa bình luận.
    label_col : str
        Tên cột chứa nhãn.
    **pd_kwargs
        Các tham số bổ sung truyền cho pd.read_csv (encoding, sep, ...).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {data_path}")

    df = pd.read_csv(path, **pd_kwargs)
    print(f"Đã đọc dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")

    # Kiểm tra cột bắt buộc
    required_cols = [comment_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print("\nLỖI: Các cột sau không tồn tại trong file CSV:")
        for col in missing:
            print(f"  - '{col}'")
        print(f"\nDanh sách cột thực tế: {df.columns.tolist()}")
        print("Vui lòng kiểm tra lại file CSV hoặc cập nhật comment_col/label_col.")
        raise SystemExit("Dừng notebook do tên cột không khớp.")
    else:
        print("Tất cả các cột yêu cầu đều tồn tại.")
        print(f"   - Cột comment: '{comment_col}'")
        print(f"   - Cột label  : '{label_col}'")

    return df
