"""
csv.py

Các hàm tiện ích để đọc và kiểm tra file CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


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
