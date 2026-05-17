"""
split.py

Các hàm chia dữ liệu (train/val/test split) cho bài toán phân loại bình luận.
Hỗ trợ stratified split, indices-based split, và các tỷ lệ chia khác nhau.

Các hàm ở đây là thuần tuý, không phụ thuộc config, chỉ dùng tham số trực tiếp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_indices(
    n_samples: int,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hàm core: chia indices thành 3 tập train, validation, test.

    Dùng np.arange(n_samples) -> train_test_split 2 lần để được 3 tập.
    Kết quả trả về là indices, có thể dùng df.iloc[idx] để lấy dữ liệu.

    Parameters
    ----------
    n_samples : int
        Tổng số mẫu.
    y : np.ndarray
        Nhãn (labels) dùng cho stratified split.
    train_ratio : float
        Tỷ lệ tập train (mặc định: 0.70).
    val_ratio : float
        Tỷ lệ tập validation (mặc định: 0.15).
    test_ratio : float
        Tỷ lệ tập test (mặc định: 0.15).
    stratify : bool
        Có stratified split hay không (mặc định: True).
    random_state : int
        Random state cho tái lập kết quả (mặc định: 42).
    shuffle : bool
        Có shuffle dữ liệu trước khi split hay không (mặc định: True).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (train_indices, val_indices, test_indices)

    Raises
    ------
    ValueError
        Nếu tổng các tỷ lệ không bằng 1.0 hoặc có tỷ lệ âm.

    Examples
    --------
    >>> train_idx, val_idx, test_idx = split_indices(len(df), y)
    >>> train_df = df.iloc[train_idx]
    >>> val_df = df.iloc[val_idx]
    >>> test_df = df.iloc[test_idx]
    >>> train_comments = df["comment"].values[train_idx]  # comment gốc
    """
    # Kiểm tra tỷ lệ
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r < 0 for r in ratios):
        raise ValueError("Tỷ lệ split không được âm.")
    if not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError(f"Tổng các tỷ lệ phải bằng 1.0, hiện tại: {sum(ratios)}")

    indices = np.arange(n_samples)
    stratify_param = y if stratify else None

    # Bước 1: train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        stratify=stratify_param,
        random_state=random_state,
        shuffle=shuffle,
    )

    # Bước 2: val vs test (trên temp)
    test_size_in_temp = test_ratio / temp_ratio if temp_ratio > 0 else 0.5
    stratify_temp = y[temp_idx] if stratify else None
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size_in_temp,
        stratify=stratify_temp,
        random_state=random_state,
        shuffle=shuffle,
    )

    return train_idx, val_idx, test_idx


def split_dataframe(
    df: pd.DataFrame,
    label_col: str = "is_toxic",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
    shuffle: bool = True,
    return_indices: bool = False,
) -> tuple:
    """
    Chia DataFrame thành 3 DataFrame: train, validation, test.

    Wrapper của split_indices: gọi split_indices rồi dùng df.iloc[idx].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đầu vào.
    label_col : str
        Tên cột nhãn dùng cho stratified split (mặc định: "is_toxic").
    train_ratio : float
        Tỷ lệ tập train (mặc định: 0.70).
    val_ratio : float
        Tỷ lệ tập validation (mặc định: 0.15).
    test_ratio : float
        Tỷ lệ tập test (mặc định: 0.15).
    stratify : bool
        Có stratified split hay không (mặc định: True).
    random_state : int
        Random state cho tái lập kết quả (mặc định: 42).
    shuffle : bool
        Có shuffle dữ liệu trước khi split hay không (mặc định: True).
    return_indices : bool
        Nếu True, trả về indices thay vì DataFrame (mặc định: False).

    Returns
    -------
    tuple
        Nếu return_indices=False:
            (train_df, val_df, test_df) - mỗi DataFrame đã reset_index(drop=True)
        Nếu return_indices=True:
            (train_indices, val_indices, test_indices)

    Examples
    --------
    >>> train_df, val_df, test_df = split_dataframe(df)
    >>> train_idx, val_idx, test_idx = split_dataframe(df, return_indices=True)
    """
    if label_col in df.columns:
        y = df[label_col].values
    else:
        y = np.zeros(len(df))

    train_idx, val_idx, test_idx = split_indices(
        n_samples=len(df),
        y=y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=stratify and label_col in df.columns,
        random_state=random_state,
        shuffle=shuffle,
    )

    if return_indices:
        return train_idx, val_idx, test_idx

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df
