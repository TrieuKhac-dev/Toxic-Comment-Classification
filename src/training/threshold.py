"""
threshold.py

Các hàm tìm threshold tối ưu cho binary classification.
Được trích xuất từ code lặp trong 15 notebooks training.

Usage:
    from src.training.threshold import find_best_threshold, find_best_threshold_cost

    # F1-max threshold
    best_thresh, best_f1 = find_best_threshold(y_val, y_val_prob)

    # Cost-based threshold (FP=1, FN=5)
    best_thresh, best_cost = find_best_threshold_cost(y_val, y_val_prob, cost_fp=1.0, cost_fn=5.0)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Tìm threshold tối ưu dựa trên metric (mặc định: F1).

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_prob : np.ndarray
        Xác suất dự đoán lớp 1.
    metric : str
        Metric để tối ưu ('f1', 'precision', 'recall').
    thresholds : np.ndarray | None
        Mảng threshold để thử. Mặc định: np.linspace(0.05, 0.95, 91).

    Returns
    -------
    tuple[float, float]
        (best_threshold, best_metric_value)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_thresh = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "precision":
            from sklearn.metrics import precision_score

            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            from sklearn.metrics import recall_score

            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


def find_best_threshold_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Tìm threshold tối ưu dựa trên chi phí FP và FN.

    Total cost = FP * cost_fp + FN * cost_fn
    Threshold tối ưu là threshold có total cost thấp nhất.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_prob : np.ndarray
        Xác suất dự đoán lớp 1.
    cost_fp : float
        Chi phí cho False Positive (mặc định: 1.0).
    cost_fn : float
        Chi phí cho False Negative (mặc định: 1.0).
    thresholds : np.ndarray | None
        Mảng threshold để thử. Mặc định: np.linspace(0.05, 0.95, 91).

    Returns
    -------
    tuple[float, float]
        (best_threshold, best_total_cost)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_thresh = 0.5
    best_cost = float("inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        total_cost = fp * cost_fp + fn * cost_fn

        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = t

    return best_thresh, best_cost


def find_best_threshold_with_recall_constraint(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.95,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Tìm threshold tối ưu với ràng buộc recall >= min_recall.
    Trong các threshold thoả mãn recall >= min_recall, chọn threshold có precision cao nhất.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_prob : np.ndarray
        Xác suất dự đoán lớp 1.
    min_recall : float
        Ngưỡng recall tối thiểu (mặc định: 0.95).
    thresholds : np.ndarray | None
        Mảng threshold để thử. Mặc định: np.linspace(0.05, 0.95, 91).

    Returns
    -------
    tuple[float, float]
        (best_threshold, best_precision)
    """
    from sklearn.metrics import precision_score, recall_score

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_thresh = 0.5
    best_precision = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred)
        if rec >= min_recall:
            prec = precision_score(y_true, y_pred, zero_division=0)
            if prec > best_precision:
                best_precision = prec
                best_thresh = t

    return best_thresh, best_precision
