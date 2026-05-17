"""
visualization.py

Các hàm vẽ biểu đồ đánh giá mô hình dùng chung cho mọi experiment.
Được trích xuất từ code lặp trong 15 notebooks training.

Usage:
    from src.training.visualization import plot_roc_curve, plot_pr_curve, plot_calibration_curve

    fig_roc = plot_roc_curve(y_test, y_test_prob)
    fig_pr = plot_pr_curve(y_test, y_test_prob)
    fig_cal = plot_calibration_curve(y_test, y_test_prob)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figsize: tuple[int, int] = (6, 5),
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """
    Vẽ ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_prob : np.ndarray
        Xác suất dự đoán lớp 1.
    figsize : tuple[int, int]
        Kích thước figure (mặc định: (6, 5)).
    title : str | None
        Tiêu đề biểu đồ.
    save_path : str | None
        Đường dẫn lưu file (nếu có).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figsize: tuple[int, int] = (6, 5),
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """
    Vẽ Precision-Recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_prob : np.ndarray
        Xác suất dự đoán lớp 1.
    figsize : tuple[int, int]
        Kích thước figure (mặc định: (6, 5)).
    title : str | None
        Tiêu đề biểu đồ.
    save_path : str | None
        Đường dẫn lưu file (nếu có).

    Returns
    -------
    matplotlib.figure.Figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
    figsize: tuple[int, int] = (6, 5),
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """
    Vẽ Calibration curve (độ tin cậy của xác suất dự đoán).

    Parameters
    ----------
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_prob : np.ndarray
        Xác suất dự đoán lớp 1.
    n_bins : int
        Số bins (mặc định: 10).
    strategy : str
        Chiến lược chia bins ('uniform' hoặc 'quantile', mặc định: 'quantile').
    figsize : tuple[int, int]
        Kích thước figure (mặc định: (6, 5)).
    title : str | None
        Tiêu đề biểu đồ.
    save_path : str | None
        Đường dẫn lưu file (nếu có).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import brier_score_loss

    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(mean_pred, frac_pos, marker="o", label=f"Brier Score = {brier:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title or "Calibration Curve")
    ax.legend()
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
