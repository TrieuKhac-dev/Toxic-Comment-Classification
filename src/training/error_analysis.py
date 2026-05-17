"""
error_analysis.py

Các hàm phân tích lỗi (FP/FN) dùng chung cho mọi experiment.
Được trích xuất từ code lặp trong 15 notebooks training.

Usage:
    from src.training.error_analysis import show_fp_fn_samples

    show_fp_fn_samples(test_comments, y_test, y_test_pred, y_test_prob, n=10)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def show_fp_fn_samples(
    comments: np.ndarray | pd.Series | list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n: int = 10,
    print_report: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Hiển thị False Positive và False Negative samples.

    Parameters
    ----------
    comments : np.ndarray | pd.Series | list[str]
        Mảng các comment gốc.
    y_true : np.ndarray
        Nhãn thật (0/1).
    y_pred : np.ndarray
        Nhãn dự đoán (0/1).
    y_prob : np.ndarray
        Xác suất dự đoán lớp 1.
    n : int
        Số lượng samples hiển thị mỗi loại (mặc định: 10).
    print_report : bool
        Có in kết quả ra console không (mặc định: True).

    Returns
    -------
    dict[str, pd.DataFrame]
        {
            "false_positives": DataFrame các FP samples,
            "false_negatives": DataFrame các FN samples,
        }
    """
    df = pd.DataFrame(
        {
            "comment": comments,
            "true_label": y_true,
            "pred_label": y_pred,
            "prob_violation": y_prob,
        }
    )

    fp = df[(df["true_label"] == 0) & (df["pred_label"] == 1)].copy()
    fn = df[(df["true_label"] == 1) & (df["pred_label"] == 0)].copy()

    if print_report:
        print(f"\n===== FALSE POSITIVES (Total: {len(fp)}) =====")
        if len(fp) > 0:
            print(
                fp.sort_values("prob_violation", ascending=False)
                .head(n)[["prob_violation", "comment"]]
                .to_string(index=False)
            )
        else:
            print("No false positives found!")

        print(f"\n===== FALSE NEGATIVES (Total: {len(fn)}) =====")
        if len(fn) > 0:
            print(
                fn.sort_values("prob_violation", ascending=True)
                .head(n)[["prob_violation", "comment"]]
                .to_string(index=False)
            )
        else:
            print("No false negatives found!")

    return {
        "false_positives": fp.sort_values("prob_violation", ascending=False).head(n),
        "false_negatives": fn.sort_values("prob_violation", ascending=True).head(n),
    }
