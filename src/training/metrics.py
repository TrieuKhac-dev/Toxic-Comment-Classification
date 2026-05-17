"""
metrics.py

Các hàm đánh giá mô hình dùng chung cho mọi experiment.
Được trích xuất từ code lặp trong 15 notebooks training.

Usage:
    from src.training.metrics import evaluate_model, run_cross_validation

    # Evaluate on test set
    metrics = evaluate_model(model, X_test_vec, y_test, threshold=0.5)

    # Cross-validation
    cv_results = run_cross_validation(model, X_train_vec, y_train, cv=5)
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_model(
    model: Any,
    X: Any,
    y_true: np.ndarray,
    threshold: float = 0.5,
    print_report: bool = True,
    return_dict: bool = True,
) -> dict[str, Any]:
    """
    Đánh giá mô hình trên tập dữ liệu: predict + in classification report + confusion matrix.

    Parameters
    ----------
    model : Any
        Model đã train (có predict_proba).
    X : Any
        Feature matrix.
    y_true : np.ndarray
        Nhãn thật.
    threshold : float
        Ngưỡng quyết định (mặc định: 0.5).
    print_report : bool
        Có in kết quả ra console không (mặc định: True).
    return_dict : bool
        Có trả về dict metrics không (mặc định: True).

    Returns
    -------
    dict[str, Any]
        Dictionary chứa: accuracy, precision, recall, f1, roc_auc, pr_auc,
        y_pred, y_prob, confusion_matrix, classification_report.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    if print_report:
        print(f"\n===== EVALUATION (threshold={threshold:.4f}) =====")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"PR-AUC   : {pr_auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

    if return_dict:
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": threshold,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "confusion_matrix": cm,
            "classification_report": report,
        }

    return {}


def run_cross_validation(
    model: Any,
    X: Any,
    y: np.ndarray,
    cv: int = 5,
    scoring: dict[str, str] | None = None,
    return_train_score: bool = False,
    n_jobs: int = -1,
    print_results: bool = True,
) -> dict[str, np.ndarray]:
    """
    Chạy cross-validation và in kết quả.

    Parameters
    ----------
    model : Any
        Model cần đánh giá.
    X : Any
        Feature matrix.
    y : np.ndarray
        Nhãn.
    cv : int
        Số folds (mặc định: 5).
    scoring : dict[str, str] | None
        Dict các metrics cần tính.
        Mặc định: accuracy, precision, recall, f1, roc_auc, pr_auc.
    return_train_score : bool
        Có tính train score không (mặc định: False).
    n_jobs : int
        Số job chạy song song (mặc định: -1 = tất cả CPU).
    print_results : bool
        Có in kết quả ra console không (mặc định: True).

    Returns
    -------
    dict[str, np.ndarray]
        Kết quả cross-validation cho từng metric.
    """
    if scoring is None:
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        }

    cv_scores = cross_validate(
        model,
        X,
        y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=return_train_score,
        n_jobs=n_jobs,
    )

    if print_results:
        print(f"\n===== CROSS-VALIDATION ({cv}-fold) =====")
        for metric in cv_scores:
            if metric.startswith("test_"):
                vals = cv_scores[metric]
                print(f"{metric}: {vals.mean():.4f} +/- {vals.std():.4f}")

    return cast(dict[str, np.ndarray], cv_scores)


def print_cv_results_table(cv_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Chuyển kết quả cross-validation thành DataFrame để dễ nhìn.

    Parameters
    ----------
    cv_scores : dict[str, np.ndarray]
        Kết quả từ run_cross_validation.

    Returns
    -------
    pd.DataFrame
        DataFrame với các cột: metric, mean, std, min, max.
    """
    rows = []
    for metric, vals in cv_scores.items():
        if metric.startswith("test_"):
            rows.append(
                {
                    "metric": metric,
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "min": vals.min(),
                    "max": vals.max(),
                }
            )
    return pd.DataFrame(rows).sort_values("metric")
