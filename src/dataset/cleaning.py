import html
import re
import unicodedata
from typing import Any

import emoji
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def normalize_text(text: str, lower: bool = True, strip_spaces: bool = True) -> str:
    """Chuẩn hóa Unicode NFC, lowercase, loại bỏ khoảng trắng thừa."""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    if lower:
        text = text.lower()
    if strip_spaces:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
    return text


def remove_html_and_entities(text: str) -> str:
    """Xóa thẻ HTML và giải mã HTML entities."""
    text = re.sub(r"<.*?>", "", text)
    text = html.unescape(text)
    return text


def remove_urls(text: str) -> str:
    """Xóa URL (http, https, ftp, www)."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_mentions(text: str) -> str:
    """Xóa mentions (@username)."""
    return re.sub(r"@\w+", "", text)


def remove_emoji(text: str) -> str:
    """Xóa emoji."""
    return str(emoji.replace_emoji(text, replace=""))


def remove_special_chars(text: str, keep_punctuation: str = r".,!?") -> str:
    """Chỉ giữ chữ, số, khoảng trắng và các dấu câu trong keep."""
    keep_pattern = re.escape(keep_punctuation)
    pattern = r"[^\w\s" + keep_pattern + r"]"
    text = re.sub(pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str, stopwords: list[str]) -> str:
    """Loại bỏ stopwords (dùng split đơn giản)."""
    if not text:
        return ""
    tokens = text.split()
    filtered = [t for t in tokens if t not in stopwords]
    return " ".join(filtered)


def remove_null_or_empty(
    df: pd.DataFrame,
    comment_col: str = "comment",
    label_col: str = "is_toxic",
    max_null_label_ratio: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Xóa dòng nếu comment rỗng hoặc nhãn bị null (nếu tỷ lệ null cho phép).
    """
    report = {
        "empty_comment_removed": 0,
        "null_label_removed": 0,
        "null_label_ratio_original": None,
        "final_rows": 0,
    }
    # Xóa comment rỗng
    mask_not_empty = df[comment_col].astype(str).apply(lambda x: len(x.strip()) > 0)
    removed_empty = (~mask_not_empty).sum()
    report["empty_comment_removed"] = removed_empty
    df_temp = df[mask_not_empty].copy()

    # Xóa nhãn null
    null_label_mask = df_temp[label_col].isna()
    null_ratio = null_label_mask.mean()
    report["null_label_ratio_original"] = null_ratio
    if null_ratio > max_null_label_ratio:
        raise ValueError(
            f"Tỷ lệ null trong nhãn ({null_ratio:.2%}) vượt quá ngưỡng {max_null_label_ratio:.0%}"
        )
    removed_null = null_label_mask.sum()
    report["null_label_removed"] = removed_null
    df_temp = df_temp[~null_label_mask].copy()
    report["final_rows"] = len(df_temp)
    return df_temp, report


def remove_non_text_comments(
    df: pd.DataFrame, comment_col: str = "comment"
) -> tuple[pd.DataFrame, dict]:
    """
    Xóa những comment không chứa chữ cái (toàn số, ký tự đặc biệt hoặc emoji).
    """
    report = {"non_text_removed": 0, "final_rows": 0}
    # Regex kiểm tra có ít nhất một ký tự chữ cái Unicode
    has_letter = (
        df[comment_col].astype(str).apply(lambda x: bool(re.search(r"\p{L}", x)))
    )
    removed_non_text = (~has_letter).sum()
    report["non_text_removed"] = removed_non_text
    df_cleaned = df[has_letter].copy()
    report["final_rows"] = len(df_cleaned)
    return df_cleaned, report


def remove_duplicate_comments(
    df: pd.DataFrame, comment_col: str = "comment", label_col: str = "is_toxic"
) -> tuple[pd.DataFrame, dict]:
    """
    Xóa duplicate:
    - Cùng nhãn: giữ lại một.
    - Khác nhãn: xóa tất cả các bản sao.
    """
    report = {
        "duplicate_comment_groups": 0,
        "removed_rows_different_labels": 0,
        "removed_rows_duplicate_same_label": 0,
        "kept_rows": 0,
    }
    keep_mask = []
    grouped = df.groupby(comment_col)
    total_removed_different = 0
    total_removed_same = 0
    for _, group in grouped:
        if len(group) == 1:
            keep_mask.append(True)
            continue
        labels = group[label_col].unique()
        if len(labels) > 1:
            keep_mask.extend([False] * len(group))
            total_removed_different += len(group)
        else:
            keep_mask.append(True)  # giữ dòng đầu
            keep_mask.extend([False] * (len(group) - 1))
            total_removed_same += len(group) - 1
    df_cleaned = df[keep_mask].copy()
    report["removed_rows_different_labels"] = total_removed_different
    report["removed_rows_duplicate_same_label"] = total_removed_same
    report["duplicate_comment_groups"] = (
        len(grouped) - df_cleaned[comment_col].nunique()
    )
    report["kept_rows"] = len(df_cleaned)
    return df_cleaned, report


def remove_outliers(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
    label_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Xóa outlier bằng Isolation Forest dựa trên các đặc trưng số.

    Parameters:
        df: DataFrame đầu vào.
        feature_cols: Danh sách cột số dùng để phát hiện outlier.
                      Nếu None, tự động lấy tất cả cột numeric (trừ label_col nếu có).
        contamination: Tỷ lệ outlier kỳ vọng (0.0-0.5).
        random_state: Hạt giống ngẫu nhiên.
        label_col: Tên cột nhãn (nếu có) – chỉ dùng để thống kê, không ảnh hưởng phát hiện.

    Returns:
        df_cleaned: DataFrame đã loại bỏ outlier.
        report: dict với các thông tin số lượng outlier, tỷ lệ theo lớp (nếu có label_col).
    """  # noqa: RUF002
    if feature_cols is None:
        # Mặc định lấy tất cả cột numeric, trừ cột nhãn
        exclude = [label_col] if label_col else []
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c not in exclude]
        if not feature_cols:
            raise ValueError(
                "Không tìm thấy cột số nào để phát hiện outlier. Hãy chỉ định feature_cols."
            )

    X = df[feature_cols].fillna(0).values
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso_forest.fit_predict(X)
    outlier_mask = preds == -1  # True là outlier

    report = {
        "total_rows": len(df),
        "outlier_count": outlier_mask.sum(),
        "outlier_ratio": outlier_mask.mean(),
        "outlier_removed_by_label": None,
        "feature_cols_used": feature_cols,
        "contamination_used": contamination,
    }

    if label_col and label_col in df.columns:
        by_label = df.loc[outlier_mask, label_col].value_counts().to_dict()
        report["outlier_removed_by_label"] = by_label

    df_cleaned = df[~outlier_mask].copy()
    return df_cleaned, report
