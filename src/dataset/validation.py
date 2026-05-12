import pandas as pd

from src.dataset.preprocessing import normalize_text


def check_null(df: pd.DataFrame, cols: list[str] | None = None) -> dict:
    """Kiểm tra số lượng null/NaN trong DataFrame."""
    if cols is None:
        cols = df.columns.tolist()
    null_counts = df[cols].isnull().sum()
    return {
        "null_counts": null_counts,
        "total_null": int(null_counts.sum()),
        "null_ratio": float(null_counts.sum() / len(df)) if len(df) > 0 else 0.0,
    }


def check_empty_or_no_letter(
    df: pd.DataFrame,
    col: str,
    check_empty: bool = True,
    check_no_letter: bool = True,
    trim: bool = True,
) -> dict:
    """Kiểm tra bình luận rỗng và/hoặc không chứa chữ cái."""
    result: dict = {"total_samples": len(df)}
    texts = df[col].astype(str)
    if trim:
        texts = texts.str.strip()

    if check_empty:
        empty_mask = texts == ""
        result["empty_count"] = int(empty_mask.sum())
        result["empty_ratio"] = (
            float(result["empty_count"]) / len(df) if len(df) > 0 else 0.0
        )

    if check_no_letter:
        no_letter_mask = ~texts.str.contains(r"[A-Za-zÀ-ỹ]", regex=True, na=False)
        result["no_letter_count"] = int(no_letter_mask.sum())
        result["no_letter_ratio"] = (
            float(result["no_letter_count"]) / len(df) if len(df) > 0 else 0.0
        )

    return result


def check_duplicates(
    df: pd.DataFrame,
    comment_col: str,
    label_col: str,
    normalize: bool = True,
    lower: bool = True,
    strip_spaces: bool = True,
    dropna_comments: bool = True,
    ignore_nan_labels_in_mixed_check: bool = True,
) -> dict:
    """
    Phát hiện comment trùng lặp và mixed-label.

    Parameters
    ----------
    df : DataFrame đầu vào
    comment_col : Tên cột comment
    label_col : Tên cột nhãn
    normalize : Chuẩn hóa text (strip, lower) trước khi so sánh
    lower : Lowercase khi normalize
    strip_spaces : Xóa khoảng trắng thừa khi normalize
    dropna_comments : Bỏ qua comment NaN
    ignore_nan_labels_in_mixed_check : NaN label không tính là mixed-label

    Returns
    -------
    dict
        - total_rows: Tổng số dòng sau xử lý
        - total_duplicated_rows: Số dòng bị trùng
        - total_duplicated_rows_ratio: Tỷ lệ dòng trùng
        - distinct_duplicated_comments: Số comment khác nhau bị trùng
        - distinct_duplicated_comments_by_class: Phân bố comment trùng theo label
        - duplicated_rows_by_class: Phân bố dòng trùng theo label
        - duplicated_rows_ratio_by_class: Tỷ lệ dòng trùng theo label
        - mixed_label_comments_count: Số comment có mixed-label
        - mixed_label_ratio: Tỷ lệ mixed-label
        - mixed_label_examples: 5 ví dụ mixed-label đầu tiên
    """
    # Empty dataframe
    if df.empty:
        return {
            "total_rows": 0,
            "total_duplicated_rows": 0,
            "total_duplicated_rows_ratio": 0.0,
            "distinct_duplicated_comments": 0,
            "distinct_duplicated_comments_by_class": {},
            "duplicated_rows_by_class": {},
            "duplicated_rows_ratio_by_class": {},
            "mixed_label_comments_count": 0,
            "mixed_label_ratio": 0.0,
            "mixed_label_examples": [],
        }

    # Copy dataframe
    df_temp = df.copy()

    # Handle NaN comments
    if dropna_comments:
        df_temp = df_temp[df_temp[comment_col].notna()].copy()
    # Convert to string
    df_temp[comment_col] = df_temp[comment_col].astype(str)
    # Remove empty comments
    df_temp = df_temp[df_temp[comment_col].str.strip().ne("")].copy()

    # Normalize text
    if normalize:
        df_temp[comment_col] = df_temp[comment_col].apply(
            lambda x: normalize_text(
                x,
                lower=lower,
                strip_spaces=strip_spaces,
            )
        )

    # Detect duplicates
    duplicated_mask = df_temp.duplicated(
        subset=[comment_col],
        keep=False,
    )
    duplicated_df = df_temp.loc[duplicated_mask].copy()
    total_rows = len(df_temp)

    # No duplicates
    if duplicated_df.empty:
        return {
            "total_rows": int(total_rows),
            "total_duplicated_rows": 0,
            "total_duplicated_rows_ratio": 0.0,
            "distinct_duplicated_comments": 0,
            "distinct_duplicated_comments_by_class": {},
            "duplicated_rows_by_class": {},
            "duplicated_rows_ratio_by_class": {},
            "mixed_label_comments_count": 0,
            "mixed_label_ratio": 0.0,
            "mixed_label_examples": [],
        }

    # Duplicate statistics by class
    distinct_counts = (
        duplicated_df.drop_duplicates(subset=[comment_col])[label_col]
        .value_counts(dropna=False)
        .to_dict()
    )
    rows_counts = duplicated_df[label_col].value_counts(dropna=False).to_dict()
    rows_ratios = {
        str(k): (float(v / total_rows) if total_rows > 0 else 0.0)
        for k, v in rows_counts.items()
    }

    # Mixed-label duplicate detection
    mixed_df = duplicated_df.copy()
    if ignore_nan_labels_in_mixed_check:
        mixed_df = mixed_df[mixed_df[label_col].notna()].copy()
    mixed = mixed_df.groupby(comment_col)[label_col].nunique()
    mixed_comments = mixed[mixed > 1]
    num_mixed = len(mixed_comments)
    total_distinct = int(duplicated_df[comment_col].nunique())
    mixed_ratio = float(num_mixed / total_distinct) if total_distinct > 0 else 0.0

    # Mixed-label examples
    mixed_examples = []
    if num_mixed > 0:
        example_comments = mixed_comments.index[:5]
        for comment in example_comments:
            labels = (
                duplicated_df.loc[
                    duplicated_df[comment_col] == comment,
                    label_col,
                ]
                .drop_duplicates()
                .tolist()
            )
            mixed_examples.append(
                {
                    "comment": comment[:200],
                    "labels": labels,
                }
            )

    # Return
    return {
        "total_rows": int(total_rows),
        "total_duplicated_rows": len(duplicated_df),
        "total_duplicated_rows_ratio": (
            float(len(duplicated_df) / total_rows) if total_rows > 0 else 0.0
        ),
        "distinct_duplicated_comments": int(duplicated_df[comment_col].nunique()),
        "distinct_duplicated_comments_by_class": {
            str(k): int(v) for k, v in distinct_counts.items()
        },
        "duplicated_rows_by_class": {str(k): int(v) for k, v in rows_counts.items()},
        "duplicated_rows_ratio_by_class": rows_ratios,
        "mixed_label_comments_count": num_mixed,
        "mixed_label_ratio": mixed_ratio,
        "mixed_label_examples": mixed_examples,
    }


def check_column_names(df: pd.DataFrame, required_cols: list[str]) -> dict:
    """Kiểm tra cột bắt buộc có tồn tại hay không."""
    return {col: col in df.columns for col in required_cols}
