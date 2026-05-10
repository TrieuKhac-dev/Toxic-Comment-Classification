import pandas as pd


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


def check_duplicates(df: pd.DataFrame, comment_col: str, label_col: str) -> dict:
    """Kiểm tra bình luận trùng lặp."""
    duplicated_mask = df.duplicated(subset=[comment_col], keep=False)
    duplicated_df = df[duplicated_mask]

    distinct_by_class = (
        duplicated_df.drop_duplicates(subset=[comment_col])[label_col]
        .value_counts()
        .rename_axis("class")
    )
    rows_by_class = duplicated_df[label_col].value_counts().rename_axis("class")
    total_rows = len(df)

    mixed = duplicated_df.groupby(comment_col)[label_col].nunique()
    mixed_comments = mixed[mixed > 1]
    num_mixed = len(mixed_comments)
    total_distinct = duplicated_df[comment_col].nunique()
    mixed_ratio = num_mixed / total_distinct if total_distinct > 0 else 0.0

    return {
        "distinct_duplicated_comments_by_class": distinct_by_class,
        "duplicated_rows_by_class": rows_by_class,
        "duplicated_rows_ratio_by_class": rows_by_class / total_rows,
        "total_duplicated_rows": len(duplicated_df),
        "total_duplicated_rows_ratio": len(duplicated_df) / total_rows,
        "mixed_label_comments_count": num_mixed,
        "mixed_label_ratio": mixed_ratio,
    }


def check_column_names(df: pd.DataFrame, required_cols: list[str]) -> dict:
    """Kiểm tra cột bắt buộc có tồn tại hay không."""
    return {col: col in df.columns for col in required_cols}
