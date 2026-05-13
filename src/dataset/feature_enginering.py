"""
feature_engineering.py

Các hàm tạo đặc trưng số từ dữ liệu văn bản bình luận.
Chỉ chứa logic thêm đặc trưng (feature engineering thuần túy),
không chứa code vẽ biểu đồ, EDA, preprocessing hay cleaning.
Các file khác (eda.py, notebooks, scripts) gọi tới để tạo đặc trưng.
"""

import emoji
import pandas as pd

from src.dataset.preprocessing import get_tokenizer


# ----------------------------------------------------------------------
# 1. Đặc trưng độ dài
# ----------------------------------------------------------------------
def add_length_features(df: pd.DataFrame, comment_col: str) -> pd.DataFrame:
    """
    Thêm cột char_len (số ký tự) và word_len (số từ dùng tokenizer mặc định).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa cột comment.
    comment_col : str
        Tên cột chứa bình luận.

    Returns
    -------
    pd.DataFrame
        DataFrame với 2 cột mới: char_len, word_len.
    """

    def count_words(text):
        tokenizer = get_tokenizer()
        return len(tokenizer(str(text)))

    df = df.copy()
    df["char_len"] = df[comment_col].astype(str).apply(len)
    df["word_len"] = df[comment_col].apply(count_words)
    return df


# ----------------------------------------------------------------------
# 2. Đặc trưng dấu câu, chữ hoa, emoji
# ----------------------------------------------------------------------
def add_punctuation_emoji_features(df: pd.DataFrame, comment_col: str) -> pd.DataFrame:
    """
    Thêm các cột: num_exclamation, num_question, num_upper, num_emoji.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa cột comment.
    comment_col : str
        Tên cột chứa bình luận.

    Returns
    -------
    pd.DataFrame
        DataFrame với 4 cột mới.
    """
    df = df.copy()
    df["num_exclamation"] = df[comment_col].str.count("!")
    df["num_question"] = df[comment_col].str.count(r"\?")
    df["num_upper"] = df[comment_col].str.count(r"[A-Z]")
    df["num_emoji"] = df[comment_col].apply(
        lambda x: sum(1 for _ in emoji.emoji_list(str(x)))
    )
    return df


# ----------------------------------------------------------------------
# 3. Thêm cột tokens (tokenization thuần túy, không norm/clean/stopwords)
# ----------------------------------------------------------------------
def add_tokens_column(
    df: pd.DataFrame,
    comment_col: str,
) -> pd.DataFrame:
    """
    Thêm cột 'tokens' chứa list token từ comment (dùng tokenizer mặc định).
    Chỉ tokenize, KHÔNG normalize, KHÔNG remove_special_chars, KHÔNG filter_stopwords.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa cột comment.
    comment_col : str
        Tên cột chứa bình luận.

    Returns
    -------
    pd.DataFrame
        DataFrame với cột 'tokens' mới.
    """
    tokenizer = get_tokenizer()
    df = df.copy()
    df["tokens"] = df[comment_col].astype(str).apply(lambda x: tokenizer(x))
    return df


# ----------------------------------------------------------------------
# 4. Type-Token Ratio (TTR)
# ----------------------------------------------------------------------
def _calculate_ttr(tokens: list[str]) -> float:
    """
    Tính Type-Token Ratio: số từ duy nhất / tổng số từ.

    Parameters
    ----------
    tokens : list[str]
        Danh sách token.

    Returns
    -------
    float
        Giá trị TTR (0.0 nếu tokens rỗng).
    """
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def add_ttr_column(df: pd.DataFrame, tokens_col: str = "tokens") -> pd.DataFrame:
    """
    Thêm cột 'ttr' dựa trên cột tokens.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa cột tokens.
    tokens_col : str
        Tên cột chứa list token.

    Returns
    -------
    pd.DataFrame
        DataFrame với cột 'ttr' mới.
    """
    df = df.copy()
    df["ttr"] = df[tokens_col].apply(_calculate_ttr)
    return df
