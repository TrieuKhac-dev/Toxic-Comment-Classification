"""
Script phân tích khám phá dữ liệu (EDA) cho bài toán phân loại bình luận.
Tận dụng các hàm từ cleaning.py và preprocessing.py.
"""

from collections import Counter
from typing import Any

import emoji
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk import ngrams
from sklearn.ensemble import IsolationForest
from underthesea import pos_tag
from wordcloud import WordCloud

from src.dataset.cleaning import remove_special_chars
from src.dataset.preprocessing import (
    filter_stopwords,
    get_tokenizer,
    normalize_text,
)


# ----------------------------------------------------------------------
# 1. Phân phối nhãn
# ----------------------------------------------------------------------
def plot_label_distribution(
    df: pd.DataFrame,
    label_col: str,
    figsize: tuple[int, int] = (6, 4),
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Vẽ biểu đồ phân bố nhãn.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(x=label_col, data=df, ax=ax)
    ax.set_title(title or f"Phân bố lớp ({label_col})")
    ax.set_xlabel(label_col)
    ax.set_ylabel("Số lượng")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# 2. Đặc trưng độ dài
# ----------------------------------------------------------------------
def add_length_features(df: pd.DataFrame, comment_col: str) -> pd.DataFrame:
    """
    Thêm cột char_len (số ký tự) và word_len (số từ dùng tokenizer mặc định).
    """

    def count_words(text):
        tokenizer = get_tokenizer()
        return len(tokenizer(str(text)))

    df = df.copy()
    df["char_len"] = df[comment_col].astype(str).apply(len)
    df["word_len"] = df[comment_col].apply(count_words)
    return df


def plot_length_distribution(
    df: pd.DataFrame, figsize: tuple[int, int] = (14, 5), save_path: str | None = None
) -> plt.Figure:
    """
    Vẽ histogram phân bố số ký tự và số từ.
    Cần DataFrame có cột 'char_len' và 'word_len'.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(df["char_len"], bins=50, kde=True, ax=axes[0])
    axes[0].set_title("Phân bố số ký tự")
    sns.histplot(df["word_len"], bins=50, kde=True, ax=axes[1])
    axes[1].set_title("Phân bố số từ")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_length_boxplot_by_label(
    df: pd.DataFrame,
    label_col: str,
    length_col: str = "word_len",
    figsize: tuple[int, int] = (10, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Boxplot so sánh độ dài (word_len hoặc char_len) giữa các lớp.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=label_col, y=length_col, data=df, ax=ax)
    ax.set_title(f"So sánh {length_col} giữa các lớp")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# 3. Đặc trưng dấu câu, chữ hoa, emoji
# ----------------------------------------------------------------------
def add_punctuation_emoji_features(df: pd.DataFrame, comment_col: str) -> pd.DataFrame:
    """
    Thêm các cột: num_exclamation, num_question, num_upper, num_emoji.
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
# 4. Phát hiện outlier bằng Isolation Forest
# ----------------------------------------------------------------------
def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    random_state: int = 42,
    label_col: str | None = None,
) -> tuple[pd.Series, plt.Figure]:
    """
    Phát hiện outlier bằng Isolation Forest.

    Returns:
        outlier_mask: Series boolean (True nếu là outlier)
        fig: Figure biểu đồ phân bố outlier theo lớp (nếu có label_col)
    """
    X = df[feature_cols].fillna(0)
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    preds = iso_forest.fit_predict(X)
    outlier_mask = pd.Series(preds == -1, index=df.index)

    fig, ax = plt.subplots(figsize=(8, 5))
    if label_col and label_col in df.columns:
        outlier_counts = df.loc[outlier_mask, label_col].value_counts().sort_index()
        ax.bar(outlier_counts.index.astype(str), outlier_counts.values)
        ax.set_title(f"Phân bố outlier theo lớp (contamination={contamination})")
        ax.set_xlabel("Lớp")
        ax.set_ylabel("Số outlier")
    else:
        ax.text(
            0.5,
            0.5,
            "Không có cột nhãn hoặc không vẽ được",
            ha="center",
            transform=ax.transAxes,
        )
        ax.set_title("Outlier detection")
    return outlier_mask, fig


# ----------------------------------------------------------------------
# 5. Tiền xử lý văn bản (token, stopwords)
# ----------------------------------------------------------------------
def preprocess_text_for_eda(
    text: str,
    stopwords: set[str] | None = None,
    keep_punctuation: str = "",
) -> list[str]:
    """
    Chuẩn hóa, loại bỏ dấu câu, tokenize bằng tokenizer mặc định, loại stopwords.
    Nếu stopwords=None, dùng stopwords mặc định từ preprocessing.
    """
    text = normalize_text(text, lower=True, strip_spaces=True)
    text = remove_special_chars(text, keep_punctuation=keep_punctuation)
    tokenizer = get_tokenizer()
    tokens = tokenizer(text)
    result = filter_stopwords(tokens, stopwords=stopwords, return_tokens=True)
    if isinstance(result, list):
        return result
    return list(result) if isinstance(result, str) else []


def add_tokens_column(
    df: pd.DataFrame,
    comment_col: str,
    stopwords: set[str] | None = None,
    keep_punctuation: str = "",
) -> pd.DataFrame:
    """Thêm cột 'tokens' chứa list token đã xử lý."""
    df = df.copy()
    df["tokens"] = df[comment_col].apply(
        lambda x: preprocess_text_for_eda(x, stopwords, keep_punctuation)
    )
    return df


# ----------------------------------------------------------------------
# 6. Type-Token Ratio (TTR)
# ----------------------------------------------------------------------
def calculate_ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def add_ttr_column(df: pd.DataFrame, tokens_col: str = "tokens") -> pd.DataFrame:
    df = df.copy()
    df["ttr"] = df[tokens_col].apply(calculate_ttr)
    return df


def plot_ttr_boxplot_by_label(
    df: pd.DataFrame,
    label_col: str,
    tokens_col: str = "tokens",
    figsize: tuple[int, int] = (8, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Boxplot so sánh TTR giữa các lớp.
    """
    df_temp = add_ttr_column(df, tokens_col)
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=label_col, y="ttr", data=df_temp, ax=ax)
    ax.set_title("So sánh Type-Token Ratio (TTR) giữa các lớp")
    ax.set_xlabel(label_col)
    ax.set_ylabel("TTR")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# 7. So sánh POS tags giữa các lớp
# ----------------------------------------------------------------------
def compare_pos_tags(
    df: pd.DataFrame,
    label_col: str,
    tokens_col: str = "tokens",
    sample_size: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    So sánh tần suất POS tags giữa các lớp.
    Trả về DataFrame so sánh và Figure biểu đồ top chênh lệch.
    """
    tag_names = {
        "N": "Danh từ chung",
        "Nc": "Danh từ chỉ người",
        "Np": "Danh từ riêng",
        "Nu": "Số từ",
        "V": "Động từ",
        "A": "Tính từ",
        "P": "Đại từ",
        "L": "Lượng từ",
        "M": "Từ chỉ số lượng",
        "R": "Phó từ",
        "E": "Thán từ",
        "C": "Liên từ",
        "CH": "Từ chuyên ngành",
        "X": "Từ không phân loại",
        "FW": "Từ nước ngoài",
        "Y": "Từ viết tắt",
        "Z": "Ký tự đặc biệt",
        "T": "Trợ từ",
        "I": "Từ cảm thán",
        "B": "Từ khóa",
        "U": "Đơn vị",
        "S": "Từ tình thái",
    }

    # Lấy mẫu mỗi lớp
    classes = df[label_col].unique()
    samples = {}
    for cls in classes:
        cls_df = df[df[label_col] == cls]
        samples[cls] = cls_df.sample(
            min(sample_size, len(cls_df)), random_state=random_state
        )

    def get_pos_counter(sub_df):
        counter: Counter = Counter()
        for tokens in sub_df[tokens_col]:
            if not tokens:
                continue
            text = " ".join(tokens)
            pos_tags = pos_tag(text)
            counter.update([tag for _, tag in pos_tags])
        return counter

    pos_counters = {cls: get_pos_counter(samples[cls]) for cls in classes}
    all_tags = set()
    for cnt in pos_counters.values():
        all_tags.update(cnt.keys())

    comparison = []
    for tag in all_tags:
        tag_display = tag_names.get(tag, tag)
        row = [tag_display]
        for cls in sorted(classes):
            row.append(pos_counters[cls].get(tag, 0))
        # Chênh lệch giữa lớp 1 và 0 (giả sử 1 là vi phạm, 0 là bình thường)
        if len(classes) == 2:
            diff = pos_counters[1].get(tag, 0) - pos_counters[0].get(tag, 0)
            row.append(diff)
        else:
            row.append(0)
        comparison.append(row)

    columns = ["Loại từ"] + [f"Lớp {cls}" for cls in sorted(classes)] + ["Chênh lệch"]
    comp_df = pd.DataFrame(comparison, columns=columns)
    comp_df = comp_df.sort_values("Chênh lệch", ascending=False)

    # Vẽ biểu đồ cho 2 lớp
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if len(classes) == 2:
        pos_diff = (
            comp_df[comp_df["Chênh lệch"] > 0]
            .sort_values("Chênh lệch", ascending=False)
            .head(10)
        )
        neg_diff = (
            comp_df[comp_df["Chênh lệch"] < 0]
            .sort_values("Chênh lệch", ascending=True)
            .head(10)
        )

        if not pos_diff.empty:
            axes[0].barh(pos_diff["Loại từ"], pos_diff["Chênh lệch"], color="salmon")
            axes[0].set_title("Loại từ xuất hiện nhiều hơn ở lớp vi phạm")
            axes[0].set_xlabel("Chênh lệch (Violent - Normal)")
            axes[0].invert_yaxis()
        else:
            axes[0].text(
                0.5, 0.5, "Không có dữ liệu", ha="center", transform=axes[0].transAxes
            )

        if not neg_diff.empty:
            axes[1].barh(
                neg_diff["Loại từ"], -neg_diff["Chênh lệch"], color="lightgreen"
            )
            axes[1].set_title("Loại từ xuất hiện nhiều hơn ở lớp bình thường")
            axes[1].set_xlabel("Chênh lệch (Normal - Violent)")
            axes[1].invert_yaxis()
        else:
            axes[1].text(
                0.5, 0.5, "Không có dữ liệu", ha="center", transform=axes[1].transAxes
            )
    else:
        axes[0].text(
            0.5,
            0.5,
            "Chỉ hỗ trợ 2 lớp cho biểu đồ",
            ha="center",
            transform=axes[0].transAxes,
        )
        axes[1].set_visible(False)

    plt.tight_layout()
    return comp_df, fig


# ----------------------------------------------------------------------
# 8. Từ xuất hiện nhiều nhất theo document frequency
# ----------------------------------------------------------------------
def get_top_tokens_by_document_frequency(
    df: pd.DataFrame, tokens_col: str = "tokens", top_n: int = 100
) -> list[tuple[str, int]]:
    """
    Đếm số document (bình luận) chứa mỗi token, trả về top_n.
    """
    token_doc_count: Counter = Counter()
    for tokens in df[tokens_col]:
        unique_tokens = set(tokens)
        token_doc_count.update(unique_tokens)
    return token_doc_count.most_common(top_n)


# ----------------------------------------------------------------------
# 9. Word Cloud
# ----------------------------------------------------------------------
def plot_wordcloud(
    df: pd.DataFrame,
    tokens_col: str = "tokens",
    max_words: int = 200,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Vẽ word cloud từ tất cả token.
    """
    all_tokens = [token for tokens in df[tokens_col] for token in tokens]
    text = " ".join(all_tokens)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=max_words,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud - Tất cả bình luận (đã loại stopwords)")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# 10. N-grams
# ----------------------------------------------------------------------
def get_top_ngrams(
    df: pd.DataFrame, tokens_col: str = "tokens", n: int = 2, top_n: int = 100
) -> list[tuple[tuple[str, ...], int]]:
    """
    Lấy top n-grams phổ biến nhất từ toàn bộ dữ liệu.
    """
    all_tokens = [token for tokens in df[tokens_col] for token in tokens]
    ngrams_list = list(ngrams(all_tokens, n))
    freq = Counter(ngrams_list)
    return freq.most_common(top_n)


def get_top_ngrams_by_label(
    df: pd.DataFrame,
    label_col: str,
    tokens_col: str = "tokens",
    n: int = 2,
    top_n: int = 100,
) -> dict[Any, list[tuple[tuple[str, ...], int]]]:
    """
    Lấy top n-grams riêng cho từng lớp.
    """
    result = {}
    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        all_tokens = [token for tokens in subset[tokens_col] for token in tokens]
        ngrams_list = list(ngrams(all_tokens, n))
        freq = Counter(ngrams_list)
        result[label] = freq.most_common(top_n)
    return result


# ----------------------------------------------------------------------
# 11. Từ đặc trưng theo tỷ lệ tần suất giữa hai lớp
# ----------------------------------------------------------------------
def get_ratio_features(
    df: pd.DataFrame, label_col: str, tokens_col: str = "tokens"
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Tính tỷ lệ (tần suất trong lớp 1 + 1) / (tần suất trong lớp 0 + 1).
    Giả sử chỉ có 2 lớp (0: normal, 1: violent).
    Trả về:
        - top_violent: 100 từ có tỷ lệ cao nhất (đặc trưng cho lớp vi phạm)
        - top_normal: 100 từ có tỷ lệ thấp nhất (đặc trưng cho lớp bình thường)
    """
    if len(df[label_col].unique()) != 2:
        raise ValueError("Hàm này chỉ hỗ trợ 2 lớp.")

    violent_tokens = [
        token for tokens in df[df[label_col] == 1][tokens_col] for token in tokens
    ]
    normal_tokens = [
        token for tokens in df[df[label_col] == 0][tokens_col] for token in tokens
    ]

    freq_v = Counter(violent_tokens)
    freq_n = Counter(normal_tokens)

    all_words = set(freq_v.keys()) | set(freq_n.keys())
    ratios = {}
    for word in all_words:
        v = freq_v.get(word, 0) + 1
        n = freq_n.get(word, 0) + 1
        ratios[word] = v / n

    sorted_v = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    sorted_n = sorted(ratios.items(), key=lambda x: x[1])
    return sorted_v[:100], sorted_n[:100]
