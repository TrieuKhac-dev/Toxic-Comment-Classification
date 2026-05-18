"""
model_loader.py

Các hàm nhỏ, mỗi hàm chỉ làm 1 việc (Single Responsibility).
Dùng để load model từ thư mục production/ dựa trên config.json.
Người dùng tự chọn hàm phù hợp với framework của họ.

Cách dùng:
    from src.serving import model_loader as ml

    # 1. Đọc config
    config = ml.load_config("models/production")

    # 2. Load model
    model = ml.load_joblib("models/production", "model.pkl")

    # 3. Load vectorizer (nếu có)
    if config.get("files", {}).get("vectorizer"):
        vectorizer = ml.load_joblib("models/production", "vectorizer.pkl")
"""

from __future__ import annotations

import json
import os
from typing import Any


def load_config(model_dir: str) -> dict[str, Any]:
    """
    Đọc config.json từ thư mục model.

    Parameters
    ----------
    model_dir : str
        Đường dẫn thư mục chứa model (vd: 'models/production').

    Returns
    -------
    dict
        Nội dung config.json.

    Raises
    ------
    FileNotFoundError
        Nếu không tìm thấy config.json.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Không tìm thấy config.json tại: {config_path}\n"
            "Hãy đảm bảo đã train và lưu model trước."
        )

    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    return config


def load_joblib(model_dir: str, filename: str) -> Any:
    """
    Load object từ file joblib/pkl.

    Parameters
    ----------
    model_dir : str
        Đường dẫn thư mục chứa model.
    filename : str
        Tên file (vd: 'model.pkl', 'vectorizer.pkl').

    Returns
    -------
    Any
        Object đã load (model, vectorizer, ...).
    """
    import joblib

    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    obj = joblib.load(filepath)
    print(f"  ✅ Loaded: {filepath}")
    return obj


def load_fasttext(model_dir: str, filename: str) -> Any:
    """
    Load FastText model.

    Parameters
    ----------
    model_dir : str
        Đường dẫn thư mục chứa model.
    filename : str
        Tên file (vd: 'fasttext_model.bin').

    Returns
    -------
    fasttext.FastText
        FastText model.
    """
    import fasttext

    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    model = fasttext.load_model(filepath)
    print(f"  ✅ Loaded: {filepath}")
    return model


def load_torch(model_dir: str, filename: str, model_class: type, **kwargs: Any) -> Any:
    """
    Load PyTorch model từ state_dict.

    Parameters
    ----------
    model_dir : str
        Đường dẫn thư mục chứa model.
    filename : str
        Tên file (vd: 'pytorch_model.bin').
    model_class : type
        Class của model (vd: BertForSequenceClassification).
    **kwargs
        Tham số khởi tạo model.

    Returns
    -------
    torch.nn.Module
        PyTorch model đã load weights.
    """
    import torch

    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    model = model_class(**kwargs)
    model.load_state_dict(torch.load(filepath, map_location="cpu"))
    model.eval()
    print(f"  ✅ Loaded: {filepath}")
    return model


def load_keras(model_dir: str, filename: str) -> Any:
    """
    Load Keras model.

    Parameters
    ----------
    model_dir : str
        Đường dẫn thư mục chứa model.
    filename : str
        Tên file (vd: 'model.h5').

    Returns
    -------
    tf.keras.Model
        Keras model.
    """
    from tensorflow import keras

    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    model = keras.models.load_model(filepath)
    print(f"  ✅ Loaded: {filepath}")
    return model
