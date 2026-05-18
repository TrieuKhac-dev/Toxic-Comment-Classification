"""
model_saver.py

Các hàm nhỏ, mỗi hàm chỉ làm 1 việc (Single Responsibility).
Dùng để lưu model sau khi train, không quan tâm framework hay kiến trúc.
Người dùng tự chọn hàm phù hợp với framework của họ.

Cách dùng:
    from src.serving import model_saver as ms

    # 1. Tạo thư mục
    exp_dir = ms.create_experiment_dir(experiment_name="lightgbm_v3")

    # 2. Lưu các file
    ms.save_joblib(model, "model.pkl", exp_dir)
    ms.save_joblib(vectorizer, "vectorizer.pkl", exp_dir)

    # 3. Lưu config
    ms.save_config({"threshold": 0.45, "metrics": {"f1": 0.92}}, exp_dir)

    # 4. Promote lên production
    ms.promote_to_production(exp_dir)
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any


def create_experiment_dir(
    base_dir: str = "models",
    experiment_name: str | None = None,
) -> str:
    """
    Tạo thư mục experiments/<tên> và trả về đường dẫn.

    Parameters
    ----------
    base_dir : str
        Thư mục gốc chứa models (mặc định: 'models' - local project root).
    experiment_name : str | None
        Tên experiment. Nếu None, tự sinh theo timestamp.

    Returns
    -------
    str
        Đường dẫn thư mục experiment đã tạo.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name or f"{timestamp}_experiment"
    exp_dir = os.path.join(base_dir, "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"  📁 Created: {exp_dir}")
    return exp_dir


def save_config(config: dict[str, Any], exp_dir: str) -> str:
    """
    Ghi config.json vào thư mục experiment.

    Parameters
    ----------
    config : dict
        Cấu hình model (threshold, metrics, features, ...).
    exp_dir : str
        Đường dẫn thư mục experiment.

    Returns
    -------
    str
        Đường dẫn file config.json.
    """
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Config saved: {config_path}")
    return config_path


def save_joblib(obj: Any, filename: str, exp_dir: str) -> str:
    """
    Lưu object bằng joblib (dùng cho sklearn, lightgbm, ...).

    Parameters
    ----------
    obj : Any
        Object cần lưu (model, vectorizer, ...).
    filename : str
        Tên file (vd: 'model.pkl', 'vectorizer.pkl').
    exp_dir : str
        Đường dẫn thư mục experiment.

    Returns
    -------
    str
        Đường dẫn file đã lưu.
    """
    import joblib

    filepath = os.path.join(exp_dir, filename)
    joblib.dump(obj, filepath)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def save_fasttext(model: Any, filename: str, exp_dir: str) -> str:
    """
    Lưu fasttext model.

    Parameters
    ----------
    model : fasttext.FastText
        FastText model đã train.
    filename : str
        Tên file (vd: 'fasttext_model.bin').
    exp_dir : str
        Đường dẫn thư mục experiment.

    Returns
    -------
    str
        Đường dẫn file đã lưu.
    """
    filepath = os.path.join(exp_dir, filename)
    model.save_model(filepath)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def save_torch(model: Any, filename: str, exp_dir: str) -> str:
    """
    Lưu PyTorch model state_dict.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    filename : str
        Tên file (vd: 'pytorch_model.bin').
    exp_dir : str
        Đường dẫn thư mục experiment.

    Returns
    -------
    str
        Đường dẫn file đã lưu.
    """
    import torch

    filepath = os.path.join(exp_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def save_keras(model: Any, filename: str, exp_dir: str) -> str:
    """
    Lưu Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        Keras model.
    filename : str
        Tên file (vd: 'model.h5').
    exp_dir : str
        Đường dẫn thư mục experiment.

    Returns
    -------
    str
        Đường dẫn file đã lưu.
    """
    filepath = os.path.join(exp_dir, filename)
    model.save(filepath)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def promote_to_production(
    exp_dir: str,
    base_dir: str = "models",
) -> str:
    """
    Copy thư mục experiment lên production/.

    Parameters
    ----------
    exp_dir : str
        Đường dẫn thư mục experiment cần promote.
    base_dir : str
        Thư mục gốc chứa models (mặc định: 'models').

    Returns
    -------
    str
        Đường dẫn thư mục production.
    """
    dst = os.path.join(base_dir, "production")

    # Xóa production cũ nếu có
    if os.path.exists(dst):
        shutil.rmtree(dst)

    # Copy toàn bộ thư mục
    shutil.copytree(exp_dir, dst)
    print(f"  ✅ Promoted to production: {dst}")
    return dst
