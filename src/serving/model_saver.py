"""
model_saver.py

Các hàm nhỏ, mỗi hàm chỉ làm 1 việc (Single Responsibility).
Dùng để lưu model sau khi train vào Model Registry.
Hỗ trợ multi-model multi-version.

Cách dùng:
    from src.serving import model_saver as ms

    # 1. Tạo thư mục model version trong registry
    model_dir = ms.create_model_version_dir(
        model_name="toxic_lstm",
        version="v1",
    )

    # 2. Lưu các file
    ms.save_joblib(model, "model.pkl", model_dir)
    ms.save_joblib(vectorizer, "vectorizer.pkl", model_dir)

    # 3. Lưu config
    ms.save_config({
        "model_name": "toxic_lstm",
        "version": "v1",
        "model_framework": "sklearn",
        "threshold": 0.45,
        "embedding": {"type": "tfidf"},
        "preprocessing": {
            "pipeline": [
                {"name": "lowercase", "enabled": True},
                {"name": "remove_url", "enabled": True},
            ]
        },
        "files": {
            "model": "model.pkl",
            "vectorizer": "vectorizer.pkl",
        },
        "metrics": {"f1": 0.92, "accuracy": 0.95},
    }, model_dir)
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any


def get_registry_dir() -> str:
    """
    Lấy thư mục registry.
    Ưu tiên env var MODEL_REGISTRY_DIR, sau đó mặc định models/registry.
    """
    env_dir = os.getenv("MODEL_REGISTRY_DIR")
    if env_dir:
        return env_dir

    # Mặc định: project_root/models/registry
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    return str(project_root / "models" / "registry")


def create_model_version_dir(
    model_name: str,
    version: str | None = None,
    base_dir: str | None = None,
) -> str:
    """
    Tạo thư mục cho model version trong registry.

    Parameters
    ----------
    model_name : str
        Tên model (vd: 'toxic_lstm', 'toxic_bert').
    version : str | None
        Version (vd: 'v1', 'v2'). Nếu None, tự sinh theo timestamp.
    base_dir : str | None
        Thư mục gốc registry. Mặc định: models/registry/.

    Returns
    -------
    str
        Đường dẫn thư mục model version đã tạo.
    """
    registry_dir = base_dir or get_registry_dir()

    if version is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"

    model_dir = os.path.join(registry_dir, model_name, version)
    os.makedirs(model_dir, exist_ok=True)
    print(f"  📁 Created: {model_dir}")
    return model_dir


def save_config(config: dict[str, Any], model_dir: str) -> str:
    """
    Ghi config.json vào thư mục model.

    Parameters
    ----------
    config : dict
        Cấu hình model (threshold, metrics, features, preprocessing, embedding, ...).
    model_dir : str
        Đường dẫn thư mục model version.

    Returns
    -------
    str
        Đường dẫn file config.json.
    """
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Config saved: {config_path}")
    return config_path


def save_joblib(obj: Any, filename: str, model_dir: str) -> str:
    """
    Lưu object bằng joblib (dùng cho sklearn, lightgbm, ...).

    Parameters
    ----------
    obj : Any
        Object cần lưu (model, vectorizer, ...).
    filename : str
        Tên file (vd: 'model.pkl', 'vectorizer.pkl').
    model_dir : str
        Đường dẫn thư mục model version.

    Returns
    -------
    str
        Đường dẫn file đã lưu.
    """
    import joblib

    filepath = os.path.join(model_dir, filename)
    joblib.dump(obj, filepath)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def save_fasttext(model: Any, filename: str, model_dir: str) -> str:
    """
    Lưu fasttext model.

    Parameters
    ----------
    model : fasttext.FastText
        FastText model đã train.
    filename : str
        Tên file (vd: 'fasttext_model.bin').
    model_dir : str
        Đường dẫn thư mục model version.

    Returns
    -------
    str
        Đường dẫn file đã lưu.
    """
    filepath = os.path.join(model_dir, filename)
    model.save_model(filepath)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def save_torch(model: Any, filename: str, model_dir: str) -> str:
    """
    Lưu PyTorch model state_dict.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    filename : str
        Tên file (vd: 'pytorch_model.bin').
    model_dir : str
        Đường dẫn thư mục model version.

    Returns
    -------
    str
        Đường dẫn file đã lưu.
    """
    import torch

    filepath = os.path.join(model_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def save_keras(model: Any, filename: str, model_dir: str) -> str:
    """
    Lưu Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        Keras model.
    filename : str
        Tên file (vd: 'model.h5').
    model_dir : str
        Đường dẫn thư mục model version.

    Returns
    -------
    str
        Đường dẫn file đã lưu.
    """
    filepath = os.path.join(model_dir, filename)
    model.save(filepath)
    print(f"  ✅ Saved: {filepath}")
    return filepath


def save_preprocessing_config(
    preprocessing_config: dict[str, Any],
    model_dir: str,
) -> str:
    """
    Lưu preprocessing config riêng (preprocessing_config.json).

    Parameters
    ----------
    preprocessing_config : dict
        Cấu hình preprocessing pipeline.
    model_dir : str
        Đường dẫn thư mục model version.

    Returns
    -------
    str
        Đường dẫn file preprocessing_config.json.
    """
    config_path = os.path.join(model_dir, "preprocessing_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_config, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Preprocessing config saved: {config_path}")
    return config_path


def save_metrics(metrics: dict[str, float], model_dir: str) -> str:
    """
    Lưu metrics riêng (metrics.json).

    Parameters
    ----------
    metrics : dict
        Dictionary chứa metrics (f1, accuracy, ...).
    model_dir : str
        Đường dẫn thư mục model version.

    Returns
    -------
    str
        Đường dẫn file metrics.json.
    """
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Metrics saved: {metrics_path}")
    return metrics_path


def promote_to_production(
    model_name: str,
    version: str,
    registry_dir: str | None = None,
    output_dir: str | None = None,
) -> str:
    """
    Promote một model version lên production.
    Copy model từ registry vào thư mục production/.

    Parameters
    ----------
    model_name : str
        Tên model.
    version : str
        Version cần promote.
    registry_dir : str | None
        Thư mục registry. Mặc định: models/registry/.
    output_dir : str | None
        Thư mục output production. Mặc định: models/production/<model_name>/.

    Returns
    -------
    str
        Đường dẫn thư mục production.
    """
    reg_dir = registry_dir or get_registry_dir()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    out_dir = output_dir or os.path.join(project_root, "models", "production")

    src = os.path.join(reg_dir, model_name, version)
    dst = os.path.join(out_dir, model_name)

    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"Source model not found: {src}\n"
            f"Make sure model '{model_name}' version '{version}' exists in registry."
        )

    # Xóa production cũ nếu có
    if os.path.exists(dst):
        shutil.rmtree(dst)

    # Copy toàn bộ thư mục
    shutil.copytree(src, dst)
    print(f"  ✅ Promoted to production: {dst}")
    return dst


def list_registry_models(base_dir: str | None = None) -> list[dict[str, str | bool]]:
    """
    Liệt kê tất cả model/version trong registry.

    Parameters
    ----------
    base_dir : str | None
        Thư mục registry. Mặc định: models/registry/.

    Returns
    -------
    list[dict]
        Danh sách các model version.
    """
    registry_dir = base_dir or get_registry_dir()

    if not os.path.isdir(registry_dir):
        return []

    models: list[dict[str, str | bool]] = []
    for model_name in sorted(os.listdir(registry_dir)):
        model_path = os.path.join(registry_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        for version in sorted(os.listdir(model_path)):
            version_path = os.path.join(model_path, version)
            if not os.path.isdir(version_path):
                continue

            # Kiểm tra config.json
            config_path = os.path.join(version_path, "config.json")
            has_config = os.path.exists(config_path)

            models.append(
                {
                    "model_name": model_name,
                    "version": version,
                    "path": version_path,
                    "has_config": bool(has_config),
                }
            )

    return models
