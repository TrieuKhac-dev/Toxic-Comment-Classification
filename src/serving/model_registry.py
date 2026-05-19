"""
model_registry.py

Model Registry — Singleton quản lý tất cả model đã load.
- Lazy loading: model chỉ load khi được request lần đầu
- Cache: giữ model trong memory
- Versioning: support model_name/version
- Hỗ trợ "latest" alias (trỏ đến version cao nhất)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, ClassVar

# Regex kiểm tra tên model hợp lệ (chỉ cho phép chữ thường, số, gạch dưới, gạch ngang)
VALID_MODEL_NAME_RE = re.compile(r"^[a-z0-9_-]+$")
VALID_VERSION_RE = re.compile(r"^v\d+$|^latest$")


class ModelInfo:
    """Thông tin metadata của một model version."""

    def __init__(self, model_name: str, version: str, config: dict[str, Any]):
        self.model_name = model_name
        self.version = version
        self.config = config
        self.framework = config.get("model_framework", "unknown")
        self.threshold = config.get("threshold", 0.5)
        self.embedding_type = config.get("embedding", {}).get("type", "unknown")

    @property
    def key(self) -> str:
        return f"{self.model_name}:{self.version}"

    def __repr__(self) -> str:
        return f"ModelInfo({self.key}, framework={self.framework})"


class ModelRegistry:
    """
    Singleton quản lý tất cả model đã load.

    Usage:
        # Lấy predictor (lazy load)
        predictor = ModelRegistry.get_predictor("toxic_lstm", "v1")

        # Lấy version mới nhất
        predictor = ModelRegistry.get_predictor("toxic_lstm", "latest")

        # Liệt kê model có sẵn
        models = ModelRegistry.list_available_models()

        # Reload model
        ModelRegistry.reload_model("toxic_lstm", "v1")
    """

    _instance: ClassVar[ModelRegistry | None] = None
    _predictors: ClassVar[dict[str, Any]] = {}
    _model_infos: ClassVar[dict[str, ModelInfo]] = {}
    _registry_dir: str = ""

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry_dir = cls._resolve_registry_dir()
        return cls._instance

    @staticmethod
    def _resolve_registry_dir() -> str:
        """Xác định thư mục registry (models/registry/)."""
        # Ưu tiên env var
        env_dir = os.getenv("MODEL_REGISTRY_DIR")
        if env_dir:
            return env_dir

        # Mặc định: project_root/models/registry
        project_root = Path(__file__).resolve().parent.parent.parent
        return str(project_root / "models" / "registry")

    @classmethod
    def get_registry_dir(cls) -> str:
        """Lấy đường dẫn thư mục registry."""
        instance = cls()
        return instance._registry_dir

    @classmethod
    def set_registry_dir(cls, path: str) -> None:
        """Set thư mục registry (dùng cho testing)."""
        instance = cls()
        instance._registry_dir = path

    @classmethod
    def _validate_model_name(cls, model_name: str) -> None:
        """Validate tên model."""
        if not VALID_MODEL_NAME_RE.match(model_name):
            raise ValueError(
                f"Invalid model name: '{model_name}'. "
                "Only lowercase letters, numbers, underscores, and hyphens allowed."
            )

    @classmethod
    def _validate_version(cls, version: str) -> None:
        """Validate version string."""
        if not VALID_VERSION_RE.match(version):
            raise ValueError(
                f"Invalid version: '{version}'. "
                "Must be 'v<number>' (e.g., v1, v2) or 'latest'."
            )

    @classmethod
    def _get_model_dir(cls, model_name: str, version: str) -> str:
        """Lấy đường dẫn thư mục của model version."""
        instance = cls()
        return os.path.join(instance._registry_dir, model_name, version)

    @classmethod
    def resolve_latest_version(cls, model_name: str) -> str:
        """
        Tìm version cao nhất của một model (public method).
        Quy tắc: so sánh số sau 'v', ví dụ v2 > v1 > v10.

        Parameters
        ----------
        model_name : str
            Tên model.

        Returns
        -------
        str
            Version cao nhất (vd: 'v2').

        Raises
        ------
        ModelNotFoundError
            Nếu không tìm thấy model hoặc không có version nào.
        """
        instance = cls()
        model_path = os.path.join(instance._registry_dir, model_name)

        if not os.path.isdir(model_path):
            raise ModelNotFoundError(
                f"Model '{model_name}' not found in registry at: {model_path}"
            )

        versions = []
        for entry in os.listdir(model_path):
            full_path = os.path.join(model_path, entry)
            if (
                os.path.isdir(full_path)
                and VALID_VERSION_RE.match(entry)
                and entry != "latest"
            ):
                # Tách số từ "v1", "v2", ... (entry đã được validate bởi VALID_VERSION_RE)
                try:
                    num = int(entry[1:])  # bỏ chữ 'v'
                    versions.append((num, entry))
                except (ValueError, IndexError):
                    # Bỏ qua version không hợp lệ
                    continue

        if not versions:
            raise ModelNotFoundError(
                f"No valid versions found for model '{model_name}'."
            )

        # Sắp xếp giảm dần theo số version
        versions.sort(key=lambda x: x[0], reverse=True)
        return versions[0][1]

    @classmethod
    def _resolve_latest_version(cls, model_name: str) -> str:
        """Alias cho resolve_latest_version (giữ backward compatibility)."""
        return cls.resolve_latest_version(model_name)

    @classmethod
    def _load_config(cls, model_dir: str) -> dict[str, Any]:
        """Đọc config.json từ thư mục model."""
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.json not found at: {config_path}\n"
                "Each model version must have a config.json."
            )
        with open(config_path, encoding="utf-8") as f:
            config_data: dict[str, Any] = json.load(f)
            return config_data

    @classmethod
    def list_available_models(cls) -> list[ModelInfo]:
        """
        Liệt kê tất cả model/version có trong registry.

        Returns
        -------
        list[ModelInfo]
            Danh sách thông tin các model version.
        """
        instance = cls()
        registry_dir = instance._registry_dir

        if not os.path.isdir(registry_dir):
            return []

        models: list[ModelInfo] = []
        for model_name in sorted(os.listdir(registry_dir)):
            model_path = os.path.join(registry_dir, model_name)
            if not os.path.isdir(model_path):
                continue

            for version in sorted(os.listdir(model_path)):
                version_path = os.path.join(model_path, version)
                if not os.path.isdir(version_path):
                    continue
                if not VALID_VERSION_RE.match(version) or version == "latest":
                    continue

                try:
                    config = cls._load_config(version_path)
                    models.append(ModelInfo(model_name, version, config))
                except (FileNotFoundError, json.JSONDecodeError):
                    # Bỏ qua nếu config bị lỗi
                    continue

        return models

    @classmethod
    def get_model_info(cls, model_name: str, version: str = "latest") -> ModelInfo:
        """
        Lấy thông tin model (không load predictor).

        Parameters
        ----------
        model_name : str
            Tên model.
        version : str
            Version (mặc định: 'latest').

        Returns
        -------
        ModelInfo
        """
        cls._validate_model_name(model_name)
        cls._validate_version(version)

        if version == "latest":
            version = cls._resolve_latest_version(model_name)

        key = f"{model_name}:{version}"

        # Kiểm tra cache
        if key in cls._model_infos:
            return cls._model_infos[key]

        # Load từ disk
        model_dir = cls._get_model_dir(model_name, version)
        config = cls._load_config(model_dir)
        info = ModelInfo(model_name, version, config)
        cls._model_infos[key] = info
        return info

    @classmethod
    def get_predictor(
        cls,
        model_name: str,
        version: str = "latest",
        force_reload: bool = False,
    ) -> Any:
        """
        Lấy predictor cho model/version.
        Tự động load nếu chưa có trong cache.

        Parameters
        ----------
        model_name : str
            Tên model.
        version : str
            Version (mặc định: 'latest').
        force_reload : bool
            Force reload dù đã có trong cache.

        Returns
        -------
        BasePredictor
        """
        cls._validate_model_name(model_name)
        cls._validate_version(version)

        if version == "latest":
            version = cls._resolve_latest_version(model_name)

        key = f"{model_name}:{version}"

        # Trả về từ cache nếu có
        if not force_reload and key in cls._predictors:
            return cls._predictors[key]

        # Load model từ disk
        model_dir = cls._get_model_dir(model_name, version)
        config = cls._load_config(model_dir)
        framework = config.get("model_framework", "sklearn")

        # Tạo predictor phù hợp dựa trên framework
        # Lazy import để tránh circular import (predictor.py import model_loader -> model_registry)
        from src.serving.predictor import BasePredictor as _BasePredictor

        predictor = _BasePredictor.create(model_dir, config)

        # Cache
        cls._predictors[key] = predictor
        info = ModelInfo(model_name, version, config)
        cls._model_infos[key] = info

        print(f"  ✅ Loaded: {key} ({framework})")
        return predictor

    @classmethod
    def reload_model(cls, model_name: str, version: str = "latest") -> Any:
        """
        Reload model (bỏ cache cũ, load lại từ disk).

        Parameters
        ----------
        model_name : str
            Tên model.
        version : str
            Version (mặc định: 'latest').

        Returns
        -------
        BasePredictor
        """
        if version == "latest":
            version = cls._resolve_latest_version(model_name)

        key = f"{model_name}:{version}"

        # Xóa cache cũ
        if key in cls._predictors:
            del cls._predictors[key]
        if key in cls._model_infos:
            del cls._model_infos[key]

        return cls.get_predictor(model_name, version, force_reload=True)

    @classmethod
    def unload_model(cls, model_name: str, version: str | None = None) -> None:
        """
        Giải phóng model khỏi cache.

        Parameters
        ----------
        model_name : str
            Tên model.
        version : str | None
            Version cụ thể. Nếu None, unload tất cả version của model.
        """
        if version:
            key = f"{model_name}:{version}"
            cls._predictors.pop(key, None)
            cls._model_infos.pop(key, None)
        else:
            # Unload tất cả version của model
            keys_to_delete = [
                k for k in cls._predictors if k.startswith(f"{model_name}:")
            ]
            for k in keys_to_delete:
                cls._predictors.pop(k, None)
                cls._model_infos.pop(k, None)

    @classmethod
    def clear_cache(cls) -> None:
        """Xóa toàn bộ cache (giải phóng memory)."""
        cls._predictors.clear()
        cls._model_infos.clear()

    @classmethod
    def get_loaded_models(cls) -> list[str]:
        """Liệt kê các model đang được load trong memory."""
        return sorted(cls._predictors.keys())

    @classmethod
    def reset(cls) -> None:
        """
        Reset toàn bộ registry — xóa cache và instance.
        Dùng cho testing để đảm bảo mỗi test bắt đầu với trạng thái sạch.
        """
        cls._predictors.clear()
        cls._model_infos.clear()
        cls._instance = None


class ModelNotFoundError(Exception):
    """Exception khi không tìm thấy model trong registry."""

    pass
