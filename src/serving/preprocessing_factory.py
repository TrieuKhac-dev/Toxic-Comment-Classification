"""
preprocessing_factory.py

Preprocessing linh hoạt — mỗi model có thể có pipeline preprocessing riêng.
Đọc cấu hình từ preprocessing_config.json hoặc từ config.json của model.

Usage:
    pipeline = PreprocessingPipeline.from_config({
        "pipeline": [
            {"name": "lowercase", "enabled": True},
            {"name": "remove_url", "enabled": True},
            {"name": "remove_html", "enabled": True},
            {"name": "remove_emoji", "enabled": False},
            {"name": "custom_regex", "patterns": [
                {"pattern": "\\d+", "replacement": "<NUM>"}
            ]}
        ]
    })
    cleaned = pipeline.apply("Hello World! Check out https://example.com")
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

from src.dataset.preprocessing import normalize_text

# === Định nghĩa các preprocessing step ===


def step_lowercase(text: str, config: dict[str, Any]) -> str:
    """Chuyển về chữ thường."""
    if config.get("enabled", True):
        return text.lower()
    return text


def step_strip_spaces(text: str, config: dict[str, Any]) -> str:
    """Xóa khoảng trắng thừa."""
    if config.get("enabled", True):
        return re.sub(r"\s+", " ", text).strip()
    return text


def step_remove_url(text: str, config: dict[str, Any]) -> str:
    """Xóa URL."""
    if config.get("enabled", True):
        return re.sub(
            r"https?://\S+|www\.\S+|\S+\.(com|vn|org|net|edu|gov)\S*",
            "",
            text,
        )
    return text


def step_remove_html(text: str, config: dict[str, Any]) -> str:
    """Xóa HTML tags."""
    if config.get("enabled", True):
        return re.sub(r"<[^>]+>", "", text)
    return text


def step_remove_emoji(text: str, config: dict[str, Any]) -> str:
    """Xóa emoji."""
    if config.get("enabled", True):
        # Pattern match emoji
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # Emoticons
            "\U0001f300-\U0001f5ff"  # Symbols & pictographs
            "\U0001f680-\U0001f6ff"  # Transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # Flags
            "\U00002702-\U000027b0"  # Dingbats
            "\U000024c2-\U0001f251"  # Enclosed characters
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)
    return text


def step_remove_punctuation(text: str, config: dict[str, Any]) -> str:
    """Xóa dấu câu."""
    if config.get("enabled", True):
        return re.sub(r"[^\w\s]", "", text)
    return text


def step_remove_numbers(text: str, config: dict[str, Any]) -> str:
    """Xóa số."""
    if config.get("enabled", True):
        return re.sub(r"\d+", "", text)
    return text


def step_replace_numbers(text: str, config: dict[str, Any]) -> str:
    """Thay số bằng token <NUM>."""
    if config.get("enabled", True):
        return re.sub(r"\d+", "<NUM>", text)
    return text


def step_custom_regex(text: str, config: dict[str, Any]) -> str:
    """Áp dụng custom regex patterns từ config."""
    patterns = config.get("patterns", [])
    for p in patterns:
        pattern = p.get("pattern", "")
        replacement = p.get("replacement", "")
        try:
            text = re.sub(pattern, replacement, text)
        except re.error:
            # Bỏ qua pattern lỗi
            pass
    return text


def step_normalize_unicode(text: str, config: dict[str, Any]) -> str:
    """Chuẩn hóa Unicode (dùng normalize_text từ dataset)."""
    if config.get("enabled", True):
        return normalize_text(
            text,
            lower=False,  # Không lower ở đây, đã có step riêng
            strip_spaces=False,
        )
    return text


# === Registry các step có sẵn ===

STEP_REGISTRY: dict[str, Callable[[str, dict[str, Any]], str]] = {
    "lowercase": step_lowercase,
    "strip_spaces": step_strip_spaces,
    "remove_url": step_remove_url,
    "remove_html": step_remove_html,
    "remove_emoji": step_remove_emoji,
    "remove_punctuation": step_remove_punctuation,
    "remove_numbers": step_remove_numbers,
    "replace_numbers": step_replace_numbers,
    "custom_regex": step_custom_regex,
    "normalize_unicode": step_normalize_unicode,
}


class PreprocessingPipeline:
    """
    Pipeline preprocessing có cấu hình riêng cho từng model.

    Usage:
        pipeline = PreprocessingPipeline.from_config(config_dict)
        cleaned = pipeline.apply("Hello World!")
    """

    def __init__(self, steps: list[tuple[str, Callable, dict[str, Any]]]):
        """
        Parameters
        ----------
        steps : list[tuple[str, Callable, dict]]
            Danh sách (tên_step, hàm_step, config_step).
        """
        self.steps = steps

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PreprocessingPipeline:
        """
        Tạo pipeline từ config dict.

        Parameters
        ----------
        config : dict
            Cấu hình preprocessing. Format:
            {
                "pipeline": [
                    {"name": "lowercase", "enabled": True},
                    {"name": "remove_url", "enabled": True},
                    {"name": "custom_regex", "patterns": [...]}
                ]
            }

        Returns
        -------
        PreprocessingPipeline
        """
        pipeline_config = config.get("pipeline", [])
        steps: list[tuple[str, Callable, dict[str, Any]]] = []

        for step_cfg in pipeline_config:
            name = step_cfg.get("name", "")
            if name not in STEP_REGISTRY:
                print(f"  ⚠️  Unknown preprocessing step: '{name}' — skipping")
                continue

            func = STEP_REGISTRY[name]
            steps.append((name, func, step_cfg))

        return cls(steps)

    @classmethod
    def from_model_config(cls, config: dict[str, Any]) -> PreprocessingPipeline:
        """
        Tạo pipeline từ config.json của model.
        Hỗ trợ cả cấu trúc cũ (preprocessing.lower, preprocessing.strip_spaces)
        và cấu trúc mới (preprocessing.pipeline).

        Parameters
        ----------
        config : dict
            Config dict của model.

        Returns
        -------
        PreprocessingPipeline
        """
        preprocess_cfg = config.get("preprocessing", {})

        # Nếu dùng cấu trúc pipeline mới
        if "pipeline" in preprocess_cfg:
            return cls.from_config(preprocess_cfg)

        # Nếu dùng cấu trúc cũ (lower, strip_spaces)
        # Chuyển đổi sang pipeline format
        pipeline_steps = []
        if preprocess_cfg.get("lower", True):
            pipeline_steps.append({"name": "lowercase", "enabled": True})
        if preprocess_cfg.get("strip_spaces", True):
            pipeline_steps.append({"name": "strip_spaces", "enabled": True})

        # Thêm normalize_unicode mặc định
        pipeline_steps.insert(0, {"name": "normalize_unicode", "enabled": True})

        return cls.from_config({"pipeline": pipeline_steps})

    def apply(self, text: str) -> str:
        """
        Áp dụng toàn bộ pipeline preprocessing lên text.

        Parameters
        ----------
        text : str
            Văn bản đầu vào.

        Returns
        -------
        str
            Văn bản đã được preprocessing.
        """
        for name, func, step_cfg in self.steps:
            try:
                text = func(text, step_cfg)
            except Exception as e:
                print(f"  ⚠️  Preprocessing step '{name}' failed: {e}")
                # Tiếp tục với step tiếp theo
                continue
        return text

    def __repr__(self) -> str:
        steps_str = " → ".join(name for name, _, _ in self.steps)
        return f"PreprocessingPipeline([{steps_str}])"


class CustomScriptPreprocessor:
    """
    Preprocessor dùng script custom từ file .py.
    Dành cho các model có preprocessing đặc biệt không có sẵn trong STEP_REGISTRY.

    Script file phải có hàm:
        def preprocess(text: str) -> str:

    Usage:
        preprocessor = CustomScriptPreprocessor("models/registry/my_model/v1/preprocessing.py")
        cleaned = preprocessor.apply("Hello World!")
    """

    def __init__(self, script_path: str):
        """
        Parameters
        ----------
        script_path : str
            Đường dẫn đến file .py chứa hàm preprocess().
        """
        import importlib.util

        self.script_path = script_path

        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Custom preprocessing script not found: {script_path}"
            )

        # Load module từ file path
        spec = importlib.util.spec_from_file_location(
            "custom_preprocessing", script_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load preprocessing script: {script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Kiểm tra hàm preprocess
        if not hasattr(module, "preprocess"):
            raise AttributeError(
                f"Custom preprocessing script must have a 'preprocess(text: str) -> str' function. "
                f"Not found in: {script_path}"
            )

        self._preprocess_fn = module.preprocess
        print(f"  ✅ Loaded custom preprocessing: {script_path}")

    def apply(self, text: str) -> str:
        """
        Áp dụng preprocessing custom lên text.

        Parameters
        ----------
        text : str
            Văn bản đầu vào.

        Returns
        -------
        str
            Văn bản đã được preprocessing.
        """
        result = self._preprocess_fn(text)
        assert isinstance(
            result, str
        ), f"Custom preprocess function must return str, got {type(result)}"
        return result

    def __repr__(self) -> str:
        return f"CustomScriptPreprocessor({self.script_path})"


# === Preprocessing step có thể mở rộng ===


def register_step(name: str, func: Callable[[str, dict[str, Any]], str]) -> None:
    """
    Đăng ký preprocessing step tùy chỉnh.

    Parameters
    ----------
    name : str
        Tên step (dùng trong config).
    func : Callable
        Hàm xử lý: func(text, config) -> str.
    """
    STEP_REGISTRY[name] = func
