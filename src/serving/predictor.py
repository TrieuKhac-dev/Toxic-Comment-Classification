"""
predictor.py

Predict pipeline: preprocess → embed → predict → threshold.
Mỗi model có preprocessing + embedding riêng.
Sử dụng Abstract Base Class để hỗ trợ nhiều loại model.

Usage:
    # Tự động tạo predictor phù hợp dựa trên config
    predictor = BasePredictor.create("models/registry/toxic_lstm/v1", config)

    # Hoặc dùng trực tiếp class cụ thể
    predictor = SklearnPredictor("models/registry/toxic_lstm/v1", config)
    result = predictor.predict("mày ngu vãi")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.serving.embedding_factory import (
    EmbeddingFactory,
)
from src.serving.preprocessing_factory import (
    CustomScriptPreprocessor,
    PreprocessingPipeline,
)


class BasePredictor(ABC):
    """
    Abstract predictor — mỗi model có preprocessing + embedding riêng.
    Factory method create() tự động chọn implementation phù hợp.
    """

    def __init__(self, model_dir: str, config: dict[str, Any]):
        """
        Parameters
        ----------
        model_dir : str
            Đường dẫn thư mục chứa model.
        config : dict
            Config dict của model (từ config.json).
        """
        self.model_dir = model_dir
        self.config = config
        self.threshold = config.get("threshold", 0.5)

        # Khởi tạo preprocessing pipeline riêng cho model này
        self.preprocessor = PreprocessingPipeline.from_model_config(config)

        # Khởi tạo embedding riêng cho model này
        # FastTextPredictor override __init__ để không gọi cái này
        self.embedding = EmbeddingFactory.create(model_dir, config)

        # Load model
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load model từ disk. Mỗi subclass implement khác nhau."""
        pass

    @abstractmethod
    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict probability. Mỗi subclass implement khác nhau.

        Parameters
        ----------
        features : np.ndarray
            Feature vectors.

        Returns
        -------
        np.ndarray
            Probability array, shape (n_samples, n_classes).
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Preprocessing riêng của model.

        Parameters
        ----------
        text : str
            Văn bản đầu vào.

        Returns
        -------
        str
            Văn bản đã preprocessing.
        """
        return self.preprocessor.apply(text)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embedding riêng của model.

        Parameters
        ----------
        texts : list[str]
            Danh sách văn bản.

        Returns
        -------
        np.ndarray
            Embedding vectors.
        """
        return self.embedding.transform(texts)

    def predict(self, text: str) -> dict[str, Any]:
        """
        Dự đoán toxicity của bình luận.

        Parameters
        ----------
        text : str
            Bình luận cần kiểm tra.

        Returns
        -------
        dict
            {'label': int, 'probability': float, 'threshold': float}
        """
        # 1. Preprocess
        processed = self.preprocess(text)

        # 2. Embed
        X = self.embed([processed])

        # 3. Predict
        proba = self._predict_proba(X)[0, 1]
        label = int(proba >= self.threshold)

        return {
            "label": label,
            "probability": round(float(proba), 4),
            "threshold": self.threshold,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Dự đoán cho nhiều bình luận cùng lúc.

        Parameters
        ----------
        texts : list[str]
            Danh sách bình luận.

        Returns
        -------
        list[dict]
            Danh sách kết quả.
        """
        # 1. Preprocess tất cả
        processed = [self.preprocess(t) for t in texts]

        # 2. Embed tất cả cùng lúc
        X = self.embed(processed)

        # 3. Predict tất cả cùng lúc
        probas = self._predict_proba(X)

        results = []
        for i in range(len(texts)):
            proba = probas[i, 1]
            label = int(proba >= self.threshold)
            results.append(
                {
                    "label": label,
                    "probability": round(float(proba), 4),
                    "threshold": self.threshold,
                }
            )

        return results

    @staticmethod
    def create(model_dir: str, config: dict[str, Any]) -> BasePredictor:
        """
        Factory method — tạo predictor phù hợp dựa trên config.

        Parameters
        ----------
        model_dir : str
            Đường dẫn thư mục chứa model.
        config : dict
            Config dict của model.

        Returns
        -------
        BasePredictor
        """
        framework = config.get("model_framework", "sklearn")

        # Kiểm tra nếu có custom preprocessing hoặc custom embedding
        preprocessing_cfg = config.get("preprocessing", {})
        embedding_cfg = config.get("embedding", {})

        has_custom_preprocessing = preprocessing_cfg.get("type") == "custom"
        has_custom_embedding = embedding_cfg.get("type") == "custom"

        if has_custom_preprocessing or has_custom_embedding:
            return CustomPredictor(model_dir, config)

        if framework in ("sklearn", "lightgbm", "xgboost"):
            return SklearnPredictor(model_dir, config)
        elif framework == "pytorch":
            return TorchPredictor(model_dir, config)
        elif framework == "tensorflow":
            return TensorFlowPredictor(model_dir, config)
        elif framework == "fasttext":
            return FastTextPredictor(model_dir, config)
        else:
            raise ValueError(
                f"Unsupported framework: '{framework}'. "
                "Supported: sklearn, lightgbm, xgboost, pytorch, tensorflow, fasttext"
            )


class SklearnPredictor(BasePredictor):
    """
    Predictor cho sklearn / lightgbm / xgboost models.
    Dùng joblib để load model.
    """

    def _load_model(self) -> None:
        """Load model từ file joblib/pkl."""
        from src.serving import model_loader as ml

        files = self.config.get("files", {})
        model_file = files.get("model", "model.pkl")
        self.model = ml.load_joblib(self.model_dir, model_file)
        print(f"  ✅ SklearnPredictor ready | threshold={self.threshold}")

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability dùng sklearn API."""
        return self.model.predict_proba(features)  # type: ignore[no-any-return]


class TorchPredictor(BasePredictor):
    """
    Predictor cho PyTorch models (BERT, LSTM, ...).
    Cần config chỉ rõ model_class và các tham số.
    """

    def _load_model(self) -> None:
        """Load PyTorch model từ state_dict."""
        import torch

        files = self.config.get("files", {})
        model_file = files.get("model", "pytorch_model.bin")

        # Lấy model_class từ config
        model_class_path = self.config.get("model_class", "")
        if not model_class_path:
            raise ValueError(
                "PyTorch model requires 'model_class' in config.json. "
                "Example: 'model_class': 'transformers.BertForSequenceClassification'"
            )

        model_class = self._import_class(model_class_path)
        model_kwargs = self.config.get("model_kwargs", {})

        filepath = os.path.join(self.model_dir, model_file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PyTorch model not found: {filepath}")

        self.model = model_class(**model_kwargs)
        self.model.load_state_dict(torch.load(filepath, map_location="cpu"))
        self.model.eval()

        # Load tokenizer nếu có
        tokenizer_dir = files.get("tokenizer")
        if tokenizer_dir:
            from transformers import AutoTokenizer

            tokenizer_path = os.path.join(self.model_dir, tokenizer_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = None

        print(f"  ✅ TorchPredictor ready | threshold={self.threshold}")

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability dùng PyTorch."""
        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(features).float()
            outputs = self.model(tensor)

            # Xử lý output: có thể là logits hoặc probabilities
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            probabilities = torch.softmax(logits, dim=1).numpy()

        return probabilities  # type: ignore[no-any-return]

    @staticmethod
    def _import_class(class_path: str) -> type:
        """Import class từ string path (vd: 'transformers.BertForSequenceClassification')."""
        import importlib

        parts = class_path.split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]

        module = importlib.import_module(module_path)
        cls: type = getattr(module, class_name)
        return cls


class TensorFlowPredictor(BasePredictor):
    """
    Predictor cho TensorFlow/Keras models.
    """

    def _load_model(self) -> None:
        """Load Keras model."""
        from src.serving import model_loader as ml

        files = self.config.get("files", {})
        model_file = files.get("model", "model.h5")
        self.model = ml.load_keras(self.model_dir, model_file)
        print(f"  ✅ TensorFlowPredictor ready | threshold={self.threshold}")

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability dùng TensorFlow."""
        return self.model.predict(features, verbose=0)  # type: ignore[no-any-return]


class FastTextPredictor(BasePredictor):
    """
    Predictor cho FastText model.
    FastText có built-in sentence vector + classification.
    Override __init__ để không tạo embedding riêng (FastText đã tích hợp sẵn).
    """

    def __init__(self, model_dir: str, config: dict[str, Any]):
        """Khởi tạo FastTextPredictor — không cần embedding riêng."""
        self.model_dir = model_dir
        self.config = config
        self.threshold = config.get("threshold", 0.5)

        # Khởi tạo preprocessing pipeline riêng cho model này
        self.preprocessor = PreprocessingPipeline.from_model_config(config)

        # FastText không cần embedding riêng — gán None
        # Override embed() để báo lỗi rõ ràng nếu ai đó gọi
        self.embedding = None  # type: ignore[assignment]

        # Load model
        self._load_model()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        FastText không cần embedding riêng — embedding được tích hợp sẵn.
        Gọi hàm này sẽ raise error để tránh nhầm lẫn.
        """
        raise NotImplementedError(
            "FastTextPredictor does not use a separate embedding step. "
            "The embedding is built into the FastText model. "
            "Use predict() or predict_batch() directly."
        )

    def _load_model(self) -> None:
        """Load FastText model."""
        from src.serving import model_loader as ml

        files = self.config.get("files", {})
        model_file = files.get("model", "fasttext_model.bin")
        self.model = ml.load_fasttext(self.model_dir, model_file)
        print(f"  ✅ FastTextPredictor ready | threshold={self.threshold}")

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        FastText predict probability.
        Vì FastText đã tích hợp sẵn preprocessing + embedding,
        nên features ở đây là text gốc (không qua embedding).
        """
        # FastText predict trực tiếp từ text
        # features là embedding vectors, nhưng FastText cần text gốc
        # Nên chúng ta override predict() để xử lý riêng
        raise NotImplementedError(
            "FastTextPredictor should use predict() directly, "
            "not _predict_proba(). The embedding is built into FastText."
        )

    def predict(self, text: str) -> dict[str, Any]:
        """
        FastText predict trực tiếp từ text (không cần embedding riêng).
        """
        processed = self.preprocess(text)
        predictions = self.model.predict(processed, k=1)
        label_raw, proba_raw = predictions

        # FastText trả về label dạng "__label__1" hoặc "__label__toxic"
        label_str = label_raw[0]
        proba = float(proba_raw[0])

        # Chuyển label về 0/1
        if "__label__1" in label_str or "toxic" in label_str.lower():
            label = 1
        else:
            label = 0

        return {
            "label": label,
            "probability": round(proba, 4),
            "threshold": self.threshold,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        FastText batch predict — dùng batch API của FastText cho tốc độ tốt hơn.
        """
        # Preprocess tất cả cùng lúc
        processed = [self.preprocess(t) for t in texts]

        # FastText predict batch — trả về (labels, probabilities)
        labels_raw, probas_raw = self.model.predict(processed, k=1)

        results = []
        for i in range(len(texts)):
            label_str = labels_raw[i][0]
            proba = float(probas_raw[i][0])

            # Chuyển label về 0/1
            if "__label__1" in label_str or "toxic" in label_str.lower():
                label = 1
            else:
                label = 0

            results.append(
                {
                    "label": label,
                    "probability": round(proba, 4),
                    "threshold": self.threshold,
                }
            )

        return results


class CustomPredictor(BasePredictor):
    """
    Predictor cho model có preprocessing + embedding custom (script-based).
    Dùng khi config có preprocessing.type = "custom" hoặc embedding.type = "custom".

    Cách hoạt động:
    - Nếu preprocessing.type == "custom": load preprocessing.py từ thư mục model
    - Nếu embedding.type == "custom": load embedding.py từ thư mục model
    - Model được load bằng joblib (sklearn format)
    """

    def __init__(self, model_dir: str, config: dict[str, Any]):
        """Khởi tạo CustomPredictor — có thể dùng custom preprocessing/embedding."""
        self.model_dir = model_dir
        self.config = config
        self.threshold = config.get("threshold", 0.5)

        # Khởi tạo preprocessing — có thể là custom script
        preprocessing_cfg = config.get("preprocessing", {})
        if preprocessing_cfg.get("type") == "custom":
            script_file = preprocessing_cfg.get("script", "preprocessing.py")
            script_path = os.path.join(model_dir, script_file)
            self.preprocessor: Any = CustomScriptPreprocessor(script_path)
        else:
            self.preprocessor = PreprocessingPipeline.from_model_config(config)

        # Khởi tạo embedding — có thể là custom script
        embedding_cfg = config.get("embedding", {})
        if embedding_cfg.get("type") == "custom":
            from src.serving.embedding_factory import CustomScriptEmbedding

            script_file = embedding_cfg.get("script", "embedding.py")
            script_path = os.path.join(model_dir, script_file)
            self.embedding = CustomScriptEmbedding(script_path)
        else:
            self.embedding = EmbeddingFactory.create(model_dir, config)

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load model từ file joblib/pkl."""
        from src.serving import model_loader as ml

        files = self.config.get("files", {})
        model_file = files.get("model", "model.pkl")
        self.model = ml.load_joblib(self.model_dir, model_file)
        print(f"  ✅ CustomPredictor ready | threshold={self.threshold}")

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability dùng sklearn API."""
        return self.model.predict_proba(features)  # type: ignore[no-any-return]


class EnsemblePredictor(BasePredictor):
    """
    Ensemble nhiều predictors.
    Kết hợp kết quả từ nhiều model khác nhau.
    """

    def __init__(
        self,
        predictors: list[BasePredictor],
        weights: list[float] | None = None,
        strategy: str = "weighted_average",
    ):
        """
        Parameters
        ----------
        predictors : list[BasePredictor]
            Danh sách các predictors.
        weights : list[float] | None
            Trọng số cho từng predictor. Mặc định: equal weight.
        strategy : str
            Chiến lược ensemble: 'weighted_average', 'majority_vote', 'max_probability'.
        """
        self.predictors = predictors
        self.weights = weights or [1.0 / len(predictors)] * len(predictors)
        self.strategy = strategy
        self.threshold = 0.5  # Mặc định, có thể override

        # Chuẩn hóa weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        print(
            f"  ✅ EnsemblePredictor ready | {len(predictors)} models | "
            f"strategy={strategy}"
        )

    def _load_model(self) -> None:
        """EnsemblePredictor không load model trực tiếp."""
        pass

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """EnsemblePredictor không dùng _predict_proba trực tiếp."""
        raise NotImplementedError("EnsemblePredictor uses custom predict logic.")

    def predict(self, text: str) -> dict[str, Any]:
        """Ensemble predict — kết hợp kết quả từ nhiều models."""
        all_results = []
        for predictor in self.predictors:
            result = predictor.predict(text)
            all_results.append(result)

        all_probas = [r["probability"] for r in all_results]

        # Kết hợp probabilities
        if self.strategy == "weighted_average":
            proba = sum(w * p for w, p in zip(self.weights, all_probas, strict=False))
        elif self.strategy == "max_probability":
            proba = max(all_probas)
        elif self.strategy == "majority_vote":
            # Mỗi model vote 0/1 dựa trên threshold riêng
            votes = [
                int(p >= predictor.threshold)
                for p, predictor in zip(all_probas, self.predictors, strict=False)
            ]
            proba = sum(votes) / len(votes)
        else:
            proba = sum(w * p for w, p in zip(self.weights, all_probas, strict=False))

        label = int(proba >= self.threshold)

        # Lấy model_name + version từ config của từng predictor (nếu có)
        individual_results = []
        for i, (predictor, result) in enumerate(
            zip(self.predictors, all_results, strict=False)
        ):
            model_name = predictor.config.get("model_name", f"model_{i}")
            version = predictor.config.get("version", "unknown")
            individual_results.append(
                {
                    "model": f"{model_name}/{version}",
                    "probability": round(float(result["probability"]), 4),
                    "label": result["label"],
                }
            )

        return {
            "label": label,
            "probability": round(float(proba), 4),
            "threshold": self.threshold,
            "strategy": self.strategy,
            "individual_results": individual_results,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Ensemble batch predict."""
        return [self.predict(t) for t in texts]
