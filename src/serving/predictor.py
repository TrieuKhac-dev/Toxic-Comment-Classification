"""
predictor.py

Predict pipeline: preprocess → predict → threshold.
Sử dụng model_loader để load model từ production/.
"""

from __future__ import annotations

import os
from typing import Any

from src.dataset.preprocessing import normalize_text
from src.serving import model_loader as ml


class Predictor:
    """
    Predictor dùng chung cho mọi kiến trúc model.
    Đọc config.json → tự động load đúng loại model.

    Cách dùng:
        predictor = Predictor("models/production")
        result = predictor.predict("mày ngu vãi")
        # {'label': 1, 'probability': 0.9234, 'threshold': 0.45}
    """

    def __init__(self, model_dir: str = "models/production"):
        """
        Parameters
        ----------
        model_dir : str
            Đường dẫn thư mục chứa model (mặc định: 'models/production').
        """
        self.model_dir = model_dir
        self.tokenizer: Any = None

        # 1. Đọc config
        self.config = ml.load_config(model_dir)
        files = self.config.get("files", {})

        # 2. Load model
        model_file = files.get("model", "model.pkl")
        framework = self.config.get("model_framework", "sklearn")

        if framework in ("sklearn", "lightgbm"):
            self.model = ml.load_joblib(model_dir, model_file)
        elif framework == "pytorch":
            # User cần override model_class trước khi dùng
            raise NotImplementedError(
                "PyTorch model cần được khởi tạo với model_class. "
                "Dùng: predictor = Predictor.from_torch(model_dir, BertForSequenceClassification)"
            )
        elif framework == "tensorflow":
            self.model = ml.load_keras(model_dir, model_file)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        # 3. Load vectorizer (nếu có)
        vec_file = files.get("vectorizer")
        if vec_file:
            self.vectorizer = ml.load_joblib(model_dir, vec_file)
        else:
            self.vectorizer = None

        # 4. Load threshold
        self.threshold = self.config.get("threshold", 0.5)

        # 5. Preprocessing config
        self.preprocess_cfg = self.config.get("preprocessing", {})

        print(
            f"  ✅ Predictor ready | framework={framework} | threshold={self.threshold}"
        )

    @classmethod
    def from_torch(
        cls,
        model_dir: str,
        model_class: type,
        **model_kwargs: Any,
    ) -> Predictor:
        """
        Khởi tạo Predictor cho PyTorch model.

        Parameters
        ----------
        model_dir : str
            Đường dẫn thư mục chứa model.
        model_class : type
            Class của model (vd: BertForSequenceClassification).
        **model_kwargs
            Tham số khởi tạo model.

        Returns
        -------
        Predictor
        """
        instance = cls.__new__(cls)
        instance.model_dir = model_dir
        instance.tokenizer = None

        # 1. Đọc config
        instance.config = ml.load_config(model_dir)
        files = instance.config.get("files", {})

        # 2. Load model
        model_file = files.get("model", "pytorch_model.bin")
        instance.model = ml.load_torch(
            model_dir, model_file, model_class, **model_kwargs
        )

        # 3. Load tokenizer (nếu có)
        tokenizer_dir = files.get("tokenizer")
        if tokenizer_dir:
            from transformers import AutoTokenizer

            tokenizer_path = os.path.join(model_dir, tokenizer_dir)
            instance.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            instance.tokenizer = None

        # 4. Load threshold
        instance.threshold = instance.config.get("threshold", 0.5)

        # 5. Preprocessing config
        instance.preprocess_cfg = instance.config.get("preprocessing", {})

        # 6. Không có vectorizer cho torch
        instance.vectorizer = None

        print(f"  ✅ Predictor (PyTorch) ready | threshold={instance.threshold}")
        return instance

    def _preprocess(self, text: str) -> str:
        """Tiền xử lý văn bản dựa trên config."""
        text = normalize_text(
            text,
            lower=self.preprocess_cfg.get("lower", True),
            strip_spaces=self.preprocess_cfg.get("strip_spaces", True),
        )
        return text

    def _extract_features(self, text: str) -> Any:
        """Trích xuất features từ text dựa trên config."""
        features = []

        # TF-IDF features
        if self.vectorizer is not None:
            tfidf_vec = self.vectorizer.transform([text])
            features.append(tfidf_vec)

        # FastText features (nếu có trong config)
        # User cần tự implement nếu dùng FastText
        # Vì FastText cần tokenize + average vectors

        if len(features) == 0:
            raise ValueError(
                "Không có feature extractor nào được cấu hình. "
                "Hãy đảm bảo config.json có vectorizer hoặc embeddings."
            )

        if len(features) == 1:
            return features[0]

        # Ghép nhiều features
        import numpy as np
        from scipy.sparse import hstack, issparse

        if issparse(features[0]):
            return hstack(features)
        return np.hstack(features)

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
        processed = self._preprocess(text)

        # 2. Extract features
        X = self._extract_features(processed)

        # 3. Predict
        proba = self.model.predict_proba(X)[0, 1]
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
        return [self.predict(text) for text in texts]
