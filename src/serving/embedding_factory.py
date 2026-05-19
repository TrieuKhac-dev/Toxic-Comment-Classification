"""
embedding_factory.py

Embedding linh hoạt — mỗi model có thể dùng embedding khác nhau.
Hỗ trợ: TF-IDF, FastText, BERT, Ensemble (kết hợp nhiều embedding).

Usage:
    factory = EmbeddingFactory()
    embedding = factory.create(model_config)
    features = embedding.transform(["text 1", "text 2"])
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEmbedding(ABC):
    """Abstract base cho tất cả embedding methods."""

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Chuyển đổi texts thành embedding vectors.

        Parameters
        ----------
        texts : list[str]
            Danh sách văn bản.

        Returns
        -------
        np.ndarray
            Embedding vectors, shape (n_texts, n_features).
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Lấy tên các features (nếu có)."""
        pass


class TFIDFEmbedding(BaseEmbedding):
    """
    TF-IDF embedding dùng vectorizer đã train sẵn (joblib/pkl).
    """

    def __init__(self, vectorizer_path: str):
        """
        Parameters
        ----------
        vectorizer_path : str
            Đường dẫn đến file vectorizer.pkl hoặc .joblib.
        """
        import joblib

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

        self.vectorizer = joblib.load(vectorizer_path)
        print(f"  ✅ Loaded TF-IDF vectorizer: {vectorizer_path}")

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts thành TF-IDF sparse matrix."""
        return self.vectorizer.transform(texts)  # type: ignore[no-any-return]

    def get_feature_names(self) -> list[str]:
        """Lấy danh sách vocabulary terms."""
        try:
            return self.vectorizer.get_feature_names_out().tolist()  # type: ignore[no-any-return]
        except AttributeError:
            return []


class FastTextEmbedding(BaseEmbedding):
    """
    FastText embedding — dùng FastText model để sinh sentence embedding.
    Hỗ trợ: average word vectors hoặc use FastText built-in sentence vector.
    """

    def __init__(
        self,
        model_path: str,
        pooling: str = "mean",
        use_sentence_vector: bool = False,
    ):
        """
        Parameters
        ----------
        model_path : str
            Đường dẫn đến FastText model (.bin).
        pooling : str
            Cách pooling word vectors: 'mean', 'min', 'max'.
        use_sentence_vector : bool
            Nếu True, dùng fasttext.get_sentence_vector() thay vì average word vectors.
        """
        import fasttext

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FastText model not found: {model_path}")

        self.model = fasttext.load_model(model_path)
        self.pooling = pooling
        self.use_sentence_vector = use_sentence_vector
        self.dim = self.model.get_dimension()
        print(f"  ✅ Loaded FastText model: {model_path} (dim={self.dim})")

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts thành FastText sentence embeddings."""
        if self.use_sentence_vector:
            # Dùng built-in sentence vector (nhanh hơn)
            vectors: np.ndarray = np.array(
                [self.model.get_sentence_vector(t) for t in texts]
            )
        else:
            # Average word vectors
            vec_list: list[np.ndarray] = []
            for text in texts:
                words = text.split()
                if not words:
                    vec_list.append(np.zeros(self.dim))
                    continue

                word_vectors = np.array([self.model.get_word_vector(w) for w in words])

                if self.pooling == "mean":
                    vec = word_vectors.mean(axis=0)
                elif self.pooling == "max":
                    vec = word_vectors.max(axis=0)
                elif self.pooling == "min":
                    vec = word_vectors.min(axis=0)
                else:
                    vec = word_vectors.mean(axis=0)

                vec_list.append(vec)

            vectors = np.array(vec_list)

        return vectors

    def get_feature_names(self) -> list[str]:
        """FastText không có feature names cố định."""
        return [f"ft_dim_{i}" for i in range(self.dim)]


class BERTEmbedding(BaseEmbedding):
    """
    BERT embedding — dùng transformer model để sinh embedding.
    Hỗ trợ: CLS token, mean pooling, max pooling.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        pooling: str = "cls",
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        model_name : str
            Tên model trên HuggingFace Hub hoặc đường dẫn local.
        max_length : int
            Độ dài tối đa của input tokens.
        pooling : str
            Cách lấy embedding: 'cls', 'mean', 'max'.
        device : str
            'cpu' hoặc 'cuda'.
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.pooling = pooling
        self.hidden_size = self.model.config.hidden_size

        print(f"  ✅ Loaded BERT model: {model_name} (hidden_size={self.hidden_size})")

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts thành BERT embeddings."""
        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        if self.pooling == "cls":
            # Use [CLS] token embedding
            embeddings: np.ndarray = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif self.pooling == "mean":
            # Mean pooling (bỏ qua padding tokens)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            embeddings_np = (outputs.last_hidden_state * attention_mask).sum(dim=1)
            embeddings_np = embeddings_np / attention_mask.sum(dim=1)
            embeddings = embeddings_np.cpu().numpy()
        elif self.pooling == "max":
            # Max pooling (bỏ qua padding tokens)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            embeddings = outputs.last_hidden_state * attention_mask
            embeddings = embeddings.max(dim=1).values.cpu().numpy()  # type: ignore[call-overload]
        else:
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def get_feature_names(self) -> list[str]:
        """BERT không có feature names cố định."""
        return [f"bert_dim_{i}" for i in range(self.hidden_size)]


class EnsembleEmbedding(BaseEmbedding):
    """
    Ensemble nhiều embedding methods.
    Kết hợp các embedding vectors bằng cách concatenate.
    """

    def __init__(self, embeddings: list[BaseEmbedding]):
        """
        Parameters
        ----------
        embeddings : list[BaseEmbedding]
            Danh sách các embedding methods.
        """
        self.embeddings = embeddings
        print(f"  ✅ Ensemble embedding with {len(embeddings)} methods")

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts bằng tất cả embedding methods và concatenate."""
        from scipy.sparse import hstack, issparse

        all_features = []
        for emb in self.embeddings:
            features = emb.transform(texts)
            all_features.append(features)

        # Kiểm tra nếu có sparse matrix (TF-IDF)
        if any(issparse(f) for f in all_features):
            # Chuyển dense về sparse để hstack
            processed = []
            for f in all_features:
                if issparse(f):
                    processed.append(f)
                else:
                    from scipy.sparse import csr_matrix

                    processed.append(csr_matrix(f))
            return hstack(processed)  # type: ignore[no-any-return]
        else:
            return np.hstack(all_features)

    def get_feature_names(self) -> list[str]:
        """Kết hợp feature names từ tất cả embeddings."""
        names = []
        for emb in self.embeddings:
            names.extend(emb.get_feature_names())
        return names


class CustomScriptEmbedding(BaseEmbedding):
    """
    Embedding dùng script custom từ file .py.
    Dành cho các model có embedding đặc biệt (vd: kết hợp TF-IDF + FastText).

    Script file phải có hàm:
        def transform(texts: list[str], model_dir: str) -> np.ndarray:

    Usage:
        embedding = CustomScriptEmbedding("models/registry/my_model/v1/embedding.py")
        features = embedding.transform(["Hello World!"])
    """

    def __init__(self, script_path: str):
        """
        Parameters
        ----------
        script_path : str
            Đường dẫn đến file .py chứa hàm transform().
        """
        import importlib.util

        self.script_path = script_path

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Custom embedding script not found: {script_path}")

        # Load module từ file path
        spec = importlib.util.spec_from_file_location("custom_embedding", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load embedding script: {script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Kiểm tra hàm transform
        if not hasattr(module, "transform"):
            raise AttributeError(
                f"Custom embedding script must have a 'transform(texts: list[str], model_dir: str) -> np.ndarray' function. "
                f"Not found in: {script_path}"
            )

        self._transform_fn = module.transform
        self._model_dir = os.path.dirname(script_path)
        print(f"  ✅ Loaded custom embedding: {script_path}")

    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Transform texts thành embedding vectors dùng script custom.

        Parameters
        ----------
        texts : list[str]
            Danh sách văn bản.

        Returns
        -------
        np.ndarray
            Embedding vectors.
        """
        result = self._transform_fn(texts, self._model_dir)
        assert isinstance(
            result, np.ndarray
        ), f"Custom embedding transform must return np.ndarray, got {type(result)}"
        return result

    def get_feature_names(self) -> list[str]:
        """Custom embedding không có feature names cố định."""
        return [f"custom_dim_{i}" for i in range(100)]

    def __repr__(self) -> str:
        return f"CustomScriptEmbedding({self.script_path})"


class EmbeddingFactory:
    """
    Factory tạo embedding dựa trên config.json của model.
    Mỗi model có thể dùng embedding khác nhau.
    """

    @staticmethod
    def create(model_dir: str, config: dict[str, Any]) -> BaseEmbedding:
        """
        Tạo embedding từ config.

        Parameters
        ----------
        model_dir : str
            Đường dẫn thư mục chứa model.
        config : dict
            Config dict của model (chứa thông tin embedding).

        Returns
        -------
        BaseEmbedding
        """
        embedding_cfg = config.get("embedding", {})
        embedding_type = embedding_cfg.get("type", "tfidf")

        # Custom embedding — load từ script file
        if embedding_type == "custom":
            script_file = embedding_cfg.get("script", "embedding.py")
            script_path = os.path.join(model_dir, script_file)
            return CustomScriptEmbedding(script_path)

        if embedding_type == "tfidf":
            files = config.get("files", {})
            vec_file = files.get("vectorizer", "vectorizer.pkl")
            vec_path = os.path.join(model_dir, vec_file)
            return TFIDFEmbedding(vec_path)

        elif embedding_type == "fasttext":
            files = config.get("files", {})
            ft_file = files.get("fasttext_model", "fasttext_model.bin")
            ft_path = os.path.join(model_dir, ft_file)
            return FastTextEmbedding(
                model_path=ft_path,
                pooling=embedding_cfg.get("pooling", "mean"),
                use_sentence_vector=embedding_cfg.get("use_sentence_vector", False),
            )

        elif embedding_type == "bert":
            return BERTEmbedding(
                model_name=embedding_cfg.get("model_name", "bert-base-uncased"),
                max_length=embedding_cfg.get("max_length", 128),
                pooling=embedding_cfg.get("pooling", "cls"),
                device=embedding_cfg.get("device", "cpu"),
            )

        elif embedding_type == "ensemble":
            # Ensemble: kết hợp nhiều embedding methods
            sub_embeddings = embedding_cfg.get("embeddings", [])
            embeddings = []
            for sub_cfg in sub_embeddings:
                # Tạo sub-config tạm thời
                sub_config = dict(config)
                sub_config["embedding"] = sub_cfg
                emb = EmbeddingFactory.create(model_dir, sub_config)
                embeddings.append(emb)
            return EnsembleEmbedding(embeddings)

        elif embedding_type == "none":
            # Không có embedding — model tự xử lý (vd: FastText)
            raise ValueError(
                "Embedding type 'none' is not supported by EmbeddingFactory. "
                "Use a predictor that handles embedding internally (e.g., FastTextPredictor)."
            )

        else:
            raise ValueError(
                f"Unsupported embedding type: '{embedding_type}'. "
                "Supported types: tfidf, fasttext, bert, ensemble, custom"
            )
