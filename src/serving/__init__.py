"""
src/serving/

Module phục vụ model (model serving) — hỗ trợ multi-model multi-version.
Bao gồm:
- model_saver.py: Lưu model sau khi train vào Model Registry
- model_loader.py: Load model từ disk
- model_registry.py: Singleton quản lý tất cả model đã load (lazy loading + cache)
- model_packager.py: Export + Deploy model từ Colab vào registry
- schemas.py: Pydantic request/response
- predictor.py: Predict pipeline (BasePredictor + các subclass)
- embedding_factory.py: Embedding factory (TF-IDF, Word2Vec, BERT, custom script)
- preprocessing_factory.py: Preprocessing pipeline (bao gồm custom script)
- app.py: FastAPI server (multi-model multi-version endpoints)
"""

from src.serving.embedding_factory import (
    BaseEmbedding,
    BERTEmbedding,
    CustomScriptEmbedding,
    EmbeddingFactory,
    EnsembleEmbedding,
    FastTextEmbedding,
    TFIDFEmbedding,
)
from src.serving.model_packager import ModelPackager
from src.serving.model_registry import ModelInfo, ModelNotFoundError, ModelRegistry
from src.serving.predictor import (
    BasePredictor,
    CustomPredictor,
    EnsemblePredictor,
    FastTextPredictor,
    SklearnPredictor,
    TensorFlowPredictor,
    TorchPredictor,
)
from src.serving.preprocessing_factory import (
    CustomScriptPreprocessor,
    PreprocessingPipeline,
)

__all__ = [
    "BaseEmbedding",
    "BasePredictor",
    "BERTEmbedding",
    "CustomPredictor",
    "CustomScriptEmbedding",
    "CustomScriptPreprocessor",
    "EmbeddingFactory",
    "EnsembleEmbedding",
    "EnsemblePredictor",
    "FastTextEmbedding",
    "FastTextPredictor",
    "ModelInfo",
    "ModelNotFoundError",
    "ModelPackager",
    "ModelRegistry",
    "PreprocessingPipeline",
    "SklearnPredictor",
    "TensorFlowPredictor",
    "TFIDFEmbedding",
    "TorchPredictor",
]
