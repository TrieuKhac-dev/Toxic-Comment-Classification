"""
demo_multi_model.py

Script demo hệ thống serving multi-model multi-version.
Tạo 2 model giả trong registry và test các API endpoints.

Cách chạy:
    # Chạy server trước:
    python -m src.serving.app

    # Sau đó chạy demo này:
    python scripts/serving/demo_multi_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.serving.model_registry import ModelRegistry
from src.serving.model_saver import (
    create_model_version_dir,
    list_registry_models,
    save_config,
)

# Thêm project root vào path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def create_demo_models() -> None:
    """
    Tạo 2 model demo trong registry để minh họa.
    Model 1: toxic_lstm/v1 (sklearn + TF-IDF)
    Model 2: toxic_lstm/v2 (sklearn + TF-IDF, threshold khác)
    Model 3: toxic_bert/v1 (pytorch + BERT)
    """
    print("=" * 60)
    print("  Creating demo models in registry...")
    print("=" * 60)

    # === Model 1: toxic_lstm/v1 ===
    model_dir = create_model_version_dir("toxic_lstm", "v1")
    save_config(
        {
            "model_name": "toxic_lstm",
            "version": "v1",
            "model_framework": "sklearn",
            "threshold": 0.5,
            "embedding": {
                "type": "tfidf",
            },
            "preprocessing": {
                "pipeline": [
                    {"name": "lowercase", "enabled": True},
                    {"name": "remove_url", "enabled": True},
                    {"name": "remove_html", "enabled": True},
                    {"name": "strip_spaces", "enabled": True},
                ]
            },
            "files": {
                "model": "model.pkl",
                "vectorizer": "vectorizer.pkl",
            },
            "metrics": {"f1": 0.85, "accuracy": 0.90},
            "description": "LSTM baseline with TF-IDF features",
        },
        model_dir,
    )

    # === Model 2: toxic_lstm/v2 ===
    model_dir = create_model_version_dir("toxic_lstm", "v2")
    save_config(
        {
            "model_name": "toxic_lstm",
            "version": "v2",
            "model_framework": "sklearn",
            "threshold": 0.45,
            "embedding": {
                "type": "tfidf",
            },
            "preprocessing": {
                "pipeline": [
                    {"name": "lowercase", "enabled": True},
                    {"name": "remove_url", "enabled": True},
                    {"name": "remove_html", "enabled": True},
                    {"name": "remove_emoji", "enabled": True},
                    {"name": "replace_numbers", "enabled": True},
                    {"name": "strip_spaces", "enabled": True},
                ]
            },
            "files": {
                "model": "model.pkl",
                "vectorizer": "vectorizer.pkl",
            },
            "metrics": {"f1": 0.88, "accuracy": 0.92},
            "description": "LSTM v2 with improved preprocessing",
        },
        model_dir,
    )

    # === Model 3: toxic_bert/v1 ===
    model_dir = create_model_version_dir("toxic_bert", "v1")
    save_config(
        {
            "model_name": "toxic_bert",
            "version": "v1",
            "model_framework": "pytorch",
            "model_class": "transformers.BertForSequenceClassification",
            "model_kwargs": {
                "num_labels": 2,
            },
            "threshold": 0.5,
            "embedding": {
                "type": "bert",
                "model_name": "bert-base-uncased",
                "max_length": 128,
                "pooling": "cls",
            },
            "preprocessing": {
                "pipeline": [
                    {"name": "normalize_unicode", "enabled": True},
                    {"name": "strip_spaces", "enabled": True},
                ]
            },
            "files": {
                "model": "pytorch_model.bin",
                "tokenizer": "bert-base-uncased",
            },
            "metrics": {"f1": 0.93, "accuracy": 0.95},
            "description": "BERT fine-tuned for toxicity detection",
        },
        model_dir,
    )

    print("\n  ✅ Demo models created successfully!")
    print("  📍 Location: models/registry/")
    print()


def test_registry() -> None:
    """Test ModelRegistry functions."""
    print("=" * 60)
    print("  Testing ModelRegistry...")
    print("=" * 60)

    # Liệt kê tất cả models
    models = ModelRegistry.list_available_models()
    print(f"\n  📦 Available models ({len(models)}):")
    for m in models:
        print(f"     - {m.key}")
        print(f"       framework: {m.framework}")
        print(f"       threshold: {m.threshold}")
        print(f"       embedding: {m.embedding_type}")

    # Test get_model_info
    print("\n  🔍 Getting model info for toxic_lstm:latest...")
    info = ModelRegistry.get_model_info("toxic_lstm", "latest")
    print(f"     Resolved: {info.key}")

    # Test list_registry_models từ model_saver
    print("\n  📋 Registry models (from model_saver):")
    registry_models = list_registry_models()
    for rm in registry_models:
        print(
            f"     - {rm['model_name']}/{rm['version']} (has_config={rm['has_config']})"
        )

    print()


def test_predictors() -> None:
    """Test tạo predictors (sẽ fail vì không có model thật, nhưng minh họa cách dùng)."""
    print("=" * 60)
    print("  Testing Predictor creation (will fail - no real models)...")
    print("=" * 60)

    try:
        # Thử load model (sẽ fail vì không có file model thật)
        predictor = ModelRegistry.get_predictor("toxic_lstm", "v1")
        print(f"  ✅ Predictor loaded: {predictor}")
    except Exception as e:
        print(f"  ⚠️  Expected error (no real model files): {e}")

    print()
    print("  💡 To use real models, train and save them to:")
    print("     models/registry/<model_name>/<version>/")
    print()
    print("  💡 Then run the server and use the API:")
    print("     python -m src.serving.app")
    print()


def print_api_docs() -> None:
    """In hướng dẫn sử dụng API."""
    print("=" * 60)
    print("  API Endpoints Guide")
    print("=" * 60)
    print("""
  📌 Legacy Endpoints (backward compatible):
     GET  /health           - Health check
     POST /predict          - Predict with default model
     POST /predict_batch    - Batch predict with default model

  📌 V1 Endpoints (multi-model multi-version):
     GET  /v1/health                          - Detailed health check
     GET  /v1/models                          - List all models
     GET  /v1/models/{model_name}             - Model details
     POST /v1/models/{model_name}/predict     - Predict with latest version
     POST /v1/models/{model_name}/{version}/predict  - Predict with specific version
     POST /v1/models/{model_name}/predict_batch      - Batch predict
     POST /v1/models/{model_name}/{version}/predict_batch  - Batch predict specific version
     POST /v1/ensemble/predict                - Ensemble multiple models

   📌 Management Endpoints:
      POST /v1/models/{model_name}/reload      - Reload model
      POST /v1/models/{model_name}/unload      - Unload model from cache
      POST /v1/cache/clear                     - Clear all cache
      POST /v1/models/deploy                   - Hot-deploy model from folder

   📌 CLI Tools:
      python scripts/serving/deploy_model.py --list                    - List models
      python scripts/serving/deploy_model.py <name> --from-folder <path>  - Deploy model
      python scripts/serving/run_server.py --deploy <folder>           - Deploy + start server

   📌 Example Requests:
     # Predict with specific model and version
     curl -X POST "http://localhost:8000/v1/models/toxic_lstm/v1/predict" \\
          -H "Content-Type: application/json" \\
          -d '{"text": "mày ngu vãi"}'

     # Ensemble multiple models
     curl -X POST "http://localhost:8000/v1/ensemble/predict" \\
          -H "Content-Type: application/json" \\
          -d '{
             "models": ["toxic_lstm/v1", "toxic_lstm/v2"],
             "text": "mày ngu vãi",
             "strategy": "weighted_average",
             "weights": [0.5, 0.5]
          }'

     # Override threshold
     curl -X POST "http://localhost:8000/v1/models/toxic_lstm/v2/predict" \\
          -H "Content-Type: application/json" \\
          -d '{"text": "mày ngu vãi", "params": {"threshold": 0.3}}'
    """)


if __name__ == "__main__":
    create_demo_models()
    test_registry()
    test_predictors()
    print_api_docs()
