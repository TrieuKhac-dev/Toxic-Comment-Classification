"""
app.py

FastAPI server cho Toxic Comment Classification.
Load model từ production/ và expose REST API.

Cách chạy:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

Hoặc:
    python -m src.serving.app
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.serving.predictor import Predictor
from src.serving.schemas import CommentRequest, HealthResponse, PredictResponse

# --- Khởi tạo app ---
app = FastAPI(
    title="Toxic Comment Classification API",
    description="API phân loại bình luận độc hại",
    version="1.0.0",
)

# CORS - cho phép mọi origin (có thể giới hạn sau)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global predictor ---
predictor: Predictor | None = None


def get_model_dir() -> str:
    """Lấy đường dẫn thư mục production."""
    # Ưu tiên env var, sau đó mặc định
    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        return env_dir

    # Tìm trong project root
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        project_root / "models" / "production",
        Path("models/production"),
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return str(candidate)

    # Mặc định
    return "models/production"


@app.on_event("startup")
def load_model() -> None:
    """Load model khi server khởi động."""
    global predictor
    model_dir = get_model_dir()
    print(f"\n{'='*50}")
    print(f"  Loading model from: {model_dir}")
    print(f"{'='*50}")

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"  ⚠️  Không tìm thấy config.json tại: {model_dir}")
        print("  ⚠️  Server sẽ khởi động nhưng /predict sẽ trả về lỗi.")
        print("  💡  Chạy 'python scripts/serving/download_model.py' để tải model.\n")
        return

    try:
        predictor = Predictor(model_dir)
        print("  ✅ Server ready!")
        print(f"{'='*50}\n")
    except Exception as e:
        print(f"  ❌ Lỗi load model: {e}")
        print(f"{'='*50}\n")


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Kiểm tra trạng thái server."""
    if predictor is None:
        return HealthResponse(
            status="error",
            model="not_loaded",
            model_framework="unknown",
        ).model_dump()

    return HealthResponse(
        status="ok",
        model=predictor.config.get("experiment_name", "unknown"),
        model_framework=predictor.config.get("model_framework", "unknown"),
    ).model_dump()


@app.post("/predict", response_model=PredictResponse)
def predict(request: CommentRequest) -> dict[str, Any]:
    """
    Dự đoán toxicity của bình luận.

    Request body:
        {"text": "mày ngu vãi"}

    Response:
        {"label": 1, "probability": 0.9234, "threshold": 0.45}
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa được load. Vui lòng kiểm tra /health",
        )

    try:
        result = predictor.predict(request.text)
        return PredictResponse(
            label=result["label"],
            probability=result["probability"],
            threshold=result["threshold"],
        ).model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi dự đoán: {e!s}",
        ) from e


@app.post("/predict_batch")
def predict_batch(requests: list[CommentRequest]) -> list[dict[str, Any]]:
    """
    Dự đoán cho nhiều bình luận cùng lúc.

    Request body:
        [{"text": "mày ngu vãi"}, {"text": "chào bạn"}]

    Response:
        [{"label": 1, "probability": 0.9234, "threshold": 0.45}, ...]
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa được load. Vui lòng kiểm tra /health",
        )

    try:
        texts = [r.text for r in requests]
        results = predictor.predict_batch(texts)
        return [
            PredictResponse(
                label=r["label"],
                probability=r["probability"],
                threshold=r["threshold"],
            ).model_dump()
            for r in results
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi dự đoán: {e!s}",
        ) from e


# --- CLI ---
def main() -> None:
    """Chạy server từ command line."""
    parser = argparse.ArgumentParser(description="Toxic Comment Classification API")
    parser.add_argument("--host", default="0.0.0.0", help="Host (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (mặc định: 8000)")
    parser.add_argument("--model-dir", default=None, help="Đường dẫn thư mục model")
    args = parser.parse_args()

    if args.model_dir:
        os.environ["MODEL_DIR"] = args.model_dir

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
