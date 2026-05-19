"""
app.py

FastAPI server cho Toxic Comment Classification.
Hỗ trợ multi-model multi-version serving.

Cách chạy:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

Hoặc:
    python -m src.serving.app
"""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi import Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware

from src.serving.model_registry import ModelNotFoundError, ModelRegistry
from src.serving.schemas import (
    CommentRequest,
    DeployRequest,
    DeployResponse,
    HealthResponse,
    ModelDetailResponse,
    ModelsListResponse,
    ModelVersionInfo,
    PredictBatchWithModelRequest,
    PredictBatchWithModelResponse,
    PredictParams,
    PredictResponse,
    PredictWithModelRequest,
    PredictWithModelResponse,
)

# === Lifespan context manager ===


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Khởi tạo server — kiểm tra registry."""
    registry_dir = ModelRegistry.get_registry_dir()
    print(f"\n{'='*50}")
    print(f"  Model Registry: {registry_dir}")
    print(f"{'='*50}")

    if not os.path.isdir(registry_dir):
        print(f"  Warning: Registry directory not found: {registry_dir}")
        print("  Hint: Create models/registry/<model_name>/<version>/ with config.json")
        print("  Hint: Or set MODEL_REGISTRY_DIR env var\n")
    else:
        models = ModelRegistry.list_available_models()
        if models:
            print("  Available models in registry:")
            for m in models:
                print(f"     - {m.key} ({m.framework}, {m.embedding_type})")
        else:
            print("  Warning: No models found in registry")
        print(f"{'='*50}\n")
    yield


# --- Khởi tạo app ---
app = FastAPI(
    title="Toxic Comment Classification API",
    description="API phân loại bình luận độc hại — hỗ trợ multi-model multi-version",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Helper functions ===


def _parse_model_version(model_spec: str) -> tuple[str, str]:
    """
    Parse model spec string thành (model_name, version).
    Format: "model_name" hoặc "model_name/version"

    Examples:
        "toxic_lstm" -> ("toxic_lstm", "latest")
        "toxic_lstm/v1" -> ("toxic_lstm", "v1")
        "toxic_lstm/v2" -> ("toxic_lstm", "v2")
    """
    if "/" in model_spec:
        parts = model_spec.split("/", 1)
        return parts[0], parts[1]
    return model_spec, "latest"


def _apply_threshold_override(
    result: dict[str, Any],
    params: PredictParams | None,
) -> dict[str, Any]:
    """
    Override threshold trong kết quả predict nếu params có threshold.
    """
    if params and params.threshold is not None:
        override_threshold = float(params.threshold)
        result["label"] = int(result["probability"] >= override_threshold)
        result["threshold"] = override_threshold
    return result


# === Health & Models (không version) ===


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Kiểm tra trạng thái server."""
    loaded = ModelRegistry.get_loaded_models()
    return HealthResponse(
        status="ok",
        version="2.0.0",
        loaded_models=loaded,
        total_models_in_registry=len(ModelRegistry.list_available_models()),
    ).model_dump()


@app.get("/models", response_model=ModelsListResponse)
def list_models() -> dict[str, Any]:
    """Liệt kê tất cả model/version có trong registry."""
    all_models = ModelRegistry.list_available_models()
    loaded = ModelRegistry.get_loaded_models()

    models_info = []
    for m in all_models:
        status = "loaded" if m.key in loaded else "not_loaded"
        models_info.append(
            ModelVersionInfo(
                model_name=m.model_name,
                version=m.version,
                framework=m.framework,
                threshold=m.threshold,
                embedding_type=m.embedding_type,
                status=status,
            )
        )

    return ModelsListResponse(
        models=models_info,
        total=len(models_info),
    ).model_dump()


@app.get("/models/{model_name}", response_model=ModelDetailResponse)
def get_model_detail(
    model_name: str = FastAPIPath(..., description="Tên model"),
) -> dict[str, Any]:
    """Lấy thông tin chi tiết của một model (tất cả versions)."""
    all_models = ModelRegistry.list_available_models()
    loaded = ModelRegistry.get_loaded_models()

    model_versions = [m for m in all_models if m.model_name == model_name]

    if not model_versions:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' không tìm thấy trong registry.",
        )

    try:
        latest = ModelRegistry.resolve_latest_version(model_name)
    except ModelNotFoundError:
        latest = "unknown"

    versions = sorted([m.version for m in model_versions])
    loaded_versions = [m.version for m in model_versions if m.key in loaded]

    return ModelDetailResponse(
        model_name=model_name,
        versions=versions,
        active_version=latest,
        loaded_versions=loaded_versions,
    ).model_dump()


# === Predict (legacy — dùng model mặc định) ===


@app.post("/predict", response_model=PredictResponse)
def predict(request: CommentRequest) -> dict[str, Any]:
    """
    Dự đoán toxicity (dùng model mặc định).
    Model mặc định là model đầu tiên tìm thấy trong registry.
    """
    models = ModelRegistry.list_available_models()
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Không có model nào trong registry. "
            "Vui lòng thêm model vào models/registry/",
        )

    first_model = models[0]
    predictor = ModelRegistry.get_predictor(first_model.model_name, "latest")

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
    Dự đoán cho nhiều bình luận cùng lúc (dùng model mặc định).
    """
    models = ModelRegistry.list_available_models()
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Không có model nào trong registry.",
        )

    first_model = models[0]
    predictor = ModelRegistry.get_predictor(first_model.model_name, "latest")

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


# === Predict với model cụ thể ===


@app.post(
    "/models/{model_name}/predict",
    response_model=PredictWithModelResponse,
)
def predict_with_model(
    request: PredictWithModelRequest,
    model_name: str = FastAPIPath(..., description="Tên model"),
) -> dict[str, Any]:
    """
    Dự đoán với model cụ thể (dùng version mới nhất).
    """
    try:
        predictor = ModelRegistry.get_predictor(model_name, "latest")
        result = predictor.predict(request.text)
        result = _apply_threshold_override(result, request.params)

        info = ModelRegistry.get_model_info(model_name, "latest")

        return PredictWithModelResponse(
            label=result["label"],
            probability=result["probability"],
            threshold=result["threshold"],
            model_name=model_name,
            version=info.version,
        ).model_dump()

    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi dự đoán: {e!s}",
        ) from e


@app.post(
    "/models/{model_name}/{version}/predict",
    response_model=PredictWithModelResponse,
)
def predict_with_model_version(
    request: PredictWithModelRequest,
    model_name: str = FastAPIPath(..., description="Tên model"),
    version: str = FastAPIPath(..., description="Version (vd: v1, v2)"),
) -> dict[str, Any]:
    """
    Dự đoán với model cụ thể và version cụ thể.
    """
    try:
        predictor = ModelRegistry.get_predictor(model_name, version)
        result = predictor.predict(request.text)
        result = _apply_threshold_override(result, request.params)

        return PredictWithModelResponse(
            label=result["label"],
            probability=result["probability"],
            threshold=result["threshold"],
            model_name=model_name,
            version=version,
        ).model_dump()

    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi dự đoán: {e!s}",
        ) from e


@app.post(
    "/models/{model_name}/predict_batch",
    response_model=PredictBatchWithModelResponse,
)
def predict_batch_with_model(
    request: PredictBatchWithModelRequest,
    model_name: str = FastAPIPath(..., description="Tên model"),
) -> dict[str, Any]:
    """
    Batch predict với model cụ thể (dùng version mới nhất).
    """
    try:
        predictor = ModelRegistry.get_predictor(model_name, "latest")
        results = predictor.predict_batch(request.texts)

        info = ModelRegistry.get_model_info(model_name, "latest")

        response_results = []
        for r in results:
            r = _apply_threshold_override(r, request.params)
            response_results.append(
                PredictWithModelResponse(
                    label=r["label"],
                    probability=r["probability"],
                    threshold=r["threshold"],
                    model_name=model_name,
                    version=info.version,
                )
            )

        return PredictBatchWithModelResponse(
            results=response_results,
            model_name=model_name,
            version=info.version,
        ).model_dump()

    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi dự đoán: {e!s}",
        ) from e


@app.post(
    "/models/{model_name}/{version}/predict_batch",
    response_model=PredictBatchWithModelResponse,
)
def predict_batch_with_model_version(
    request: PredictBatchWithModelRequest,
    model_name: str = FastAPIPath(..., description="Tên model"),
    version: str = FastAPIPath(..., description="Version (vd: v1, v2)"),
) -> dict[str, Any]:
    """
    Batch predict với model cụ thể và version cụ thể.
    """
    try:
        predictor = ModelRegistry.get_predictor(model_name, version)
        results = predictor.predict_batch(request.texts)

        response_results = []
        for r in results:
            r = _apply_threshold_override(r, request.params)
            response_results.append(
                PredictWithModelResponse(
                    label=r["label"],
                    probability=r["probability"],
                    threshold=r["threshold"],
                    model_name=model_name,
                    version=version,
                )
            )

        return PredictBatchWithModelResponse(
            results=response_results,
            model_name=model_name,
            version=version,
        ).model_dump()

    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi dự đoán: {e!s}",
        ) from e


# === Model management endpoints ===


@app.post("/models/{model_name}/reload")
def reload_model(
    model_name: str = FastAPIPath(..., description="Tên model"),
    version: str = "latest",
) -> dict[str, str]:
    """Reload model (bỏ cache cũ, load lại từ disk)."""
    try:
        ModelRegistry.reload_model(model_name, version)
        return {
            "status": "ok",
            "message": f"Model '{model_name}:{version}' đã được reload.",
        }
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi reload model: {e!s}",
        ) from e


@app.post("/models/{model_name}/unload")
def unload_model(
    model_name: str = FastAPIPath(..., description="Tên model"),
    version: str | None = None,
) -> dict[str, str]:
    """Giải phóng model khỏi cache."""
    try:
        ModelRegistry.unload_model(model_name, version)
        if version:
            msg = f"Model '{model_name}:{version}' đã được unload."
        else:
            msg = f"Tất cả versions của model '{model_name}' đã được unload."
        return {"status": "ok", "message": msg}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi unload model: {e!s}",
        ) from e


@app.post("/cache/clear")
def clear_cache() -> dict[str, str]:
    """Xóa toàn bộ cache (giải phóng memory)."""
    count = len(ModelRegistry.get_loaded_models())
    ModelRegistry.clear_cache()
    return {
        "status": "ok",
        "message": f"Đã xóa {count} models khỏi cache.",
    }


@app.post("/models/deploy", response_model=DeployResponse)
def deploy_model(request: DeployRequest) -> dict[str, Any]:
    """
    Hot-deploy model từ folder vào registry (không cần restart server).
    """
    from src.serving.model_packager import ModelPackager

    try:
        dest = ModelPackager.deploy(
            model_name=request.model_name,
            from_folder=request.from_folder,
            version=request.version,
            force=request.force,
        )

        version = os.path.basename(dest)

        return DeployResponse(
            status="ok",
            model_name=request.model_name,
            version=version,
            path=dest,
            message=f"Model '{request.model_name}:{version}' đã được deploy thành công.",
        ).model_dump()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi deploy model: {e!s}",
        ) from e


# === CLI ===
def main() -> None:
    """Chạy server từ command line."""
    parser = argparse.ArgumentParser(
        description="Toxic Comment Classification API (Multi-Model Multi-Version)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (mặc định: 8000)")
    parser.add_argument(
        "--registry-dir",
        default=None,
        help="Đường dẫn thư mục model registry",
    )
    args = parser.parse_args()

    if args.registry_dir:
        os.environ["MODEL_REGISTRY_DIR"] = args.registry_dir

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
