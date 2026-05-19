"""
schemas.py

Pydantic schemas cho request/response của FastAPI server.
Hỗ trợ multi-model multi-version.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# === Basic schemas (giữ nguyên từ phiên bản cũ) ===


class CommentRequest(BaseModel):
    """Request body cho endpoint /predict."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Bình luận cần kiểm tra",
        json_schema_extra={"example": "mày ngu vãi"},
    )


class PredictResponse(BaseModel):
    """Response body từ endpoint /predict."""

    label: int = Field(
        ...,
        description="Kết quả phân loại: 1 = toxic, 0 = non-toxic",
        json_schema_extra={"example": 1},
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Xác suất là toxic (0.0 - 1.0)",
        json_schema_extra={"example": 0.9234},
    )
    threshold: float = Field(
        ...,
        description="Ngưỡng quyết định đang dùng",
        json_schema_extra={"example": 0.45},
    )


class HealthResponse(BaseModel):
    """Response body từ endpoint /health."""

    status: str = Field(
        ..., description="Trạng thái server", json_schema_extra={"example": "ok"}
    )
    model: str = Field(
        ...,
        description="Thông tin model đang serve",
        json_schema_extra={"example": "lightgbm_fasttext_v3"},
    )
    model_framework: str = Field(
        ...,
        description="Framework của model",
        json_schema_extra={"example": "lightgbm"},
    )


# === New schemas cho multi-model multi-version ===


class ModelVersionInfo(BaseModel):
    """Thông tin một model version."""

    model_name: str = Field(..., description="Tên model")
    version: str = Field(..., description="Version")
    framework: str = Field(..., description="Framework")
    threshold: float = Field(..., description="Threshold đang dùng")
    embedding_type: str = Field(..., description="Loại embedding")
    status: str = Field("loaded", description="Trạng thái: loaded/not_loaded/error")


class ModelsListResponse(BaseModel):
    """Response cho GET /v1/models."""

    models: list[ModelVersionInfo] = Field(
        ..., description="Danh sách tất cả model/version"
    )
    total: int = Field(..., description="Tổng số model/version")


class ModelDetailResponse(BaseModel):
    """Response cho GET /v1/models/{model_name}."""

    model_name: str = Field(..., description="Tên model")
    versions: list[str] = Field(..., description="Danh sách version có sẵn")
    active_version: str = Field(..., description="Version đang active (latest)")
    loaded_versions: list[str] = Field(
        ..., description="Các version đang được load trong memory"
    )


class PredictWithModelRequest(CommentRequest):
    """Request body cho predict với model cụ thể (có thể thêm params)."""

    params: dict[str, Any] | None = Field(
        None,
        description="Tham số bổ sung (ví dụ: override threshold)",
        json_schema_extra={"example": {"threshold": 0.3}},
    )


class PredictWithModelResponse(PredictResponse):
    """Response cho predict với model cụ thể."""

    model_name: str = Field(..., description="Tên model đã dùng")
    version: str = Field(..., description="Version đã dùng")


class PredictBatchWithModelRequest(BaseModel):
    """Request body cho batch predict với model cụ thể."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Danh sách bình luận cần kiểm tra",
        json_schema_extra={"example": ["mày ngu vãi", "chào bạn"]},
    )
    params: dict[str, Any] | None = Field(
        None,
        description="Tham số bổ sung",
    )


class PredictBatchWithModelResponse(BaseModel):
    """Response cho batch predict với model cụ thể."""

    results: list[PredictWithModelResponse] = Field(
        ..., description="Danh sách kết quả"
    )
    model_name: str = Field(..., description="Tên model đã dùng")
    version: str = Field(..., description="Version đã dùng")


class EnsemblePredictRequest(BaseModel):
    """Request body cho ensemble predict."""

    models: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Danh sách model/version (vd: ['toxic_lstm/v1', 'toxic_bert/v1'])",
        json_schema_extra={"example": ["toxic_lstm/v1", "toxic_bert/v1"]},
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Bình luận cần kiểm tra",
        json_schema_extra={"example": "mày ngu vãi"},
    )
    weights: list[float] | None = Field(
        None,
        description="Trọng số cho từng model (mặc định: equal weight)",
        json_schema_extra={"example": [0.7, 0.3]},
    )
    strategy: str = Field(
        "weighted_average",
        description="Chiến lược ensemble: weighted_average, majority_vote, max_probability",
        json_schema_extra={"example": "weighted_average"},
    )

    @field_validator("models")
    @classmethod
    def validate_model_spec(cls, value: list[str]) -> list[str]:
        """
        Validate format của từng model spec.
        Hợp lệ: "model_name" hoặc "model_name/version" (vd: toxic_lstm/v1).
        """
        import re

        valid_spec = re.compile(r"^[a-z0-9_-]+(/v\d+)?$")
        for spec in value:
            if not valid_spec.match(spec):
                raise ValueError(
                    f"Invalid model spec: '{spec}'. "
                    "Expected format: 'model_name' or 'model_name/v<number>' "
                    "(e.g., 'toxic_lstm' or 'toxic_lstm/v1')"
                )
        return value


class EnsemblePredictResponse(BaseModel):
    """Response cho ensemble predict."""

    label: int = Field(..., description="Kết quả ensemble")
    probability: float = Field(..., description="Xác suất ensemble")
    threshold: float = Field(..., description="Threshold đang dùng")
    strategy: str = Field(..., description="Chiến lược ensemble")
    individual_results: list[dict[str, Any]] = Field(
        ..., description="Kết quả từ từng model"
    )


class HealthV1Response(BaseModel):
    """Response cho GET /v1/health."""

    status: str = Field(..., description="Trạng thái server")
    version: str = Field(..., description="API version")
    models: dict[str, str] = Field(
        ..., description="Trạng thái từng model (model_name/version -> status)"
    )
    loaded_models: list[str] = Field(
        ..., description="Danh sách model đang load trong memory"
    )
    total_models_in_registry: int = Field(
        ..., description="Tổng số model/version trong registry"
    )


class DeployRequest(BaseModel):
    """Request body cho POST /v1/models/deploy — hot-deploy model."""

    model_name: str = Field(
        ...,
        description="Tên model trong registry",
        json_schema_extra={"example": "ban10_fasttext"},
    )
    from_folder: str = Field(
        ...,
        description="Đường dẫn folder chứa model files (đã export từ Colab)",
        json_schema_extra={"example": "/path/to/ban10_fasttext_model"},
    )
    version: str | None = Field(
        None,
        description="Version cụ thể (vd: v1, v2). Mặc định: tự động tăng",
        json_schema_extra={"example": "v2"},
    )
    force: bool = Field(
        False,
        description="Ghi đè version cũ nếu đã tồn tại",
    )


class DeployResponse(BaseModel):
    """Response cho POST /v1/models/deploy."""

    status: str = Field(..., description="Trạng thái deploy")
    model_name: str = Field(..., description="Tên model")
    version: str = Field(..., description="Version đã deploy")
    path: str = Field(..., description="Đường dẫn thư mục model trong registry")
    message: str = Field(..., description="Thông báo")
