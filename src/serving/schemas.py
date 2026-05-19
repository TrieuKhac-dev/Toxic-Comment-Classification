"""
schemas.py

Pydantic schemas cho request/response của FastAPI server.
Hỗ trợ multi-model multi-version.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# === Basic schemas ===


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
    version: str = Field(
        ..., description="Phiên bản API", json_schema_extra={"example": "2.0.0"}
    )
    loaded_models: list[str] = Field(
        ..., description="Danh sách model đang load trong memory"
    )
    total_models_in_registry: int = Field(
        ..., description="Tổng số model/version trong registry"
    )


# === Multi-model schemas ===


class ModelVersionInfo(BaseModel):
    """Thông tin một model version."""

    model_name: str = Field(..., description="Tên model")
    version: str = Field(..., description="Version")
    framework: str = Field(..., description="Framework")
    threshold: float = Field(..., description="Threshold đang dùng")
    embedding_type: str = Field(..., description="Loại embedding")
    status: str = Field("loaded", description="Trạng thái: loaded/not_loaded/error")


class ModelsListResponse(BaseModel):
    """Response cho GET /models."""

    models: list[ModelVersionInfo] = Field(
        ..., description="Danh sách tất cả model/version"
    )
    total: int = Field(..., description="Tổng số model/version")


class ModelDetailResponse(BaseModel):
    """Response cho GET /models/{model_name}."""

    model_name: str = Field(..., description="Tên model")
    versions: list[str] = Field(..., description="Danh sách version có sẵn")
    active_version: str = Field(..., description="Version đang active (latest)")
    loaded_versions: list[str] = Field(
        ..., description="Các version đang được load trong memory"
    )


class PredictParams(BaseModel):
    """Tham số tùy chọn cho predict."""

    threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override threshold (mặc định: dùng threshold của model)",
        json_schema_extra={"example": 0.3},
    )


class PredictWithModelRequest(BaseModel):
    """Request body cho predict với model cụ thể."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Bình luận cần kiểm tra",
        json_schema_extra={"example": "mày ngu vãi"},
    )
    params: PredictParams | None = Field(
        None,
        description="Tham số tùy chọn (ví dụ: override threshold)",
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
    params: PredictParams | None = Field(
        None,
        description="Tham số tùy chọn (áp dụng cho tất cả texts)",
    )


class PredictBatchWithModelResponse(BaseModel):
    """Response cho batch predict với model cụ thể."""

    results: list[PredictWithModelResponse] = Field(
        ..., description="Danh sách kết quả"
    )
    model_name: str = Field(..., description="Tên model đã dùng")
    version: str = Field(..., description="Version đã dùng")


class DeployRequest(BaseModel):
    """Request body cho POST /models/deploy — hot-deploy model."""

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
    """Response cho POST /models/deploy."""

    status: str = Field(..., description="Trạng thái deploy")
    model_name: str = Field(..., description="Tên model")
    version: str = Field(..., description="Version đã deploy")
    path: str = Field(..., description="Đường dẫn thư mục model trong registry")
    message: str = Field(..., description="Thông báo")
