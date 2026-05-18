"""
schemas.py

Pydantic schemas cho request/response của FastAPI server.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
