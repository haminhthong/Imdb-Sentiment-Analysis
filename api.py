"""REST API phục vụ phân loại cảm xúc bằng FastAPI."""

import os
from functools import lru_cache
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentiment.inference import Predictor, load_predictor

APP_NAME = "CineSentiment REST API"
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "artifacts/model.pt")

app = FastAPI(
    title=APP_NAME,
    description="REST API phục vụ phân loại cảm xúc đánh giá phim IMDB với xác suất hiệu chuẩn.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ReviewText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=10_000,
        description="Nội dung văn bản đánh giá phim tiếng Anh.",
        examples=["This movie had a fantastic storyline and outstanding performances!"],
    ),
]


class PredictionRequest(BaseModel):
    """Schema yêu cầu dự đoán cho một câu đơn."""

    text: ReviewText


class BatchPredictionRequest(BaseModel):
    """Schema yêu cầu dự đoán theo lô (batch)."""

    texts: list[ReviewText] = Field(
        min_length=1,
        max_length=100,
        description="Danh sách các câu đánh giá (tối đa 100 câu mỗi request).",
    )


class PredictionResponse(BaseModel):
    """Schema kết quả dự đoán cảm xúc tinh gọn."""

    label: str = Field(description="Nhãn dự đoán: 'Positive' hoặc 'Negative'.")
    probability: float = Field(
        description="Xác suất thuộc lớp Positive đã qua Temperature Scaling."
    )
    truncated: bool = Field(description="Cờ đánh dấu câu bị cắt bớt do vượt max_length.")
    oov_rate: float = Field(description="Tỷ lệ từ ngoài từ điển (OOV Rate).")
    token_count: int = Field(description="Tổng số token của văn bản gốc.")
    warnings: list[str] = Field(default_factory=list, description="Danh sách cảnh báo độ tin cậy.")


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Nạp checkpoint mô hình vào bộ nhớ."""
    return load_predictor(CHECKPOINT_PATH, device="cpu")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Kiểm tra tình trạng hoạt động của service."""
    return {
        "status": "healthy",
        "service": APP_NAME,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    """Dự đoán cảm xúc của một câu đánh giá đơn lẻ."""
    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Mô hình chưa sẵn sàng: {exc}",
        ) from exc

    try:
        res = predictor.predict(request.text)
        return PredictionResponse(
            label=res.label,
            probability=res.probability,
            truncated=res.truncated,
            oov_rate=res.oov_rate,
            token_count=res.token_count,
            warnings=res.warnings,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi khi xử lý dự đoán: {exc}",
        ) from exc


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_sentiment_batch(request: BatchPredictionRequest) -> list[PredictionResponse]:
    """Dự đoán cảm xúc của danh sách câu đánh giá theo lô."""
    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Mô hình chưa sẵn sàng: {exc}",
        ) from exc

    try:
        results = predictor.predict_batch(request.texts)
        return [
            PredictionResponse(
                label=r.label,
                probability=r.probability,
                truncated=r.truncated,
                oov_rate=r.oov_rate,
                token_count=r.token_count,
                warnings=r.warnings,
            )
            for r in results
        ]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi khi xử lý dự đoán theo lô: {exc}",
        ) from exc


def run() -> None:
    """Chạy API server qua Uvicorn."""
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
