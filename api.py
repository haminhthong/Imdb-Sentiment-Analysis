"""REST API phục vụ phân loại cảm xúc bằng FastAPI và PyTorch.

Cung cấp các endpoints chính:
- `GET /health`: Kiểm tra trạng thái hệ thống.
- `GET /ready`: Kiểm tra khả năng phục vụ và checkpoint mô hình.
- `GET /info`: Lấy thông tin chi tiết cấu hình và số tham số của mô hình đang nạp.
- `POST /predict`: Dự đoán cảm xúc của một câu đánh giá đơn lẻ.
- `POST /predict/batch`: Dự đoán cảm xúc danh sách câu theo lô.
"""

import os
import threading
import time
import uuid
from collections import defaultdict, deque
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from sentiment.inference import SentimentPredictor

APP_NAME = "CineSentiment AI REST API"
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "artifacts/bilstm/model.pt")
API_KEY = os.getenv("API_KEY")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
if RATE_LIMIT_PER_MINUTE <= 0:
    raise ValueError("RATE_LIMIT_PER_MINUTE phải lớn hơn 0.")

REQUEST_COUNTS = {"total": 0, "errors": 0, "latency_seconds": 0.0}
METRICS_LOCK = threading.Lock()
RATE_LIMIT_LOCK = threading.Lock()
REQUEST_TIMESTAMPS: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(
    title=APP_NAME,
    description=(
        "REST API hiệu năng cao phân loại cảm xúc đánh giá phim tiếng Anh "
        "sử dụng các mô hình PyTorch Recurrent Neural Network (LSTM, GRU, BiLSTM)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Kiểu dữ liệu kiểm tra nội dung câu đánh giá
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
    """Schema yêu cầu dự đoán cho câu đơn."""

    text: ReviewText


class BatchPredictionRequest(BaseModel):
    """Schema yêu cầu dự đoán theo lô (batch)."""

    texts: list[ReviewText] = Field(
        min_length=1,
        max_length=100,
        description="Danh sách các câu đánh giá (tối đa 100 câu mỗi request).",
    )


class PredictionResponse(BaseModel):
    """Schema kết quả dự đoán cảm xúc."""

    label: str = Field(description="Nhãn dự đoán: 'Positive' hoặc 'Negative'.")
    positive_probability: float = Field(
        description="Xác suất thuộc lớp Positive (0.0 -> 1.0)."
    )
    confidence: float = Field(description="Độ tin cậy của dự đoán (max(p, 1-p)).")


class ModelInfoResponse(BaseModel):
    """Schema thông tin mô hình đang nạp."""

    model_type: str
    total_parameters: int
    vocabulary_size: int
    max_length: int
    checkpoint_path: str


@lru_cache(maxsize=1)
def get_predictor() -> SentimentPredictor:
    """Nạp trọng số mô hình vào RAM một lần duy nhất (Singleton Pattern).

    Returns:
        SentimentPredictor: Đối tượng suy luận đã nạp mô hình.
    """
    return SentimentPredictor(CHECKPOINT_PATH, device="cpu")


def get_ready_predictor() -> SentimentPredictor:
    """Lấy predictor hoặc trả lỗi 503 khi model chưa sẵn sàng."""
    try:
        return get_predictor()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Dịch vụ chưa sẵn sàng: {error}",
        ) from error


def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Yêu cầu API key khi biến môi trường API_KEY đã được cấu hình."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key không hợp lệ.",
        )


def enforce_rate_limit(request: Request) -> None:
    """Giới hạn số yêu cầu dự đoán theo địa chỉ client trong một phút."""
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - 60
    with RATE_LIMIT_LOCK:
        timestamps = REQUEST_TIMESTAMPS[client]
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Đã vượt giới hạn yêu cầu. Vui lòng thử lại sau.",
            )
        timestamps.append(now)


@app.middleware("http")
async def observe_request(request: Request, call_next):
    """Gắn request ID và thu thập metric tối thiểu cho vận hành."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        with METRICS_LOCK:
            REQUEST_COUNTS["total"] += 1
            REQUEST_COUNTS["errors"] += 1
            REQUEST_COUNTS["latency_seconds"] += time.perf_counter() - started
        raise

    elapsed = time.perf_counter() - started
    with METRICS_LOCK:
        REQUEST_COUNTS["total"] += 1
        REQUEST_COUNTS["errors"] += int(response.status_code >= 500)
        REQUEST_COUNTS["latency_seconds"] += elapsed
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", tags=["System"], summary="Kiểm tra health system")
def health() -> dict[str, str]:
    """Trả về trạng thái hoạt động của server API."""
    return {"status": "healthy", "service": APP_NAME}


@app.get("/metrics", tags=["System"], summary="Metric vận hành cơ bản")
def metrics() -> Response:
    """Trả metric dạng Prometheus text format mà không làm lộ nội dung request."""
    with METRICS_LOCK:
        total = REQUEST_COUNTS["total"]
        errors = REQUEST_COUNTS["errors"]
        latency = REQUEST_COUNTS["latency_seconds"]
    body = (
        "# TYPE cinesentiment_requests_total counter\n"
        f"cinesentiment_requests_total {total}\n"
        "# TYPE cinesentiment_errors_total counter\n"
        f"cinesentiment_errors_total {errors}\n"
        "# TYPE cinesentiment_request_latency_seconds_total counter\n"
        f"cinesentiment_request_latency_seconds_total {latency:.6f}\n"
    )
    return Response(body, media_type="text/plain; version=0.0.4")


@app.get("/ready", tags=["System"], summary="Kiểm tra sẵn sàng suy luận")
def readiness() -> dict[str, Any]:
    """Kiểm tra mô hình đã được nạp thành công vào RAM chưa."""
    predictor = get_ready_predictor()
    return {
        "status": "ready",
        "model_type": predictor.config.model_type,
        "checkpoint": str(predictor.checkpoint_path),
    }


@app.get(
    "/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Lấy thông tin cấu hình mô hình",
)
def model_info() -> dict[str, Any]:
    """Lấy thông tin siêu tham số và số lượng tham số mô hình."""
    predictor = get_ready_predictor()
    return {
        "model_type": predictor.config.model_type,
        "total_parameters": predictor.model.count_parameters(),
        "vocabulary_size": len(predictor.vocabulary),
        "max_length": predictor.config.max_length,
        "checkpoint_path": str(predictor.checkpoint_path),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Phân tích cảm xúc câu đơn",
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Dự đoán cảm xúc cho một câu đánh giá phim duy nhất."""
    try:
        result = get_ready_predictor().predict(request.text)
        return result.to_dict()
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    tags=["Prediction"],
    summary="Phân tích cảm xúc danh sách câu (Batch)",
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
def predict_batch(request: BatchPredictionRequest) -> list[dict[str, Any]]:
    """Dự đoán cảm xúc theo lô cho danh sách nhiều câu đánh giá."""
    try:
        results = get_ready_predictor().predict_batch(request.texts)
        return [res.to_dict() for res in results]
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
