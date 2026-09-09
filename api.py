"""REST API phục vụ phân loại cảm xúc bằng FastAPI và PyTorch.

Được thiết kế theo định hướng kỹ thuật chuẩn mực (Production-Oriented NLP Portfolio System):
- Endpoints chuẩn: `/health`, `/ready`, `/metrics`, `/info`, `/predict`, `/predict/batch`.
- Schema Version 3: Tách bạch giữa calibrated `positive_probability` và raw `positive_score`.
- Decision policy: Cung cấp nhãn `decision` ('accepted' / 'review_required') và cờ `uncertain`.
- Reliability warnings: Trả về metadata số token, tỷ lệ OOV, cờ cắt chuỗi và cảnh báo ngôn ngữ.
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

from sentiment.inference import Predictor, load_predictor

APP_NAME = "CineSentiment AI REST API"
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "artifacts/releases/v1.0.0/model.pt")
API_KEY = os.getenv("API_KEY")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
if RATE_LIMIT_PER_MINUTE <= 0:
    raise ValueError("RATE_LIMIT_PER_MINUTE phải lớn hơn 0.")

REQUEST_COUNTS = {
    "total": 0,
    "errors": 0,
    "latency_seconds": 0.0,
    "accepted_predictions": 0,
    "review_required": 0,
    "positive_predictions": 0,
    "negative_predictions": 0,
    "truncated_inputs": 0,
    "high_oov_inputs": 0,
}
LATENCY_SAMPLES: deque[float] = deque(maxlen=10_000)
METRICS_LOCK = threading.Lock()
RATE_LIMIT_LOCK = threading.Lock()
REQUEST_TIMESTAMPS: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(
    title=APP_NAME,
    description=(
        "REST API phục vụ phân loại cảm xúc đánh giá phim tiếng Anh "
        "dưới kiến trúc production-oriented NLP portfolio system với xác suất hiệu chuẩn và kiểm toán đầu vào."
    ),
    version="2.0.0",
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
    """Schema kết quả dự đoán cảm xúc đã hiệu chuẩn và báo cáo kiểm toán."""

    label: str = Field(description="Nhãn dự đoán: 'Positive' hoặc 'Negative'.")
    positive_probability: float = Field(
        description="Xác suất thuộc lớp Positive đã qua Temperature Scaling hiệu chuẩn."
    )
    confidence: float = Field(description="Độ tin cậy của dự đoán (max(p, 1-p)).")
    positive_score: float = Field(description="Điểm sigmoid thô chưa qua hiệu chuẩn.")
    decision: str = Field(description="Quyết định hệ thống: 'accepted' hoặc 'review_required'.")
    uncertain: bool = Field(description="Cờ báo hiệu dự đoán rơi vào vùng bất định.")
    input_tokens: int = Field(description="Tổng số token của văn bản gốc.")
    used_tokens: int = Field(description="Số token thực tế được đưa vào mô hình.")
    truncated: bool = Field(description="Cờ đánh dấu văn bản bị cắt do vượt max_length.")
    input_oov_rate: float = Field(description="OOV rate trên toàn bộ input.")
    used_oov_rate: float = Field(description="OOV rate trên token model thực sự sử dụng.")
    oov_rate: float = Field(description="Alias tương thích, bằng used_oov_rate.")
    warnings: list[str] = Field(default_factory=list, description="Danh sách cảnh báo độ tin cậy.")
    model_version: str = Field(description="Phiên bản model bundle đang phục vụ.")
    tokenizer_version: str = Field(description="Phiên bản text contract.")


class ModelInfoResponse(BaseModel):
    """Schema thông tin mô hình đang nạp."""

    model_type: str
    total_parameters: int
    vocabulary_size: int
    max_length: int
    temperature: float
    decision_threshold: float
    confidence_threshold: float
    model_version: str
    tokenizer_version: str
    checkpoint_path: str


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Nạp model.pt hoặc model.joblib vào RAM một lần duy nhất."""
    return load_predictor(CHECKPOINT_PATH, device="cpu")


def get_ready_predictor() -> Predictor:
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
            elapsed = time.perf_counter() - started
            REQUEST_COUNTS["latency_seconds"] += elapsed
            LATENCY_SAMPLES.append(elapsed)
        raise

    elapsed = time.perf_counter() - started
    with METRICS_LOCK:
        REQUEST_COUNTS["total"] += 1
        REQUEST_COUNTS["errors"] += int(response.status_code >= 500)
        REQUEST_COUNTS["latency_seconds"] += elapsed
        LATENCY_SAMPLES.append(elapsed)
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
        counters = {
            key: REQUEST_COUNTS[key]
            for key in (
                "accepted_predictions",
                "review_required",
                "positive_predictions",
                "negative_predictions",
                "truncated_inputs",
                "high_oov_inputs",
            )
        }
        latency_samples = sorted(LATENCY_SAMPLES)

    def percentile(values: list[float], ratio: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, int((len(values) - 1) * ratio))
        return values[index]

    body = (
        "# TYPE cinesentiment_requests_total counter\n"
        f"cinesentiment_requests_total {total}\n"
        "# TYPE cinesentiment_errors_total counter\n"
        f"cinesentiment_errors_total {errors}\n"
        "# TYPE cinesentiment_request_latency_seconds_total counter\n"
        f"cinesentiment_request_latency_seconds_total {latency:.6f}\n"
        + "".join(
            f"# TYPE cinesentiment_{key}_total counter\ncinesentiment_{key}_total {value}\n"
            for key, value in counters.items()
        )
        + "# TYPE cinesentiment_request_latency_seconds gauge\n"
        f"cinesentiment_request_latency_seconds_p50 {percentile(latency_samples, 0.50):.6f}\n"
        f"cinesentiment_request_latency_seconds_p95 {percentile(latency_samples, 0.95):.6f}\n"
    )
    return Response(body, media_type="text/plain; version=0.0.4")


@app.get("/ready", tags=["System"], summary="Kiểm tra sẵn sàng suy luận")
def readiness() -> dict[str, Any]:
    """Kiểm tra mô hình đã được nạp thành công vào RAM chưa."""
    predictor = get_ready_predictor()
    return {
        "status": "ready",
        "model_type": predictor.config.model_type,
        "temperature": predictor.temperature,
        "checkpoint": str(predictor.checkpoint_path),
    }


@app.get(
    "/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Lấy thông tin cấu hình mô hình",
)
def model_info() -> dict[str, Any]:
    """Lấy thông tin siêu tham số, nhiệt độ hiệu chuẩn và số tham số mô hình."""
    predictor = get_ready_predictor()
    return {
        "model_type": predictor.config.model_type,
        "total_parameters": predictor.model.count_parameters(),
        "vocabulary_size": len(predictor.vocabulary),
        "max_length": predictor.config.max_length,
        "temperature": predictor.temperature,
        "decision_threshold": predictor.decision_threshold,
        "confidence_threshold": predictor.confidence_threshold,
        "model_version": predictor.model_version,
        "tokenizer_version": predictor.tokenizer_version,
        "checkpoint_path": str(predictor.checkpoint_path),
    }


def record_prediction_metrics(results: list[dict[str, Any]]) -> None:
    """Ghi metric chất lượng model, không ghi raw review text."""
    with METRICS_LOCK:
        for result in results:
            REQUEST_COUNTS["accepted_predictions"] += int(result["decision"] == "accepted")
            REQUEST_COUNTS["review_required"] += int(result["decision"] == "review_required")
            REQUEST_COUNTS["positive_predictions"] += int(result["label"] == "Positive")
            REQUEST_COUNTS["negative_predictions"] += int(result["label"] == "Negative")
            REQUEST_COUNTS["truncated_inputs"] += int(result["truncated"])
            REQUEST_COUNTS["high_oov_inputs"] += int(result["used_oov_rate"] > 0.20)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Phân tích cảm xúc câu đơn",
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Dự đoán cảm xúc cho một câu đánh giá phim kèm độ tin cậy và kiểm toán."""
    if not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="EMPTY_TEXT: Nội dung đánh giá không được để trống hoặc toàn khoảng trắng.",
        )
    try:
        result = get_ready_predictor().predict(request.text)
        payload = result.to_dict()
        record_prediction_metrics([payload])
        return payload
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
    if any(not text.strip() for text in request.texts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="EMPTY_TEXT: Mỗi mẫu đánh giá trong batch phải là chuỗi không rỗng.",
        )
    try:
        results = get_ready_predictor().predict_batch(request.texts)
        payload = [res.to_dict() for res in results]
        record_prediction_metrics(payload)
        return payload
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


def run() -> None:
    """Entry point ``cinesentiment-api`` cho uvicorn."""
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
