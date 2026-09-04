"""Các hàm tiện ích hệ thống: Cố định seed tái lập kết quả, chọn device và đo độ trễ.

Module này cung cấp:
1. `seed_everything`: Đặt hạt giống ngẫu nhiên cho random, numpy và PyTorch.
2. `select_device`: Tự động nhận diện CPU hoặc GPU CUDA khả dụng.
3. `measure_latency`: Đo thời gian thực thi suy luận trung bình của mô hình (Inference Latency).
"""

import random
import time
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Cố định seed ngẫu nhiên cho toàn bộ thư viện để tái lập kết quả thí nghiệm.

    Args:
        seed (int): Giá trị seed số nguyên (ví dụ: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Đảm bảo cuDNN chạy chế độ deterministic (tái lập tối đa)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str = "auto") -> torch.device:
    """Tự động lựa chọn hoặc kiểm tra tính hợp lệ của thiết bị tính toán PyTorch.

    Args:
        requested (str): 'auto', 'cpu', hoặc 'cuda'. Mặc định là 'auto'.

    Returns:
        torch.device: Thiết bị PyTorch được khởi tạo.

    Raises:
        RuntimeError: Nếu yêu cầu CUDA nhưng hệ thống không có GPU tương thích.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Đã chỉ định CUDA nhưng không tìm thấy GPU khả dụng trên hệ thống."
        )
    return torch.device(requested)


def measure_latency(predictor: Any, sample_text: str, runs: int = 50) -> float:
    """Đo độ trễ suy luận trung bình (milliseconds) của mô hình trên một câu mẫu.

    Args:
        predictor (Any): Đối tượng SentimentPredictor.
        sample_text (str): Câu văn bản mẫu dùng để suy luận thử nghiệm.
        runs (int): Số lần chạy lặp để lấy trung bình. Mặc định là 50.

    Returns:
        float: Độ trễ suy luận trung bình tính bằng miligiây (ms).
    """
    # Khởi động (Warm-up)
    for _ in range(5):
        _ = predictor.predict(sample_text)

    start_time = time.perf_counter()
    for _ in range(runs):
        _ = predictor.predict(sample_text)
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / runs) * 1000.0
    return float(avg_ms)
