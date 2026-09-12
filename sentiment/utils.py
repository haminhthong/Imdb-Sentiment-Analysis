"""Các hàm tiện ích hệ thống: Cố định seed tái lập kết quả, chọn device và đo độ trễ."""

import contextlib
import random
import sys
import time
from typing import Any

import numpy as np
import torch


def configure_utf8_output() -> None:
    """Đảm bảo sys.stdout và sys.stderr không phát sinh lỗi mã hóa trên Windows console."""
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def seed_everything(seed: int) -> None:
    """Cố định seed ngẫu nhiên cho toàn bộ thư viện để tái lập kết quả thí nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str = "auto") -> torch.device:
    """Tự động lựa chọn hoặc kiểm tra tính hợp lệ của thiết bị tính toán PyTorch."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã chỉ định CUDA nhưng không tìm thấy GPU khả dụng trên hệ thống.")
    return torch.device(requested)


def measure_latency(predictor: Any, sample_text: str, runs: int = 50) -> float:
    """Đo độ trễ suy luận trung bình (milliseconds) của mô hình trên một câu mẫu."""
    for _ in range(5):
        _ = predictor.predict(sample_text)

    start_time = time.perf_counter()
    for _ in range(runs):
        _ = predictor.predict(sample_text)
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / runs) * 1000.0
    return float(avg_ms)
