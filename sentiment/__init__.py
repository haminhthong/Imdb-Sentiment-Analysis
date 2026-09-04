"""Gói phân loại cảm xúc đánh giá phim IMDB.

Các thành phần không được import sẵn để những tiện ích nhẹ như tokenizer vẫn dùng
được trong môi trường chưa cài PyTorch.
"""

from .config import ExperimentConfig

__all__ = ["ExperimentConfig"]
