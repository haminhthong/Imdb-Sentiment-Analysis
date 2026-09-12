"""Tách tập dữ liệu phát triển và khởi tạo PyTorch DataLoaders.

Nguyên tắc cốt lõi:
1. Đọc Official Train và tách thành 3 tập có phân tầng (Stratified):
   - Train (80%): Dùng để cập nhật gradient mô hình.
   - Validation (10%): Dùng để theo dõi early stopping và so sánh mô hình.
   - Calibration (10%): Dùng riêng để học hệ số Temperature Scaling.
2. Từ điển (Vocabulary) CHỈ được xây dựng từ tập Train để chống rò rỉ (Train-only vocabulary).
3. Đóng gói DataBundle chuẩn cho huấn luyện PyTorch.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data_validation import compute_token_length_distribution, load_dataset
from .text import (
    TruncationStrategy,
    Vocabulary,
    build_vocabulary,
    encode_and_pad,
    tokenize,
)


class IMDBDataset(Dataset):
    """PyTorch Dataset mã hóa văn bản theo nhu cầu (On-the-fly Encoding)."""

    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: Vocabulary,
        max_length: int,
        strategy: TruncationStrategy = "head_tail",
    ):
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].tolist()
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.strategy = strategy

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, length = encode_and_pad(
            self.texts[index],
            self.vocabulary,
            self.max_length,
            strategy=self.strategy,
        )
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


@dataclass
class DataBundle:
    """Gói dữ liệu hoàn chỉnh chứa DataLoaders, bộ từ vựng và thống kê kiểm toán."""

    train: DataLoader
    validation: DataLoader
    calibration: DataLoader
    vocabulary: Vocabulary
    sizes: dict[str, int]
    audit: dict[str, Any] = field(default_factory=dict)


def calculate_split_oov_rate(texts: list[str], vocabulary: Vocabulary) -> float:
    """Tính tỷ lệ từ ngoài từ điển (OOV Rate) của một split dữ liệu."""
    total_tokens = 0
    oov_tokens = 0
    for text in texts:
        tokens = tokenize(text)
        total_tokens += len(tokens)
        oov_tokens += sum(1 for t in tokens if t not in vocabulary.token_to_index)
    return round(oov_tokens / total_tokens, 4) if total_tokens > 0 else 0.0


def split_development_frame(
    source_frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tách Official Train thành train / validation / calibration có phân tầng (Stratified)."""
    if len(source_frame) < 3:
        raise ValueError("Official Train cần ít nhất 3 mẫu để tạo ba split.")

    validation_count = max(1, round(len(source_frame) * config.validation_size))
    calibration_count = max(1, round(len(source_frame) * config.calibration_size))
    while validation_count + calibration_count >= len(source_frame):
        if validation_count >= calibration_count and validation_count > 1:
            validation_count -= 1
        elif calibration_count > 1:
            calibration_count -= 1
        else:
            raise ValueError("Official Train quá nhỏ để chia train/validation/calibration.")
    holdout_count = validation_count + calibration_count

    def stratify_or_none(frame: pd.DataFrame, test_size: int) -> pd.Series | None:
        class_count = int(frame["label"].nunique())
        train_count = len(frame) - test_size
        if class_count >= 2 and test_size >= class_count and train_count >= class_count:
            return frame["label"]
        return None

    train_frame, holdout_frame = train_test_split(
        source_frame,
        test_size=holdout_count,
        random_state=config.seed,
        stratify=stratify_or_none(source_frame, holdout_count),
    )
    validation_frame, calibration_frame = train_test_split(
        holdout_frame,
        test_size=calibration_count,
        random_state=config.seed,
        stratify=stratify_or_none(holdout_frame, calibration_count),
    )
    return tuple(
        frame.reset_index(drop=True) for frame in (train_frame, validation_frame, calibration_frame)
    )  # type: ignore[return-value]


def _make_loader(
    frame: pd.DataFrame,
    vocabulary: Vocabulary,
    config: ExperimentConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    """Tạo DataLoader theo cấu hình của ExperimentConfig."""
    return DataLoader(
        IMDBDataset(
            frame.reset_index(drop=True),
            vocabulary,
            config.max_length,
            strategy=config.truncation_strategy,
        ),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_data_bundle(
    train_path: str | Path,
    config: ExperimentConfig,
    max_samples: int | None = None,
) -> DataBundle:
    """Tạo DataBundle gồm train, validation, calibration và từ điển train-only."""
    source_train = load_dataset(train_path)
    if max_samples is not None and max_samples > 0:
        source_train = source_train.head(max_samples)
    train_frame, validation_frame, calibration_frame = split_development_frame(source_train, config)

    # Từ điển CHỈ được xây dựng từ tập Train
    vocabulary = build_vocabulary(
        train_frame["text"], config.min_frequency, config.max_vocabulary_size
    )

    frames = {
        "train": train_frame,
        "validation": validation_frame,
        "calibration": calibration_frame,
    }

    audit = {
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "calibration_samples": len(calibration_frame),
        "seed": config.seed,
        "vocabulary_size": len(vocabulary),
        "truncation_strategy": config.truncation_strategy,
        "token_length_distribution": {
            name: compute_token_length_distribution(frame["text"], config.max_length)
            for name, frame in frames.items()
        },
        "oov_rates": {
            name: calculate_split_oov_rate(frame["text"].tolist(), vocabulary)
            for name, frame in frames.items()
        },
    }

    return DataBundle(
        train=_make_loader(train_frame, vocabulary, config, shuffle=True),
        validation=_make_loader(validation_frame, vocabulary, config, shuffle=False),
        calibration=_make_loader(calibration_frame, vocabulary, config, shuffle=False),
        vocabulary=vocabulary,
        sizes={name: len(frame) for name, frame in frames.items()},
        audit=audit,
    )
