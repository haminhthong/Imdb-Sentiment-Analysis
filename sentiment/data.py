"""Nạp dữ liệu, kiểm tra tính hợp lệ và khởi tạo PyTorch DataLoaders.

Module này chịu trách nhiệm:
1. Đọc và làm sạch dữ liệu CSV với cơ chế chống rò rỉ đa tầng (Raw + Normalized hash).
2. Tách tập Train/Validation bảo toàn tỷ lệ nhãn (Stratified Split).
3. Xây dựng Vocabulary DUY NHẤT từ tập Train.
4. Thu thập các chỉ số kiểm toán dữ liệu (Token Length Distribution, OOV Rates, Hashes).
5. Định nghĩa IMDBDataset và đóng gói DataBundle chuẩn cho PyTorch.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data_validation import (
    compute_dataset_hash,
    compute_token_length_distribution,
    load_dataset,
    remove_train_test_overlap,
)
from .text import (
    TruncationStrategy,
    Vocabulary,
    build_vocabulary,
    encode_and_pad,
    tokenize,
)


class IMDBDataset(Dataset):
    """PyTorch Dataset nạp và mã hóa văn bản theo nhu cầu (On-the-fly Encoding).

    Mã hóa văn bản khi truy cập item giúp tiết kiệm bộ nhớ RAM đáng kể so với
    việc lưu trước toàn bộ mảng tensor mã hóa.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: Vocabulary,
        max_length: int,
        strategy: TruncationStrategy = "first",
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
    """Gói dữ liệu hoàn chỉnh chứa DataLoaders, Bộ từ vựng và Báo cáo kiểm toán.

    Attributes:
        train (DataLoader): DataLoader cho tập huấn luyện (shuffled).
        validation (DataLoader): DataLoader cho tập kiểm định (not shuffled).
        test (DataLoader): DataLoader cho tập đánh giá độc lập (not shuffled).
        vocabulary (Vocabulary): Bộ từ vựng được xây chỉ từ tập train.
        sizes (dict[str, int]): Số lượng mẫu của từng tập ('train', 'validation', 'test').
        audit (dict[str, Any]): Thống kê chất lượng dữ liệu, độ dài token, OOV rates và hashes.
    """

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    vocabulary: Vocabulary
    sizes: dict[str, int]
    audit: dict[str, Any] = field(default_factory=dict)


def calculate_split_oov_rate(texts: list[str], vocabulary: Vocabulary) -> float:
    """Tính tỷ lệ token OOV trên toàn bộ một tập văn bản."""
    total_tokens = 0
    oov_tokens = 0
    for text in texts:
        tokens = tokenize(text)
        total_tokens += len(tokens)
        oov_tokens += sum(1 for t in tokens if t not in vocabulary.token_to_index)
    return round(oov_tokens / total_tokens, 4) if total_tokens > 0 else 0.0


def create_data_bundle(
    train_path: str | Path, test_path: str | Path, config: ExperimentConfig
) -> DataBundle:
    """Đọc dữ liệu, chống rò rỉ, chia train/validation và tạo DataBundle hoàn chỉnh.

    Quy trình thực hiện:
    1. Đọc tệp train CSV và test CSV, loại trùng lặp exact & normalized hash.
    2. Loại bỏ các mẫu test có nội dung đã xuất hiện trong tập train (Anti-leakage).
    3. Chia tập train thành train/validation theo tỷ lệ phân bố nhãn (Stratified Split).
    4. Xây dựng bộ từ vựng Vocabulary CHỈ từ tập train.
    5. Tính toán các chỉ số audit: phân phối độ dài token, OOV rate từng tập, data hash.
    6. Đóng gói thành các DataLoader thích hợp với GPU memory pinning nếu có CUDA.
    """
    source_train = load_dataset(train_path)
    test_frame = load_dataset(test_path)

    test_frame = remove_train_test_overlap(source_train, test_frame)

    # Chia train / validation có bảo toàn tỷ lệ nhãn (Stratified Split)
    train_frame, validation_frame = train_test_split(
        source_train,
        test_size=config.validation_size,
        random_state=config.seed,
        stratify=source_train["label"],
    )

    # Xây dựng từ điển duy nhất từ tập train
    vocabulary = build_vocabulary(
        train_frame["text"], config.min_frequency, config.max_vocabulary_size
    )

    # Thu thập thống kê kiểm toán dữ liệu (Data Quality Audit)
    truncation_strategy = getattr(config, "truncation_strategy", "first")
    audit_report = {
        "train_data_hash": compute_dataset_hash(train_frame),
        "validation_data_hash": compute_dataset_hash(validation_frame),
        "test_data_hash": compute_dataset_hash(test_frame),
        "vocabulary_hash": vocabulary.compute_hash(),
        "vocabulary_size": len(vocabulary),
        "truncation_strategy": truncation_strategy,
        "token_length_distribution": {
            "train": compute_token_length_distribution(train_frame["text"], config.max_length),
            "validation": compute_token_length_distribution(validation_frame["text"], config.max_length),
            "test": compute_token_length_distribution(test_frame["text"], config.max_length),
        },
        "oov_rates": {
            "train": calculate_split_oov_rate(train_frame["text"].tolist(), vocabulary),
            "validation": calculate_split_oov_rate(validation_frame["text"].tolist(), vocabulary),
            "test": calculate_split_oov_rate(test_frame["text"].tolist(), vocabulary),
        },
    }

    def make_loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        return DataLoader(
            IMDBDataset(
                frame.reset_index(drop=True),
                vocabulary,
                config.max_length,
                strategy=truncation_strategy,
            ),
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return DataBundle(
        train=make_loader(train_frame, True),
        validation=make_loader(validation_frame, False),
        test=make_loader(test_frame, False),
        vocabulary=vocabulary,
        sizes={
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        audit=audit_report,
    )
