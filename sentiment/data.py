"""Chia development data và khởi tạo PyTorch DataLoaders.

Module này chịu trách nhiệm:
1. Đọc Official Train và tách thành train/validation/calibration.
2. Xây dựng vocabulary từ train (hoặc train + validation ở final fit).
3. Thu thập audit của development split, không đọc Official Test.
4. Định nghĩa IMDBDataset và đóng gói DataBundle chuẩn cho PyTorch.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data_validation import compute_dataset_hash, compute_token_length_distribution, load_dataset
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
    """Gói dữ liệu hoàn chỉnh chứa DataLoaders, Bộ từ vựng và Báo cáo kiểm toán.

    Attributes:
        train (DataLoader): DataLoader cho tập huấn luyện (shuffled).
        validation (DataLoader): DataLoader cho tập kiểm định (not shuffled).
        calibration (DataLoader): DataLoader chỉ dành cho temperature/policy.
        vocabulary (Vocabulary): Bộ từ vựng fit theo scope của lifecycle hiện tại.
        sizes (dict[str, int]): Kích thước của các split development.
        audit (dict[str, Any]): Hash, token distribution và OOV của development.
    """

    train: DataLoader
    validation: DataLoader
    calibration: DataLoader
    vocabulary: Vocabulary
    sizes: dict[str, int]
    audit: dict[str, Any] = field(default_factory=dict)
    # Chỉ dùng bởi compatibility path ba đối số; lifecycle mới không tạo test loader.
    test: DataLoader | None = None


def calculate_split_oov_rate(texts: list[str], vocabulary: Vocabulary) -> float:
    """Tính OOV rate của một split development."""
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
    """Tách Official Train thành train/validation/calibration có stratify.

    Với 25.000 mẫu và cấu hình mặc định, ba split có kích thước 20.000, 2.500
    và 2.500. Official Test không được truyền vào hàm này.
    """
    if len(source_frame) < 3:
        raise ValueError("Official Train cần ít nhất 3 mẫu để tạo ba split.")

    # Chuyển tỷ lệ sang số lượng nguyên để cả smoke dataset nhỏ cũng tạo được
    # đủ ba vai trò (mỗi split tối thiểu một mẫu), trong khi dataset chính vẫn
    # giữ đúng kích thước kỳ vọng 80/10/10.
    validation_count = max(1, round(len(source_frame) * config.validation_size))
    calibration_count = max(1, round(len(source_frame) * config.calibration_size))
    while validation_count + calibration_count >= len(source_frame):
        if validation_count >= calibration_count and validation_count > 1:
            validation_count -= 1
        elif calibration_count > 1:
            calibration_count -= 1
        else:
            raise ValueError("Official Train quá nhỏ để tạo train/validation/calibration.")
    holdout_count = validation_count + calibration_count

    def stratify_or_none(frame: pd.DataFrame, test_size: int) -> pd.Series | None:
        """Dùng stratify khi kích thước nhỏ vẫn đủ mẫu cho mỗi lớp ở hai phía."""
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
        frame.reset_index(drop=True)
        for frame in (train_frame, validation_frame, calibration_frame)
    )  # type: ignore[return-value]


def _make_loader(
    frame: pd.DataFrame,
    vocabulary: Vocabulary,
    config: ExperimentConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    """Tạo DataLoader theo đúng preprocessing contract của config."""
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


def _build_audit(
    frames: dict[str, pd.DataFrame],
    vocabulary: Vocabulary,
    config: ExperimentConfig,
    *,
    source_train_hash: str,
    vocabulary_scope: str,
) -> dict[str, Any]:
    """Tạo audit từ development split, không tính bất kỳ thống kê test nào."""
    return {
        "source_train_hash": source_train_hash,
        "split_hashes": {name: compute_dataset_hash(frame) for name, frame in frames.items()},
        "train_data_hash": compute_dataset_hash(frames["train"]),
        "validation_data_hash": compute_dataset_hash(frames["validation"]),
        "calibration_data_hash": compute_dataset_hash(frames["calibration"]),
        "vocabulary_hash": vocabulary.compute_hash(),
        "vocabulary_scope": vocabulary_scope,
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


def _create_development_bundle(train_path: str | Path, config: ExperimentConfig) -> DataBundle:
    """Tạo bundle chuẩn train/validation/calibration từ Official Train."""
    source_train = load_dataset(train_path)
    train_frame, validation_frame, calibration_frame = split_development_frame(
        source_train, config
    )
    vocabulary = build_vocabulary(
        train_frame["text"], config.min_frequency, config.max_vocabulary_size
    )
    frames = {
        "train": train_frame,
        "validation": validation_frame,
        "calibration": calibration_frame,
    }
    audit = _build_audit(
        frames,
        vocabulary,
        config,
        source_train_hash=compute_dataset_hash(source_train),
        vocabulary_scope="train_only",
    )
    return DataBundle(
        train=_make_loader(train_frame, vocabulary, config, shuffle=True),
        validation=_make_loader(validation_frame, vocabulary, config, shuffle=False),
        calibration=_make_loader(calibration_frame, vocabulary, config, shuffle=False),
        vocabulary=vocabulary,
        sizes={name: len(frame) for name, frame in frames.items()},
        audit=audit,
    )


def create_data_bundle(
    train_path: str | Path,
    config_or_legacy_test_path: ExperimentConfig | str | Path,
    legacy_config: ExperimentConfig | None = None,
) -> DataBundle:
    """Tạo DataBundle development.

    Cách gọi chuẩn là ``create_data_bundle(train_path, config)``. Chữ ký cũ
    ``create_data_bundle(train_path, test_path, config)`` chỉ là compatibility
    path cho notebook cũ; CLI mới không truyền test path và không tạo test loader.

    Nhánh compatibility cũ vẫn trả thêm ``test`` nhưng không được dùng trong
    training/release code.
    """
    if legacy_config is None:
        if not isinstance(config_or_legacy_test_path, ExperimentConfig):
            raise TypeError("Cách gọi chuẩn yêu cầu create_data_bundle(train_path, config).")
        return _create_development_bundle(train_path, config_or_legacy_test_path)

    # Nhánh cũ chỉ phục vụ notebook/test cũ; không dùng trong training lifecycle mới.
    from .data_validation import remove_train_test_overlap

    test_path = config_or_legacy_test_path
    config = legacy_config
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
        calibration=make_loader(validation_frame.iloc[0:0].copy(), False),
        test=make_loader(test_frame, False),
        vocabulary=vocabulary,
        sizes={
            "train": len(train_frame),
            "validation": len(validation_frame),
            "calibration": 0,
            "test": len(test_frame),
        },
        audit=audit_report,
    )


def create_final_fit_bundle(
    train_path: str | Path, config: ExperimentConfig
) -> tuple[DataBundle, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tạo final-fit bundle với vocabulary fit trên train + validation.

    Calibration được giữ riêng để chỉ dùng cho Temperature Scaling và policy,
    tuyệt đối không đi vào gradient hoặc vocabulary.
    """
    source_train = load_dataset(train_path)
    train_frame, validation_frame, calibration_frame = split_development_frame(
        source_train, config
    )
    final_fit_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
    vocabulary = build_vocabulary(
        final_fit_frame["text"], config.min_frequency, config.max_vocabulary_size
    )
    frames = {
        "train": final_fit_frame,
        "validation": validation_frame,
        "calibration": calibration_frame,
    }
    audit = _build_audit(
        frames,
        vocabulary,
        config,
        source_train_hash=compute_dataset_hash(source_train),
        vocabulary_scope="train_plus_validation",
    )
    bundle = DataBundle(
        train=_make_loader(final_fit_frame, vocabulary, config, shuffle=True),
        validation=_make_loader(validation_frame, vocabulary, config, shuffle=False),
        calibration=_make_loader(calibration_frame, vocabulary, config, shuffle=False),
        vocabulary=vocabulary,
        sizes={name: len(frame) for name, frame in frames.items()},
        audit=audit,
    )
    return bundle, train_frame, validation_frame, calibration_frame
