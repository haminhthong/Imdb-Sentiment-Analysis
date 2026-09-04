"""Nạp dữ liệu, kiểm tra tính hợp lệ và khởi tạo PyTorch DataLoaders.

Module này chịu trách nhiệm:
1. Đọc và làm sạch dữ liệu CSV (loại bỏ lặp, kiểm tra cột 'text' và 'label').
2. Ngăn ngừa rò rỉ dữ liệu (Data Leakage) bằng cách loại bỏ mẫu test trùng với train.
3. Chia dữ liệu theo tỷ lệ nhãn (Stratified Split) để cân bằng phân phối nhãn.
4. Định nghĩa IMDBDataset và đóng gói DataBundle chuẩn cho PyTorch DataLoader.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data_validation import load_dataset, remove_train_test_overlap
from .text import Vocabulary, build_vocabulary, encode_and_pad


class IMDBDataset(Dataset):
    """PyTorch Dataset nạp và mã hóa văn bản theo nhu cầu (On-the-fly Encoding).

    Mã hóa văn bản khi truy cập item giúp tiết kiệm bộ nhớ RAM đáng kể so với
    việc lưu trước toàn bộ mảng tensor mã hóa.

    Args:
        frame (pd.DataFrame): DataFrame chứa dữ liệu 'text' và 'label'.
        vocabulary (Vocabulary): Bộ từ vựng dùng để mã hóa văn bản.
        max_length (int): Độ dài chuỗi tối đa.
    """

    def __init__(self, frame: pd.DataFrame, vocabulary: Vocabulary, max_length: int):
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].tolist()
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        """Tổng số mẫu trong Dataset."""
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lấy một mẫu dữ liệu theo chỉ số.

        Args:
            index (int): Vị trí của mẫu dữ liệu.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - tokens: Tensor 1D chứa chỉ số từ (torch.long).
                - length: Tensor 0D chứa độ dài thực tế của câu (torch.long).
                - label: Tensor 0D chứa nhãn 0.0 hoặc 1.0 (torch.float32).
        """
        encoded, length = encode_and_pad(
            self.texts[index], self.vocabulary, self.max_length
        )
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


@dataclass
class DataBundle:
    """Gói dữ liệu hoàn chỉnh chứa DataLoaders và Bộ từ vựng.

    Attributes:
        train (DataLoader): DataLoader cho tập huấn luyện (shuffled).
        validation (DataLoader): DataLoader cho tập kiểm định (not shuffled).
        test (DataLoader): DataLoader cho tập đánh giá độc lập (not shuffled).
        vocabulary (Vocabulary): Bộ từ vựng được xây từ tập train.
        sizes (dict[str, int]): Số lượng mẫu của từng tập ('train', 'validation', 'test').
    """

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    vocabulary: Vocabulary
    sizes: dict[str, int]


def create_data_bundle(
    train_path: str | Path, test_path: str | Path, config: ExperimentConfig
) -> DataBundle:
    """Đọc dữ liệu, chống rò rỉ, chia train/validation và tạo DataBundle.

    Quy trình thực hiện:
    1. Đọc tệp train CSV và test CSV.
    2. Loại bỏ các mẫu test có nội dung đã xuất hiện trong tập train (Anti-leakage).
    3. Chia tập train thành train/validation theo tỷ lệ phân bố nhãn (Stratified Split).
    4. Xây dựng bộ từ vựng Vocabulary CHỈ từ tập train.
    5. Đóng gói thành các DataLoader thích hợp với GPU memory pinning nếu có CUDA.

    Args:
        train_path (str | Path): Đường dẫn tệp CSV dữ liệu huấn luyện.
        test_path (str | Path): Đường dẫn tệp CSV dữ liệu kiểm thử.
        config (ExperimentConfig): Cấu hình siêu tham số thí nghiệm.

    Returns:
        DataBundle: Bộ DataLoaders hoàn chỉnh kèm Vocabulary và thống kê số mẫu.
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

    def make_loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        return DataLoader(
            IMDBDataset(frame.reset_index(drop=True), vocabulary, config.max_length),
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
    )
