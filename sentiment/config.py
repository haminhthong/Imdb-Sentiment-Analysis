"""Cấu hình tập trung cho mô hình và quá trình huấn luyện CineSentiment."""

from dataclasses import asdict, dataclass
from typing import Any, Literal

TruncationType = Literal["first", "head_tail"]


@dataclass(slots=True)
class ExperimentConfig:
    """Cấu hình siêu tham số mô hình và huấn luyện.

    Attributes:
        model_type: Loại mô hình (mặc định 'bilstm').
        seed: Hạt giống ngẫu nhiên đảm bảo tính tái lập.
        validation_size: Tỷ lệ tập validation (mặc định 0.1 = 10%).
        calibration_size: Tỷ lệ tập calibration (mặc định 0.1 = 10%).
        min_frequency: Tần suất từ tối thiểu để đưa vào từ điển.
        max_vocabulary_size: Kích thước từ điển tối đa.
        max_length: Độ dài chuỗi tối đa sau khi pad/truncate.
        truncation_strategy: Chiến lược cắt ngắn chuỗi ('first' hoặc 'head_tail').
        batch_size: Kích thước batch huấn luyện.
        embedding_dim: Số chiều vector nhúng từ.
        hidden_dim: Số chiều vector ẩn LSTM.
        num_layers: Số lớp LSTM xếp chồng.
        dropout: Tỷ lệ dropout.
        learning_rate: Tốc độ học của AdamW.
        weight_decay: Trọng số L2 regularization.
        epochs: Số epoch tối đa.
        patience: Số epoch chờ trước khi dừng sớm (Early Stopping).
        num_workers: Số worker nạp dữ liệu.
        temperature: Hệ số Temperature Scaling sau calibration.
        decision_threshold: Ngưỡng quyết định nhị phân (mặc định 0.5).
    """

    model_type: str = "bilstm"
    seed: int = 42
    validation_size: float = 0.1
    calibration_size: float = 0.1
    min_frequency: int = 5
    max_vocabulary_size: int = 50_000
    max_length: int = 256
    truncation_strategy: TruncationType = "head_tail"
    batch_size: int = 128
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 15
    patience: int = 3
    num_workers: int = 0
    temperature: float | None = None
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của các siêu tham số."""
        if self.truncation_strategy not in {"first", "head_tail"}:
            raise ValueError(
                f"Chiến lược cắt chuỗi '{self.truncation_strategy}' không hợp lệ. "
                "Chấp nhận: 'first', 'head_tail'."
            )
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size phải nằm trong khoảng (0, 1).")
        if not 0 < self.calibration_size < 1:
            raise ValueError("calibration_size phải nằm trong khoảng (0, 1).")
        if self.validation_size + self.calibration_size >= 1:
            raise ValueError("Tổng validation_size và calibration_size phải nhỏ hơn 1.")

        positive_values = {
            "min_frequency": self.min_frequency,
            "max_vocabulary_size": self.max_vocabulary_size,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Các tham số sau phải có giá trị dương (> 0): {', '.join(invalid)}")

        if not 0 <= self.dropout < 1:
            raise ValueError("dropout phải nằm trong khoảng [0, 1).")

        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold phải nằm trong khoảng (0, 1).")

    def to_dict(self) -> dict[str, Any]:
        """Chuyển cấu hình thành dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Nạp cấu hình từ dictionary, bỏ qua các trường không còn sử dụng."""
        valid_fields = {f for f in cls.__slots__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
