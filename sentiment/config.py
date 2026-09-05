"""Cấu hình tập trung cho toàn bộ thí nghiệm phân loại cảm xúc.

Module này định nghĩa lớp ExperimentConfig quản lý toàn bộ siêu tham số
huấn luyện, tiền xử lý, chiến lược cắt chuỗi, hiệu chuẩn xác suất và kiến trúc mô hình RNN.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

# Kiểu kiến trúc mô hình được hỗ trợ
ModelType = Literal["lstm", "gru", "bilstm"]
TruncationType = Literal["first", "head_tail"]


@dataclass(slots=True)
class ExperimentConfig:
    """Cấu hình siêu tham số thí nghiệm có thể tái sử dụng và lưu cùng checkpoint.

    Attributes:
        model_type (ModelType): Loại kiến trúc RNN ('lstm', 'gru', 'bilstm'). Mặc định là 'bilstm'.
        seed (int): Hạt giống ngẫu nhiên để đảm bảo khả năng tái lập kết quả. Mặc định là 42.
        validation_size (float): Tỷ lệ validation từ dữ liệu train. Mặc định 0.2.
        min_frequency (int): Tần suất tối thiểu của từ để đưa vào từ điển. Mặc định là 5.
        max_vocabulary_size (int): Kích thước tối đa của bộ từ vựng. Mặc định là 50,000.
        max_length (int): Độ dài chuỗi tối đa sau khi pad/truncate. Mặc định là 256.
        truncation_strategy (TruncationType): Chiến lược cắt chuỗi ('first' hoặc 'head_tail'). Mặc định 'first'.
        batch_size (int): Kích thước lô (batch size) khi huấn luyện. Mặc định là 128.
        embedding_dim (int): Số chiều của không gian nhúng từ (word embedding). Mặc định là 128.
        hidden_dim (int): Số chiều ẩn của lớp RNN. Mặc định là 128.
        num_layers (int): Số lớp RNN xếp chồng (stacked layers). Mặc định là 2.
        dropout (float): Tỷ lệ dropout phòng chống overfitting. Mặc định là 0.4.
        learning_rate (float): Tốc độ học của thuật toán tối ưu (AdamW). Mặc định là 1e-3.
        weight_decay (float): Hệ số L2 regularization cho AdamW. Mặc định là 1e-5.
        epochs (int): Số epoch huấn luyện tối đa. Mặc định là 15.
        patience (int): Số epoch chờ trước khi dừng sớm. Mặc định là 3.
        num_workers (int): Số tiến trình nạp dữ liệu cho PyTorch DataLoader. Mặc định là 0.
        temperature (float | None): Hệ số hiệu chuẩn Temperature Scaling (fit trên validation). Mặc định None.
        decision_threshold (float): Ngưỡng phân loại nhị phân. Mặc định là 0.5.
        uncertain_lower (float): Ranh giới dưới của vùng bất định. Mặc định là 0.40.
        uncertain_upper (float): Ranh giới trên của vùng bất định. Mặc định là 0.60.
    """

    model_type: ModelType = "bilstm"
    seed: int = 42
    validation_size: float = 0.2
    min_frequency: int = 5
    max_vocabulary_size: int = 50_000
    max_length: int = 256
    truncation_strategy: TruncationType = "first"
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
    uncertain_lower: float = 0.40
    uncertain_upper: float = 0.60

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của các siêu tham số sau khi khởi tạo."""
        if self.model_type not in {"lstm", "gru", "bilstm"}:
            raise ValueError(
                f"Kiến trúc mô hình '{self.model_type}' không hợp lệ. "
                "Chấp nhận: 'lstm', 'gru', 'bilstm'."
            )
        if self.truncation_strategy not in {"first", "head_tail"}:
            raise ValueError(
                f"Chiến lược cắt chuỗi '{self.truncation_strategy}' không hợp lệ. "
                "Chấp nhận: 'first', 'head_tail'."
            )
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size phải nằm trong khoảng (0, 1).")

        positive_values = {
            "min_frequency": self.min_frequency,
            "max_vocabulary_size": self.max_vocabulary_size,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
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

        if not (0.0 <= self.uncertain_lower < self.uncertain_upper <= 1.0):
            raise ValueError("Vùng bất định yêu cầu: 0.0 <= uncertain_lower < uncertain_upper <= 1.0.")

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi cấu hình thành dictionary để lưu vào JSON hoặc Checkpoint."""
        return asdict(self)
