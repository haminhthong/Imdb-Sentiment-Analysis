"""Nạp checkpoint mô hình và phục vụ suy luận (Inference).

Module này hỗ trợ suy luận dự đoán câu đơn lẻ hoặc dự đoán theo lô (batch),
được chia sẻ dùng chung cho các giao diện CLI (`predict.py`), REST API (`api.py`),
và Web App Streamlit (`app.py`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .model import SentimentRNN
from .text import Vocabulary, encode_and_pad


@dataclass(frozen=True)
class PredictionResult:
    """Kết quả dự đoán cảm xúc của một câu đánh giá.

    Attributes:
        label (str): Nhãn cảm xúc dự đoán ('Positive' hoặc 'Negative').
        positive_probability (float): Xác suất câu thuộc nhãn Positive (từ 0.0 đến 1.0).
        confidence (float): Độ tin cậy của dự đoán (max(p, 1-p)).
    """

    label: str
    positive_probability: float
    confidence: float

    def to_dict(self) -> dict[str, str | float]:
        """Chuyển đổi đối tượng kết quả thành Dictionary dạng JSON.

        Returns:
            dict[str, str | float]: Dictionary chứa 'label', 'positive_probability', 'confidence'.
        """
        return {
            "label": self.label,
            "positive_probability": self.positive_probability,
            "confidence": self.confidence,
        }


class SentimentPredictor:
    """Lớp đóng gói mô hình RNN và bộ từ vựng để dự đoán nhanh và an toàn.

    Attributes:
        checkpoint_path (Path): Đường dẫn đến tệp checkpoint PyTorch (`.pt`).
        device (torch.device): Thiết bị tính toán chạy suy luận.
        config (ExperimentConfig): Cấu hình siêu tham số khôi phục từ checkpoint.
        vocabulary (Vocabulary): Bộ từ vựng khôi phục từ checkpoint.
        model (SentimentRNN): Mô hình SentimentRNN ở chế độ `eval()`.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        """Nạp trọng số mô hình và cấu hình từ tệp checkpoint PyTorch.

        Args:
            checkpoint_path (str | Path): Đường dẫn tệp checkpoint `.pt`.
            device (str): Thiết bị tính toán ('cpu', 'cuda', hoặc 'auto').

        Raises:
            FileNotFoundError: Nếu không tìm thấy tệp checkpoint.
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy tệp checkpoint: {self.checkpoint_path}. "
                "Vui lòng huấn luyện mô hình trước bằng lệnh `python train.py`."
            )

        self.device = torch.device(device)
        checkpoint: dict[str, Any] = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            # Checkpoint chỉ chứa tensor và kiểu dữ liệu cơ bản; không cho phép unpickle tùy ý.
            weights_only=True,
        )

        self.config = ExperimentConfig(**checkpoint["config"])
        self.vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])

        self.model = SentimentRNN(
            len(self.vocabulary), self.vocabulary.pad_index, self.config
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, text: str) -> PredictionResult:
        """Dự đoán cảm xúc của một câu đánh giá đơn lẻ.

        Args:
            text (str): Nội dung câu đánh giá.

        Returns:
            PredictionResult: Kết quả nhãn, xác suất tích cực và độ tin cậy.
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Dự đoán danh sách nhiều câu đánh giá theo lô (Batch Processing).

        Args:
            texts (list[str]): Danh sách các câu đánh giá.

        Returns:
            list[PredictionResult]: Danh sách các kết quả dự đoán tương ứng.

        Raises:
            ValueError: Nếu danh sách rỗng hoặc chứa câu không phải string / chuỗi rỗng.
        """
        if not texts:
            raise ValueError("Danh sách câu đánh giá đầu vào không được để trống.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Mỗi mẫu đánh giá phải là một chuỗi văn bản không rỗng.")

        encoded_batch, length_batch = zip(
            *(
                encode_and_pad(text, self.vocabulary, self.config.max_length)
                for text in texts
            )
        )

        tokens = torch.tensor(encoded_batch, dtype=torch.long, device=self.device)
        lengths = torch.tensor(length_batch, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            logits = self.model(tokens, lengths)
            probabilities = torch.sigmoid(logits).cpu().tolist()

        results: list[PredictionResult] = []
        for probability in probabilities:
            label = "Positive" if probability >= 0.5 else "Negative"
            confidence = probability if label == "Positive" else 1.0 - probability
            results.append(PredictionResult(label, probability, confidence))

        return results
