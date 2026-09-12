"""Phục vụ suy luận phân loại cảm xúc cho câu đơn lẻ hoặc theo lô (Batch)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import torch

from .config import ExperimentConfig
from .model import BiLSTMSentimentClassifier
from .text import (
    Vocabulary,
    detect_language_warning,
    encode_with_audit,
    tokenize,
)


@dataclass(frozen=True)
class PredictionResult:
    """Kết quả dự đoán cảm xúc tinh gọn và chuẩn mực.

    Attributes:
        label: Nhãn cảm xúc ('Positive' hoặc 'Negative').
        probability: Xác suất lớp Positive đã qua Temperature Scaling.
        truncated: Cờ đánh dấu câu bị cắt bớt do vượt max_length.
        oov_rate: Tỷ lệ từ ngoài từ điển (OOV Rate).
        token_count: Tổng số token của văn bản gốc.
        warnings: Danh sách cảnh báo độ tin cậy (nếu có).
    """

    label: str
    probability: float
    truncated: bool = False
    oov_rate: float = 0.0
    token_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi kết quả thành dictionary."""
        return {
            "label": self.label,
            "probability": round(self.probability, 4),
            "truncated": self.truncated,
            "oov_rate": round(self.oov_rate, 4),
            "token_count": self.token_count,
            "warnings": list(self.warnings),
        }


class SentimentPredictor:
    """Predictor cho mô hình BiLSTM nạp từ checkpoint PyTorch."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint tại: {self.checkpoint_path}. "
                "Vui lòng huấn luyện mô hình trước bằng lệnh `python train.py`."
            )

        self.device = torch.device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        config_dict = checkpoint.get("config", {})
        self.config = ExperimentConfig.from_dict(config_dict)
        self.vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.decision_threshold = float(self.config.decision_threshold)

        self.model = BiLSTMSentimentClassifier(
            vocabulary_size=len(self.vocabulary),
            padding_index=self.vocabulary.pad_index,
            config=self.config,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, text: str) -> PredictionResult:
        """Dự đoán cảm xúc cho một câu đánh giá đơn lẻ."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Dự đoán danh sách nhiều câu đánh giá theo lô."""
        if not texts:
            raise ValueError("Danh sách câu đánh giá đầu vào không được để trống.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Mỗi mẫu đánh giá phải là một chuỗi văn bản không rỗng.")

        truncation_strategy = getattr(self.config, "truncation_strategy", "head_tail")
        max_length = getattr(self.config, "max_length", 256)

        encoded_batch = []
        length_batch = []
        audits = []

        for text in texts:
            enc, length, audit = encode_with_audit(
                text,
                self.vocabulary,
                max_length,
                strategy=truncation_strategy,
            )
            encoded_batch.append(enc)
            length_batch.append(length)
            audits.append(audit)

        tokens = torch.tensor(encoded_batch, dtype=torch.long, device=self.device)
        lengths = torch.tensor(length_batch, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            logits = self.model(tokens, lengths)
            scaled_logits = logits / max(1e-4, self.temperature)
            calibrated_probs = torch.sigmoid(scaled_logits).cpu().tolist()

        # Đảm bảo calibrated_probs luôn là list float
        if isinstance(calibrated_probs, float):
            calibrated_probs = [calibrated_probs]

        results: list[PredictionResult] = []
        for i, text in enumerate(texts):
            prob = float(calibrated_probs[i])
            audit = audits[i]

            label = "Positive" if prob >= self.decision_threshold else "Negative"

            warnings: list[str] = []
            if audit["is_truncated"]:
                warnings.append("TRUNCATED_INPUT")
            if audit["used_oov_rate"] > 0.20:
                warnings.append("HIGH_OOV_WARNING")

            lang_warnings = detect_language_warning(text)
            warnings.extend(lang_warnings)

            results.append(
                PredictionResult(
                    label=label,
                    probability=prob,
                    truncated=audit["is_truncated"],
                    oov_rate=audit["used_oov_rate"],
                    token_count=audit["input_tokens"],
                    warnings=warnings,
                )
            )

        return results


class BaselinePredictor:
    """Predictor cho baseline TF-IDF + Logistic Regression."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy baseline model tại: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        self.config = ExperimentConfig(model_type="tfidf_logistic_regression")

    def predict(self, text: str) -> PredictionResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        if not texts:
            raise ValueError("Danh sách văn bản không được để trống.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Văn bản đầu vào không được rỗng.")

        probs = self.pipeline.predict_proba(texts)[:, 1]
        results = []
        for text, prob in zip(texts, probs, strict=False):
            tokens = tokenize(text)
            warnings = detect_language_warning(text)
            label = "Positive" if prob >= 0.5 else "Negative"
            results.append(
                PredictionResult(
                    label=label,
                    probability=float(prob),
                    truncated=False,
                    oov_rate=0.0,
                    token_count=len(tokens),
                    warnings=warnings,
                )
            )
        return results


Predictor = SentimentPredictor | BaselinePredictor


def load_predictor(path: str | Path, device: str = "cpu") -> Predictor:
    """Tự động nạp predictor tương ứng theo đuôi tệp .pt hoặc .joblib."""
    path = Path(path)
    if path.suffix == ".joblib":
        return BaselinePredictor(path)
    return SentimentPredictor(path, device=device)
