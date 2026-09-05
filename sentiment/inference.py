"""Nạp checkpoint mô hình và phục vụ suy luận (Inference).

Module này hỗ trợ suy luận dự đoán câu đơn lẻ hoặc theo lô (batch),
chia sẻ dùng chung cho CLI (`predict.py`), REST API (`api.py`), và Streamlit (`app.py`).

Các tính năng nổi bật:
1. Phục hồi Artifact Schema v2 (trọng số, vocabulary, temperature scaling, decision threshold).
2. Tách bạch rõ giữa raw sigmoid score và calibrated probability.
3. Chính sách quyết định với vùng bất định (decision: 'accepted' hoặc 'review_required').
4. Kiểm toán chuỗi đầu vào: phát hiện cắt chuỗi (truncation), tỷ lệ OOV cao và cảnh báo ngôn ngữ.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration import DecisionPolicy, DecisionResult
from .config import ExperimentConfig
from .model import SentimentRNN
from .text import Vocabulary, detect_language_warning, encode_with_audit


@dataclass(frozen=True)
class PredictionResult:
    """Kết quả dự đoán cảm xúc hoàn chỉnh của một câu đánh giá.

    Attributes:
        label (str): Nhãn cảm xúc dự đoán ('Positive' hoặc 'Negative').
        positive_probability (float): Xác suất thuộc nhãn Positive đã qua calibration.
        confidence (float): Độ tin cậy của dự đoán (max(p, 1-p)).
        positive_score (float): Điểm sigmoid thô chưa qua hiệu chuẩn.
        decision (str): Quyết định hệ thống ('accepted' hoặc 'review_required').
        uncertain (bool): Đánh dấu dự đoán rơi vào vùng bất định ranh giới.
        input_tokens (int): Tổng số token gốc của câu.
        used_tokens (int): Số token thực tế được đưa vào mô hình.
        truncated (bool): Cờ đánh dấu câu bị cắt bớt do vượt max_length.
        oov_rate (float): Tỷ lệ token không nằm trong từ điển (OOV).
        warnings (list[str]): Danh sách các cảnh báo độ tin cậy.
    """

    label: str
    positive_probability: float
    confidence: float
    positive_score: float = 0.0
    decision: str = "accepted"
    uncertain: bool = False
    input_tokens: int = 0
    used_tokens: int = 0
    truncated: bool = False
    oov_rate: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi đối tượng kết quả thành Dictionary dạng JSON."""
        return {
            "label": self.label,
            "positive_probability": round(self.positive_probability, 4),
            "confidence": round(self.confidence, 4),
            "positive_score": round(self.positive_score, 4),
            "decision": self.decision,
            "uncertain": self.uncertain,
            "input_tokens": self.input_tokens,
            "used_tokens": self.used_tokens,
            "truncated": self.truncated,
            "oov_rate": round(self.oov_rate, 4),
            "warnings": list(self.warnings),
        }


class SentimentPredictor:
    """Lớp đóng gói mô hình RNN và bộ từ vựng để dự đoán tin cậy và an toàn."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
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
            weights_only=True,
        )

        config_dict = checkpoint.get("config", {})
        self.config = ExperimentConfig(**config_dict)
        self.vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])

        # Nạp các tham số hiệu chuẩn từ checkpoint
        self.temperature = float(checkpoint.get("temperature") or self.config.temperature or 1.0)
        self.decision_threshold = float(
            checkpoint.get("decision_threshold") or self.config.decision_threshold or 0.5
        )
        self.decision_policy = DecisionPolicy(
            threshold=self.decision_threshold,
            uncertain_band=(self.config.uncertain_lower, self.config.uncertain_upper),
        )

        self.model = SentimentRNN(
            len(self.vocabulary), self.vocabulary.pad_index, self.config
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, text: str) -> PredictionResult:
        """Dự đoán cảm xúc của một câu đánh giá đơn lẻ."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Dự đoán danh sách nhiều câu đánh giá theo lô (Batch Processing)."""
        if not texts:
            raise ValueError("Danh sách câu đánh giá đầu vào không được để trống.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Mỗi mẫu đánh giá phải là một chuỗi văn bản không rỗng.")

        truncation_strategy = getattr(self.config, "truncation_strategy", "first")

        encoded_batch = []
        length_batch = []
        audits = []

        for text in texts:
            enc, length, audit = encode_with_audit(
                text,
                self.vocabulary,
                self.config.max_length,
                strategy=truncation_strategy,
            )
            encoded_batch.append(enc)
            length_batch.append(length)
            audits.append(audit)

        tokens = torch.tensor(encoded_batch, dtype=torch.long, device=self.device)
        lengths = torch.tensor(length_batch, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            logits = self.model(tokens, lengths)
            raw_scores = torch.sigmoid(logits).cpu().tolist()

            # Áp dụng Temperature Scaling cho calibrated probabilities
            scaled_logits = logits / max(1e-4, self.temperature)
            calibrated_probs = torch.sigmoid(scaled_logits).cpu().tolist()

        results: list[PredictionResult] = []
        for i, text in enumerate(texts):
            prob = float(calibrated_probs[i])
            score = float(raw_scores[i])
            audit = audits[i]

            decision_res: DecisionResult = self.decision_policy.decide(prob)
            confidence = prob if decision_res.label == "Positive" else 1.0 - prob

            warnings: list[str] = []
            if audit["is_truncated"]:
                warnings.append("TRUNCATED_INPUT")
            if audit["oov_rate"] > 0.20:
                warnings.append("HIGH_OOV_WARNING")

            lang_warnings = detect_language_warning(text)
            warnings.extend(lang_warnings)

            results.append(
                PredictionResult(
                    label=decision_res.label,
                    positive_probability=prob,
                    confidence=confidence,
                    positive_score=score,
                    decision=decision_res.decision,
                    uncertain=decision_res.uncertain,
                    input_tokens=audit["input_tokens"],
                    used_tokens=audit["used_tokens"],
                    truncated=audit["is_truncated"],
                    oov_rate=audit["oov_rate"],
                    warnings=warnings,
                )
            )

        return results
