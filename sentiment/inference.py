"""Nạp checkpoint mô hình và phục vụ suy luận (Inference).

Module này hỗ trợ suy luận dự đoán câu đơn lẻ hoặc theo lô (batch),
chia sẻ dùng chung cho CLI (`predict.py`), REST API (`api.py`), và Streamlit (`app.py`).

Các tính năng nổi bật:
1. Phục hồi Artifact Schema v3 (trọng số, vocabulary và toàn bộ policy đã khóa).
2. Tách bạch rõ giữa raw sigmoid score và calibrated probability.
3. Chính sách selective classification theo confidence threshold.
4. Kiểm toán chuỗi đầu vào: phát hiện cắt chuỗi (truncation), tỷ lệ OOV cao và cảnh báo ngôn ngữ.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import torch

from .calibration import DecisionPolicy, DecisionResult
from .config import ExperimentConfig
from .model import SentimentRNN
from .text import (
    TOKENIZER_VERSION,
    Vocabulary,
    detect_language_warning,
    encode_with_audit,
    tokenize,
)


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
        input_oov_rate (float): OOV rate trên toàn bộ input.
        used_oov_rate (float): OOV rate trên token thực sự được model dùng.
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
    input_oov_rate: float = 0.0
    used_oov_rate: float = 0.0
    # Alias hiển thị tương thích client cũ; giá trị đại diện cho used_oov_rate.
    oov_rate: float = 0.0
    model_version: str = "unknown"
    tokenizer_version: str = "unknown"
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
            "input_oov_rate": round(self.input_oov_rate, 4),
            "used_oov_rate": round(self.used_oov_rate, 4),
            "oov_rate": round(self.oov_rate, 4),
            "warnings": list(self.warnings),
            "model_version": self.model_version,
            "tokenizer_version": self.tokenizer_version,
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

        self._validate_checkpoint(checkpoint)
        config_dict = checkpoint.get("config", {})
        self.config = ExperimentConfig(**config_dict)
        self.vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])
        self.model_version = str(checkpoint.get("model_version", "legacy"))
        self.tokenizer_version = str(checkpoint.get("tokenizer_version", "unknown"))

        # Nạp các tham số hiệu chuẩn từ checkpoint
        self.temperature = float(checkpoint.get("temperature", self.config.temperature or 1.0))
        self.decision_threshold = float(
            checkpoint.get("decision_threshold", self.config.decision_threshold)
        )
        self.confidence_threshold = float(
            checkpoint.get("confidence_threshold", self.config.confidence_threshold)
        )
        self.decision_policy = DecisionPolicy(
            threshold=self.decision_threshold,
            confidence_threshold=self.confidence_threshold,
            # Artifact v2 không có confidence_threshold; khôi phục band cũ.
            uncertain_band=(self.config.uncertain_lower, self.config.uncertain_upper)
            if "confidence_threshold" not in checkpoint
            and checkpoint.get("artifact_schema_version", 2) < 3
            else None,
        )

        self.model = SentimentRNN(len(self.vocabulary), self.vocabulary.pad_index, self.config).to(
            self.device
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @staticmethod
    def _validate_checkpoint(checkpoint: dict[str, Any]) -> None:
        """Fail-fast khi artifact thiếu trường bắt buộc hoặc sai tokenizer."""
        required = {"model_state", "vocabulary", "config", "tokenizer_version"}
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(f"Artifact không hợp lệ, thiếu trường: {', '.join(missing)}")
        if checkpoint["tokenizer_version"] != TOKENIZER_VERSION:
            raise ValueError(
                "Tokenizer version của artifact không khớp runtime: "
                f"{checkpoint['tokenizer_version']} != {TOKENIZER_VERSION}."
            )
        schema = int(checkpoint.get("schema_version", checkpoint.get("artifact_schema_version", 2)))
        if schema not in {2, 3}:
            raise ValueError(f"Artifact schema v{schema} không được runtime hỗ trợ.")

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
            confidence = max(prob, 1.0 - prob)

            warnings: list[str] = []
            if audit["is_truncated"]:
                warnings.append("TRUNCATED_INPUT")
            if audit["used_oov_rate"] > 0.20:
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
                    input_oov_rate=audit["input_oov_rate"],
                    used_oov_rate=audit["used_oov_rate"],
                    oov_rate=audit["used_oov_rate"],
                    warnings=warnings,
                    model_version=self.model_version,
                    tokenizer_version=self.tokenizer_version,
                )
            )

        return results


@dataclass(frozen=True)
class _BaselineConfig:
    """Metadata tối thiểu để baseline dùng chung contract serving với RNN."""

    model_type: str = "tfidf_logistic_regression"
    max_length: int = 0
    truncation_strategy: str = "none"


class BaselinePredictor:
    """Adapter suy luận cho release TF-IDF + Logistic Regression.

    Baseline không có vocabulary neural hoặc truncation. Adapter vẫn trả về
    cùng ``PredictionResult`` để API, CLI và Streamlit không phải biết chi tiết
    implementation của từng champion.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.checkpoint_path = Path(model_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy baseline artifact: {self.checkpoint_path}")

        self.pipeline = joblib.load(self.checkpoint_path)
        if not hasattr(self.pipeline, "predict_proba"):
            raise ValueError("Baseline artifact không cung cấp predict_proba().")

        self.model = self
        self.config = _BaselineConfig()
        self.temperature = 1.0
        self.decision_threshold = 0.5
        self.confidence_threshold = 0.5
        self.model_version = "1.0.0-baseline"
        self.tokenizer_version = TOKENIZER_VERSION
        self.vocabulary = self._get_vocabulary()
        self._load_metadata()
        self.decision_policy = DecisionPolicy(
            threshold=self.decision_threshold,
            confidence_threshold=self.confidence_threshold,
        )

    def _get_vocabulary(self) -> dict[str, int]:
        """Lấy vocabulary n-gram của TF-IDF để hiển thị thông tin artifact."""
        try:
            return dict(self.pipeline.named_steps["tfidf"].vocabulary_)
        except (AttributeError, KeyError, TypeError):
            return {}

    def _load_metadata(self) -> None:
        """Nạp metadata release nếu package baseline đã tạo tệp metadata."""
        metadata_path = self.checkpoint_path.with_name("baseline_metadata.json")
        if not metadata_path.is_file():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model_version = str(metadata.get("model_version", self.model_version))
        self.decision_threshold = float(metadata.get("decision_threshold", self.decision_threshold))
        self.confidence_threshold = float(
            metadata.get("confidence_threshold", self.confidence_threshold)
        )

    def count_parameters(self) -> int:
        """Trả số hệ số tuyến tính để giữ API ``/info`` đồng nhất."""
        classifier = self.pipeline.named_steps.get("classifier")
        if classifier is None:
            return 0
        return int(classifier.coef_.size + classifier.intercept_.size)

    def predict(self, text: str) -> PredictionResult:
        """Dự đoán một văn bản bằng baseline."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Dự đoán batch và ánh xạ kết quả về response contract chung."""
        if not texts:
            raise ValueError("Danh sách câu đánh giá đầu vào không được để trống.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Mỗi mẫu đánh giá phải là một chuỗi văn bản không rỗng.")

        probabilities = self.pipeline.predict_proba(texts)[:, 1]
        results: list[PredictionResult] = []
        for text, probability in zip(texts, probabilities, strict=True):
            probability = float(probability)
            decision = self.decision_policy.decide(probability)
            warnings = detect_language_warning(text)
            token_count = len(tokenize(text))
            results.append(
                PredictionResult(
                    label=decision.label,
                    positive_probability=probability,
                    confidence=max(probability, 1.0 - probability),
                    positive_score=probability,
                    decision=decision.decision,
                    uncertain=decision.uncertain,
                    input_tokens=token_count,
                    used_tokens=token_count,
                    truncated=False,
                    input_oov_rate=0.0,
                    used_oov_rate=0.0,
                    oov_rate=0.0,
                    model_version=self.model_version,
                    tokenizer_version=self.tokenizer_version,
                    warnings=warnings,
                )
            )
        return results


Predictor = SentimentPredictor | BaselinePredictor


def load_predictor(model_path: str | Path, device: str = "cpu") -> Predictor:
    """Nạp đúng adapter theo loại artifact mà không làm caller biết format file."""
    path = Path(model_path)
    if path.suffix == ".pt":
        return SentimentPredictor(path, device=device)
    if path.suffix == ".joblib":
        return BaselinePredictor(path)
    raise ValueError("Artifact phải có phần mở rộng .pt hoặc .joblib.")
