"""Kỹ thuật hiệu chuẩn xác suất (Probability Calibration) và chính sách quyết định (Decision Policy).

Module này cung cấp:
1. TemperatureScaler: Tối ưu hóa tham số nhiệt độ T > 0 trên Validation Logits để
   chuẩn hóa phân phối xác suất đầu ra (chống Overconfidence).
2. Các chỉ số đo lường hiệu chuẩn:
   - Expected Calibration Error (ECE)
   - Brier Score
   - Negative Log-Likelihood (Log Loss)
3. DecisionPolicy: Xác định nhãn, độ tin cậy và vùng bất định (Uncertainty Band)
   cho phép gắn cờ các trường hợp cần con người rà soát (Human-in-the-loop review).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Tính Brier Score: Sai số toàn phương trung bình giữa xác suất dự đoán và nhãn thực tế.

    Brier score càng thấp (gần 0) thể hiện mô hình dự đoán càng chính xác và tin cậy.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def compute_log_loss_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Tính Binary Cross-Entropy / Log Loss trên xác suất đã kẹp giá trị (clipped)."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    loss = -(y_true_arr * np.log(y_prob_arr) + (1.0 - y_true_arr) * np.log(1.0 - y_prob_arr))
    return float(np.mean(loss))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Tính Expected Calibration Error (ECE) với phân chia đều các khoảng xác suất (Equal-width binning).

    ECE đo lường khoảng cách trung bình có trọng số giữa độ tin cậy và độ chính xác thực tế:
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true (np.ndarray): Nhãn nhị phân thực tế {0, 1}.
        y_prob (np.ndarray): Xác suất dự đoán thuộc lớp Positive [0, 1].
        n_bins (int): Số lượng khoảng chia (mặc định 10).

    Returns:
        float: Giá trị ECE nằm trong khoảng [0, 1].
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    total_samples = len(y_true_arr)
    if total_samples == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Lấy các mẫu rơi vào bin [bin_lower, bin_upper]
        if i == n_bins - 1:
            in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr <= bin_upper)
        else:
            in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true_arr[in_bin])
            bin_conf = np.mean(y_prob_arr[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


class TemperatureScaler:
    """Hiệu chuẩn logits bằng một tham số vô hướng nhiệt độ T > 0 (Temperature Scaling).

    Hàm tối ưu hóa: tìm T > 0 sao cho NLL (Negative Log-Likelihood) trên tập Validation nhỏ nhất:
        min_T BCEWithLogitsLoss(logits / T, labels)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(1e-4, float(temperature))

    def fit(self, logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> float:
        """Học tham số nhiệt độ T CHỈ từ logits và labels của tập Validation.

        Args:
            logits: Logits thô của mô hình trên tập validation.
            labels: Nhãn thực tế {0, 1} của tập validation.

        Returns:
            float: Tham số nhiệt độ T tối ưu sau khi fit.
        """
        if isinstance(logits, np.ndarray):
            logits_tensor = torch.from_numpy(logits).float()
        else:
            logits_tensor = logits.detach().float()

        if isinstance(labels, np.ndarray):
            labels_tensor = torch.from_numpy(labels).float()
        else:
            labels_tensor = labels.detach().float()

        # Khởi tạo log_temperature để đảm bảo T = exp(log_T) luôn dương
        log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=100)

        def closure():
            optimizer.zero_grad()
            t = torch.exp(log_temp)
            loss = criterion(logits_tensor / t, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temp).item())
        return self.temperature

    def calibrate(self, logits: torch.Tensor | np.ndarray) -> np.ndarray:
        """Áp dụng nhiệt độ T và chuyển đổi sang calibrated probabilities qua hàm sigmoid."""
        if isinstance(logits, np.ndarray):
            scaled = logits / self.temperature
            return 1.0 / (1.0 + np.exp(-scaled))

        with torch.inference_mode():
            scaled = logits / self.temperature
            return torch.sigmoid(scaled).cpu().numpy()


@dataclass(frozen=True)
class DecisionResult:
    """Kết quả ra quyết định từ xác suất đã hiệu chuẩn."""

    label: str
    decision: str
    uncertain: bool


class DecisionPolicy:
    """Chính sách quyết định dựa trên ngưỡng và vùng bất định."""

    def __init__(
        self,
        threshold: float = 0.5,
        uncertain_band: tuple[float, float] = (0.40, 0.60),
    ) -> None:
        self.threshold = float(threshold)
        self.uncertain_lower = float(uncertain_band[0])
        self.uncertain_upper = float(uncertain_band[1])

    def decide(self, probability: float) -> DecisionResult:
        """Đưa ra quyết định cho một xác suất đơn lẻ."""
        label = "Positive" if probability >= self.threshold else "Negative"
        uncertain = bool(self.uncertain_lower <= probability <= self.uncertain_upper)
        decision = "review_required" if uncertain else "accepted"
        return DecisionResult(label=label, decision=decision, uncertain=uncertain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_threshold": self.threshold,
            "uncertain_lower": self.uncertain_lower,
            "uncertain_upper": self.uncertain_upper,
        }
