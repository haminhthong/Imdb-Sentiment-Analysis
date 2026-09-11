"""Vòng lặp huấn luyện (Training Loop), đánh giá (Evaluation) và Dừng sớm (Early Stopping).

Module này chứa các hàm thực thi cốt lõi cho quá trình học của mô hình:
1. `run_epoch`: Chạy một epoch ở chế độ huấn luyện hoặc đánh giá.
2. `train_model`: Huấn luyện theo dõi Validation Loss, áp dụng Early Stopping.
3. `evaluate_model`: Đo lường toàn diện các chỉ số: Loss, Accuracy, Macro-F1,
   ROC-AUC, PR-AUC, Brier Score, ECE.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .calibration import compute_brier_score, compute_ece, compute_log_loss_score


@dataclass
class EpochMetrics:
    """Chỉ số đánh giá của một epoch."""

    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """Chạy một epoch; nếu truyền `optimizer` sẽ huấn luyện, ngược lại sẽ đánh giá."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for tokens, lengths, labels in tqdm(loader, leave=False):
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            if is_training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(tokens, lengths)
            loss = loss_function(logits, labels)

            if is_training and optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += ((logits >= 0) == labels.bool()).sum().item()
            total_samples += batch_size

    return EpochMetrics(
        loss=total_loss / total_samples if total_samples > 0 else 0.0,
        accuracy=total_correct / total_samples if total_samples > 0 else 0.0,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    on_epoch_end: Callable[[int, EpochMetrics, EpochMetrics], None] | None = None,
) -> dict[str, Any]:
    """Huấn luyện và early-stop chỉ dựa trên Validation Loss.

    ``best_epoch`` được trả ra để final-fit có thể train cố định trên Train + Val
    sau khi kiến trúc và số epoch đã được đóng băng.
    """
    history: dict[str, Any] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_function, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, loss_function, device)

        history["train_loss"].append(train_metrics.loss)
        history["train_accuracy"].append(train_metrics.accuracy)
        history["validation_loss"].append(validation_metrics.loss)
        history["validation_accuracy"].append(validation_metrics.accuracy)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss={train_metrics.loss:.4f}, Acc={train_metrics.accuracy:.2%} | "
            f"Val Loss={validation_metrics.loss:.4f}, Acc={validation_metrics.accuracy:.2%}"
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_metrics, validation_metrics)

        # Kiểm tra điều kiện lưu mô hình tốt nhất
        if validation_metrics.loss < best_loss:
            best_loss = validation_metrics.loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"Dừng sớm (Early Stopping) tại epoch {epoch} "
                    f"vì Validation Loss không giảm trong {patience} epoch liên tiếp."
                )
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = best_loss
    return history


def train_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    epochs: int,
) -> dict[str, list[float]]:
    """Final fit không early stopping, dùng epoch đã chọn từ development."""
    if epochs <= 0:
        raise ValueError("epochs phải lớn hơn 0.")
    history: dict[str, list[float]] = {"train_loss": [], "train_accuracy": []}
    for epoch in range(1, epochs + 1):
        metrics = run_epoch(model, train_loader, loss_function, device, optimizer)
        history["train_loss"].append(metrics.loss)
        history["train_accuracy"].append(metrics.accuracy)
        print(
            f"Final fit epoch {epoch:02d}/{epochs:02d} | "
            f"Loss={metrics.loss:.4f}, Acc={metrics.accuracy:.2%}"
        )
    return history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    temperature: float = 1.0,
    decision_threshold: float = 0.5,
    return_raw: bool = False,
) -> dict[str, Any]:
    """Đánh giá toàn diện các chỉ số phân loại và chỉ số hiệu chuẩn (Calibration).

    Chỉ số tính toán bao gồm:
    - Loss & Log-Loss
    - Accuracy & Macro-F1
    - Precision, Recall
    - ROC-AUC & PR-AUC
    - Brier Score & Expected Calibration Error (ECE)
    - Confusion Matrix
    """
    model.eval()
    labels_all: list[int] = []
    logits_all: list[float] = []
    losses: list[float] = []

    with torch.inference_mode():
        for tokens, lengths, labels in tqdm(loader, leave=False):
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(tokens, lengths)
            loss_val = loss_function(logits, labels)

            losses.append(loss_val.item() * labels.size(0))
            labels_all.extend(labels.int().cpu().tolist())
            logits_all.extend(logits.cpu().tolist())

    total_samples = len(labels_all)
    if total_samples == 0:
        return {}

    logits_arr = np.array(logits_all, dtype=np.float32)
    labels_arr = np.array(labels_all, dtype=np.int32)

    # Tính toán xác suất (có áp dụng Temperature Scaling nếu T != 1.0)
    scaled_logits = logits_arr / max(1e-4, temperature)
    # Tránh overflow khi report trên checkpoint có logit cực lớn.
    scaled_logits = np.clip(scaled_logits, -60.0, 60.0)
    probabilities = 1.0 / (1.0 + np.exp(-scaled_logits))
    predictions = (probabilities >= decision_threshold).astype(int)

    # ROC-AUC và PR-AUC (bảo vệ trường hợp chỉ có 1 class trong mẫu nhỏ)
    unique_labels = set(labels_arr)
    if len(unique_labels) > 1:
        roc_auc = float(roc_auc_score(labels_arr, probabilities))
        pr_auc = float(average_precision_score(labels_arr, probabilities))
    else:
        roc_auc = 0.5
        pr_auc = 0.5

    report = classification_report(
        labels_arr,
        predictions,
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "loss": float(np.sum(losses) / total_samples),
        "accuracy": float(accuracy_score(labels_arr, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": compute_brier_score(labels_arr, probabilities),
        "ece": compute_ece(labels_arr, probabilities, n_bins=10),
        "log_loss": compute_log_loss_score(labels_arr, probabilities),
        "temperature": float(temperature),
        "decision_threshold": float(decision_threshold),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels_arr, predictions).tolist(),
    }

    if return_raw:
        metrics["raw_logits"] = logits_arr
        metrics["raw_labels"] = labels_arr

    return metrics
