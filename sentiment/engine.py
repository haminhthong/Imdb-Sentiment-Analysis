"""Vòng lặp huấn luyện (Training Loop) và đánh giá (Evaluation) cho mô hình PyTorch."""

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

from .calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_ece,
    compute_log_loss_score,
)


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
    """Chạy một epoch huấn luyện hoặc đánh giá."""
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
) -> tuple[dict[str, Any], int]:
    """Huấn luyện mô hình và Early Stopping dựa trên Validation Loss."""
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
    return history, best_epoch


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    temperature: float = 1.0,
    decision_threshold: float = 0.5,
    return_raw: bool = False,
) -> dict[str, Any]:
    """Đánh giá toàn diện mô hình: Loss, Accuracy, Macro-F1, ROC-AUC, PR-AUC, Brier, ECE."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[float] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for tokens, lengths, labels in loader:
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(tokens, lengths)
            loss = loss_function(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.cpu().int().tolist())

    y_true = np.array(all_labels, dtype=int)
    logits_arr = np.array(all_logits, dtype=np.float32)

    scaler = TemperatureScaler(temperature)
    probabilities = scaler.calibrate(logits_arr)
    predictions = (probabilities >= decision_threshold).astype(int)

    unique_labels = set(y_true)
    if len(unique_labels) > 1:
        roc_auc = float(roc_auc_score(y_true, probabilities))
        pr_auc = float(average_precision_score(y_true, probabilities))
    else:
        roc_auc = 0.5
        pr_auc = 0.5

    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": compute_brier_score(y_true, probabilities),
        "ece": compute_ece(y_true, probabilities, n_bins=10),
        "log_loss": compute_log_loss_score(y_true, probabilities),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }

    if return_raw:
        metrics["raw_logits"] = logits_arr
        metrics["raw_labels"] = y_true
        metrics["probabilities"] = probabilities

    return metrics
