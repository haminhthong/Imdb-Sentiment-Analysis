"""Vòng lặp huấn luyện (Training Loop), đánh giá (Evaluation) và Dừng sớm (Early Stopping).

Module này chứa các hàm thực thi cốt lõi cho quá trình học của mô hình:
1. `run_epoch`: Chạy một epoch ở chế độ huấn luyện hoặc đánh giá.
2. `train_model`: Thực hiện chuỗi các epochs huấn luyện, theo dõi loss validation,
   khôi phục trọng số mô hình tốt nhất và áp dụng cơ chế Early Stopping.
3. `evaluate_model`: Đo lường chi tiết kết quả mô hình trên tập test, tính loss,
   accuracy, classification report và confusion matrix.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EpochMetrics:
    """Chỉ số đánh giá của một epoch.

    Attributes:
        loss (float): Giá trị hàm mất mát trung bình trên toàn bộ mẫu.
        accuracy (float): Độ chính xác phân loại trung bình.
    """

    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """Chạy một epoch; nếu truyền `optimizer` sẽ huấn luyện, ngược lại sẽ đánh giá.

    Args:
        model (nn.Module): Mô hình SentimentRNN.
        loader (DataLoader): DataLoader chứa tập dữ liệu.
        loss_function (nn.Module): Hàm mất mát (BCEWithLogitsLoss).
        device (torch.device): Thiết bị tính toán (CPU hoặc CUDA GPU).
        optimizer (torch.optim.Optimizer | None): Optimizer (AdamW). Nếu None, chạy eval mode.

    Returns:
        EpochMetrics: Đối tượng chứa loss và accuracy trung bình của epoch.
    """
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
                # Cắt bớt gradient norm để chống bùng nổ gradient (Gradient Exploding)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            # Dự đoán Positive nếu logit >= 0 (xác suất Sigmoid >= 0.5)
            total_correct += ((logits >= 0) == labels.bool()).sum().item()
            total_samples += batch_size

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
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
) -> dict[str, list[float]]:
    """Huấn luyện mô hình với cơ chế Early Stopping và lưu lại trọng số tốt nhất.

    Args:
        model (nn.Module): Mô hình SentimentRNN cần huấn luyện.
        train_loader (DataLoader): DataLoader tập train.
        validation_loader (DataLoader): DataLoader tập validation.
        optimizer (torch.optim.Optimizer): Thuật toán tối ưu hóa (ví dụ: AdamW).
        loss_function (nn.Module): Hàm mất mát binary classification.
        device (torch.device): Thiết bị tính toán.
        epochs (int): Số lượng epoch huấn luyện tối đa.
        patience (int): Số epoch kiên nhẫn khi validation loss không giảm.

    Returns:
        dict[str, list[float]]: Lịch sử chỉ số ('train_loss', 'train_accuracy',
            'validation_loss', 'validation_accuracy') qua từng epoch.
    """
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, loss_function, device
        )

        history["train_loss"].append(train_metrics.loss)
        history["train_accuracy"].append(train_metrics.accuracy)
        history["validation_loss"].append(validation_metrics.loss)
        history["validation_accuracy"].append(validation_metrics.accuracy)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss={train_metrics.loss:.4f}, Acc={train_metrics.accuracy:.2%} | "
            f"Val Loss={validation_metrics.loss:.4f}, Acc={validation_metrics.accuracy:.2%}"
        )

        # Kiểm tra điều kiện lưu mô hình tốt nhất
        if validation_metrics.loss < best_loss:
            best_loss = validation_metrics.loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"Dừng sớm (Early Stopping) tại epoch {epoch} "
                    f"vì Validation Loss không giảm trong {patience} epoch liên tiếp."
                )
                break

    # Khôi phục mô hình về trạng thái có Validation Loss tốt nhất
    model.load_state_dict(best_state)
    return history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Đánh giá chi tiết mô hình trên tập dữ liệu kiểm thử (Test Set).

    Args:
        model (nn.Module): Mô hình đã huấn luyện.
        loader (DataLoader): DataLoader tập test.
        loss_function (nn.Module): Hàm mất mát.
        device (torch.device): Thiết bị tính toán.

    Returns:
        dict[str, Any]: Kết quả chi tiết bao gồm:
            - 'loss': Loss trung bình.
            - 'accuracy': Độ chính xác tổng thể.
            - 'classification_report': Precision, Recall, F1-score từng lớp.
            - 'confusion_matrix': Ma trận nhầm lẫn dạng 2D list.
    """
    model.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
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
            predictions_all.extend((logits >= 0).int().cpu().tolist())

    total_samples = len(labels_all)
    return {
        "loss": float(np.sum(losses) / total_samples),
        "accuracy": float(accuracy_score(labels_all, predictions_all)),
        "classification_report": classification_report(
            labels_all,
            predictions_all,
            target_names=["Negative", "Positive"],
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(labels_all, predictions_all).tolist(),
    }


