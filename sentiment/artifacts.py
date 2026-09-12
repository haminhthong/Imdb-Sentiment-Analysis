"""Quản lý lưu trữ checkpoint mô hình, tệp JSON và biểu đồ trực quan."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn

from .config import ExperimentConfig
from .text import Vocabulary


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    vocabulary: Vocabulary,
    config: ExperimentConfig,
    temperature: float | None = None,
) -> None:
    """Lưu checkpoint mô hình tinh gọn gồm trọng số, từ điển, cấu hình và temperature."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_val = temperature if temperature is not None else (config.temperature or 1.0)

    checkpoint_payload = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "vocabulary": vocabulary.to_dict(),
        "config": config.to_dict(),
        "temperature": temp_val,
    }
    torch.save(checkpoint_payload, path)


def save_json(path: str | Path, data: Any) -> None:
    """Ghi dữ liệu ra tệp JSON định dạng UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_plots(
    output_dir: str | Path,
    history: dict[str, list[float]],
    confusion: list[list[int]],
) -> None:
    """Lưu 2 đồ thị: Lịch sử huấn luyện (Loss & Accuracy) và Ma trận nhầm lẫn (Confusion Matrix)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đồ thị lịch sử huấn luyện
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(history["validation_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Quá Trình Mất Mát (Loss)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history["train_accuracy"], label="Train Accuracy", color="#2ca02c", linewidth=2)
    axes[1].plot(history["validation_accuracy"], label="Val Accuracy", color="#d62728", linewidth=2)
    axes[1].set_title("Quá Trình Độ Chính Xác (Accuracy)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=300)
    plt.close(figure)

    # 2. Confusion Matrix
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=axis,
        cbar=True,
    )
    axis.set_title("Ma Trận Nhầm Lẫn (Confusion Matrix)", fontsize=12, fontweight="bold")
    axis.set_xlabel("Dự Đoán (Predicted)", fontsize=10)
    axis.set_ylabel("Thực Tế (Actual)", fontsize=10)

    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close(figure)


def save_reliability_diagram(
    output_dir: str | Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    filename: str = "reliability_diagram.png",
) -> None:
    """Vẽ và lưu Reliability Diagram (Calibration Curve) cho đánh giá xác suất."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = []
    bin_accs = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (
            (y_prob >= lower) & (y_prob <= upper)
            if i == n_bins - 1
            else (y_prob >= lower) & (y_prob < upper)
        )
        if np.sum(mask) > 0:
            bin_confs.append(float(np.mean(y_prob[mask])))
            bin_accs.append(float(np.mean(y_true[mask])))

    figure, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    if bin_confs:
        ax.plot(bin_confs, bin_accs, "s-", color="#1f77b4", label="Model Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.set_title("Reliability Diagram (Calibration Curve)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=300)
    plt.close(figure)
