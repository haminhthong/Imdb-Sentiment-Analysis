"""Quản lý lưu trữ Checkpoint, tệp JSON kết quả và biểu đồ biểu diễn trực quan.

Module này cung cấp các chức năng ghi xuất kết quả thí nghiệm bao gồm:
1. Ghi tệp PyTorch Checkpoint (`model.pt`) gồm trọng số, từ điển và siêu tham số.
2. Ghi tệp JSON báo cáo chỉ số (`metrics.json`, `history.json`).
3. Vẽ và lưu đồ thị tiến trình huấn luyện Loss/Accuracy và Ma trận nhầm lẫn (Confusion Matrix).
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
) -> None:
    """Lưu mô hình PyTorch, bộ từ vựng và cấu hình siêu tham số vào tệp `.pt`.

    Args:
        path (str | Path): Đường dẫn đến tệp lưu checkpoint.
        model (nn.Module): Mô hình SentimentRNN.
        vocabulary (Vocabulary): Bộ từ vựng huấn luyện.
        config (ExperimentConfig): Siêu tham số mô hình.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabulary": vocabulary.to_dict(),
            "config": config.to_dict(),
        },
        path,
    )


def save_json(path: str | Path, data: Any) -> None:
    """Ghi dữ liệu dưới dạng tệp JSON định dạng UTF-8 đẹp mắt.

    Args:
        path (str | Path): Đường dẫn tệp JSON đầu ra.
        data (Any): Dữ liệu cần ghi (dict, list, primitive values).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_plots(
    output_dir: str | Path,
    history: dict[str, list[float]],
    confusion: list[list[int]],
) -> None:
    """Vẽ và lưu hai đồ thị: Lịch sử huấn luyện (Loss & Acc) và Confusion Matrix.

    Args:
        output_dir (str | Path): Thư mục lưu xuất đồ thị `.png`.
        history (dict[str, list[float]]): Lịch sử train/val loss và accuracy.
        confusion (list[list[int]]): Ma trận nhầm lẫn 2D list.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đồ thị 1: Lịch sử Loss và Accuracy theo Epoch
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

    # Đồ thị 2: Seaborn Heatmap Ma Trận Nhầm Lẫn (Confusion Matrix)
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


