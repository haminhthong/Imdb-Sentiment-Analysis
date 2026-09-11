"""Quản lý checkpoint training/release, tệp JSON và biểu đồ trực quan.

Module này cung cấp các chức năng:
1. Ghi PyTorch Checkpoint (`model.pt`) theo Artifact Schema Version 3 (metadata phong phú).
2. Ghi tệp JSON báo cáo (`validation_metrics.json`, `test_metrics.json`, `history.json`).
3. Vẽ và lưu đồ thị huấn luyện (Loss/Accuracy), Ma trận nhầm lẫn và Biểu đồ
   độ tin cậy (Reliability Diagram).
"""

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn

from .config import ExperimentConfig
from .text import TOKENIZER_VERSION, Vocabulary


def resolve_git_commit() -> str:
    """Lấy Git SHA hiện tại để artifact có thể truy nguyên về source code."""
    try:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unspecified"
    except (OSError, subprocess.CalledProcessError):
        return "unspecified"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    vocabulary: Vocabulary,
    config: ExperimentConfig,
    training_data_hash: str | None = None,
    *,
    model_version: str = "1.0.0",
    checkpoint_kind: str = "training",
    source_dataset_hash: str | None = None,
    train_split_hash: str | None = None,
    validation_split_hash: str | None = None,
    calibration_split_hash: str | None = None,
    official_test_hash: str | None = None,
    git_commit: str | None = None,
    best_dev_epoch: int | None = None,
    training_epoch: int | None = None,
    final_fit_epoch: int | None = None,
    final_metrics: dict[str, Any] | None = None,
) -> None:
    """Lưu checkpoint PyTorch theo chuẩn Artifact Schema v3.

    Bao gồm đầy đủ trọng số mô hình, từ điển, cấu hình siêu tham số,
    phiên bản tokenizer, thông số calibration và fingerprint của từng split.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = {
        "artifact_schema_version": 3,
        "schema_version": 3,
        "model_version": model_version,
        "checkpoint_kind": checkpoint_kind,
        "architecture": {"type": config.model_type},
        "model_type": config.model_type,  # Alias để đọc artifact v2.
        "tokenizer_version": TOKENIZER_VERSION,
        "vocabulary_hash": vocabulary.compute_hash(),
        "training_data_hash": training_data_hash or "unspecified",
        "source_dataset_hash": source_dataset_hash or "unspecified",
        "train_split_hash": train_split_hash or training_data_hash or "unspecified",
        "validation_split_hash": validation_split_hash or "unspecified",
        "calibration_split_hash": calibration_split_hash or "unspecified",
        "official_test_hash": official_test_hash or "unspecified",
        "decision_threshold": config.decision_threshold,
        "confidence_threshold": config.confidence_threshold,
        "temperature": config.temperature if config.temperature is not None else 1.0,
        "max_length": config.max_length,
        "truncation_strategy": config.truncation_strategy,
        "preprocessing_contract": {
            "tokenizer_version": TOKENIZER_VERSION,
            "max_length": config.max_length,
            "truncation_strategy": config.truncation_strategy,
        },
        "git_commit": git_commit or resolve_git_commit(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "test_protocol": "imdb-official-v1",
        "final_fit_epoch": final_fit_epoch,
        "best_dev_epoch": best_dev_epoch,
        "training_epoch": training_epoch,
        "final_metrics": final_metrics or {},
        # Lưu state CPU để artifact có thể chuyển máy/GPU an toàn.
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "vocabulary": vocabulary.to_dict(),
        "config": config.to_dict(),
    }
    torch.save(checkpoint_payload, path)


def save_json(path: str | Path, data: Any) -> None:
    """Ghi dữ liệu dưới dạng tệp JSON định dạng UTF-8 đẹp mắt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_plots(
    output_dir: str | Path,
    history: dict[str, list[float]],
    confusion: list[list[int]],
) -> None:
    """Vẽ và lưu hai đồ thị: Lịch sử huấn luyện (Loss & Acc) và Confusion Matrix."""
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


def save_reliability_diagram(
    output_dir: str | Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    filename: str = "reliability_diagram.png",
) -> None:
    """Vẽ và lưu biểu đồ Reliability Diagram (Calibration Curve) cho đánh giá xác suất."""
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
