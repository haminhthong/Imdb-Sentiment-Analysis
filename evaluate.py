"""Đánh giá mô hình (BiLSTM hoặc Baseline) trên tập Test chính thức.

Nguyên tắc:
- Tập Test hoàn toàn độc lập, không dùng để chọn mô hình hay tune threshold.
- Báo cáo đầy đủ: Accuracy, Macro-F1, ROC-AUC, PR-AUC, Brier Score, ECE.
- Xuất biểu đồ Reliability Diagram để kiểm tra độ tin cậy của xác suất.
"""

import argparse
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader

from sentiment.artifacts import save_json, save_reliability_diagram
from sentiment.config import ExperimentConfig
from sentiment.data import IMDBDataset
from sentiment.data_validation import load_dataset
from sentiment.engine import evaluate_model
from sentiment.model import BiLSTMSentimentClassifier
from sentiment.text import Vocabulary
from sentiment.utils import configure_utf8_output, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình CineSentiment trên tập Test")
    parser.add_argument(
        "--test-data",
        default="data/raw/test.csv",
        help="Đường dẫn tập Test (mặc định: data/raw/test.csv hoặc test.csv)",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/model.pt",
        help="Đường dẫn checkpoint BiLSTM (model.pt) hoặc baseline (model.joblib)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Thư mục xuất kết quả đánh giá (mặc định: artifacts)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Thiết bị tính toán cho PyTorch (mặc định: auto)",
    )
    return parser.parse_args()


def evaluate_baseline_test(model_path: Path, test_frame) -> dict:
    """Đánh giá pipeline TF-IDF + Logistic Regression trên tập Test."""
    from baseline import evaluate_baseline

    pipeline = joblib.load(model_path)
    metrics = evaluate_baseline(pipeline, test_frame["text"], test_frame["label"].astype(int))
    probabilities = pipeline.predict_proba(test_frame["text"])[:, 1]
    metrics["probabilities"] = probabilities
    metrics["raw_labels"] = test_frame["label"].to_numpy(dtype=int)
    return metrics


def evaluate_bilstm_test(checkpoint_path: Path, test_frame, device_name: str) -> dict:
    """Đánh giá checkpoint BiLSTM trên tập Test."""
    device = select_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ExperimentConfig.from_dict(checkpoint.get("config", {}))
    vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])
    temperature = float(checkpoint.get("temperature", 1.0))

    model = BiLSTMSentimentClassifier(
        vocabulary_size=len(vocabulary),
        padding_index=vocabulary.pad_index,
        config=config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_dataset = IMDBDataset(
        test_frame,
        vocabulary,
        max_length=config.max_length,
        strategy=config.truncation_strategy,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    metrics = evaluate_model(
        model=model,
        loader=test_loader,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        device=device,
        temperature=temperature,
        return_raw=True,
    )
    return metrics


def main() -> None:
    configure_utf8_output()
    args = parse_args()
    test_path = Path(args.test_data)
    if not test_path.is_file() and Path("test.csv").is_file():
        test_path = Path("test.csv")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_frame = load_dataset(test_path, deduplicate=False)

    print("=" * 60)
    print("*** ĐÁNH GIÁ TRÊN TẬP TEST CHÍNH THỨC ***")
    print("=" * 60)
    print(f"-> Tập Test: {test_path} ({len(test_frame):,} mẫu)")
    print(f"-> Model: {checkpoint_path}")
    print("-" * 60)

    if checkpoint_path.suffix == ".joblib":
        metrics = evaluate_baseline_test(checkpoint_path, test_frame)
    else:
        metrics = evaluate_bilstm_test(checkpoint_path, test_frame, args.device)

    probs = metrics.pop("probabilities")
    labels = metrics.pop("raw_labels")
    metrics.pop("raw_logits", None)

    print(f"-> Test Accuracy   : {metrics['accuracy']:.2%}")
    print(f"-> Test Macro-F1   : {metrics['macro_f1']:.2%}")
    print(f"-> Test ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"-> Test Brier Score: {metrics['brier_score']:.4f}")
    print(f"-> Test ECE        : {metrics['ece']:.4f}")
    print("-" * 60)

    save_json(output_dir / "test_metrics.json", metrics)
    save_reliability_diagram(
        output_dir,
        labels,
        probs,
        filename="test_reliability_diagram.png",
    )
    print(f"✅ Đã lưu test metrics tại: {output_dir / 'test_metrics.json'}")
    print(f"✅ Đã lưu reliability diagram tại: {output_dir / 'test_reliability_diagram.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
