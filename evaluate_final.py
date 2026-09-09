"""Đánh giá mô hình Champion duy nhất trên tập Official Test bị đóng băng (Locked Final Test).

Đây là script DUY NHẤT trong toàn bộ platform được phép mở tập Official Test:
- Khôi phục nguyên vẹn trọng số, từ điển, cấu hình siêu tham số và hệ số hiệu chuẩn Temperature.
- Đánh giá tập Test đúng 1 lần (Single-pass locked evaluation) để chống rò rỉ và overfit trên test.
- Báo cáo toàn diện: Accuracy, Macro-F1, ROC-AUC, PR-AUC, Brier Score, ECE và các lát cắt lỗi (Error Slices).
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from sentiment.config import ExperimentConfig
from sentiment.data import IMDBDataset
from sentiment.data_validation import (
    compute_dataset_hash,
    load_official_dataset,
    validate_official_test_independence,
)
from sentiment.engine import evaluate_model
from sentiment.model import SentimentRNN
from sentiment.text import Vocabulary, encode_with_audit, tokenize
from sentiment.utils import select_device


LINGUISTIC_PATTERNS = {
    "has_negation": re.compile(
        r"\b(not|never|no|hardly|barely|scarcely|without)\b|n't\b", re.IGNORECASE
    ),
    "has_mixed_sentiment": re.compile(
        r"\b(but|however|although|though|except|yet|despite|in spite of)\b", re.IGNORECASE
    ),
    "has_sentiment_reversal": re.compile(
        r"\b(at first|initially|started (out|well)|turned out|until the ending)\b", re.IGNORECASE
    ),
    "has_intensifier": re.compile(
        r"\b(absolutely|totally|utterly|completely|extremely|truly)\b", re.IGNORECASE
    ),
}


def build_evaluation_records(
    texts: list[str],
    labels: list[int],
    probabilities: np.ndarray,
    threshold: float,
    confidence_threshold: float,
    vocabulary: Vocabulary | None = None,
    max_length: int = 256,
    truncation_strategy: str = "head_tail",
) -> pd.DataFrame:
    """Tạo bảng prediction facts để error analysis không đọc Test lần hai.

    Bảng này không chứa raw review text. Official Test chỉ được đọc trong
    release evaluator; các bước phân tích sau đó chỉ dùng bảng facts này.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    predictions = (probabilities >= threshold).astype(int)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    accepted = confidence >= confidence_threshold
    token_audits = [
        encode_with_audit(text, vocabulary, max_length, strategy=truncation_strategy)[2]
        if vocabulary is not None
        else {
            "input_tokens": len(tokenize(text)),
            "used_tokens": len(tokenize(text)),
            "is_truncated": False,
            "input_oov_rate": 0.0,
            "used_oov_rate": 0.0,
        }
        for text in texts
    ]
    return pd.DataFrame(
        {
            "row_id": np.arange(len(texts), dtype=int),
            "label": labels_array,
            "prediction": predictions,
            "positive_probability": probabilities,
            "confidence": confidence,
            "decision": np.where(accepted, "accepted", "review_required"),
            "is_error": labels_array != predictions,
            "input_tokens": [audit["input_tokens"] for audit in token_audits],
            "used_tokens": [audit["used_tokens"] for audit in token_audits],
            "truncated": [audit["is_truncated"] for audit in token_audits],
            "input_oov_rate": [audit["input_oov_rate"] for audit in token_audits],
            "used_oov_rate": [audit["used_oov_rate"] for audit in token_audits],
            **{
                name: [bool(pattern.search(text)) for text in texts]
                for name, pattern in LINGUISTIC_PATTERNS.items()
            },
        }
    )


def benchmark_inference(model_dir: Path, texts: list[str], runs: int = 20) -> dict[str, float]:
    """Đo latency CPU p50/p95 và throughput trên một batch đại diện."""
    sample = texts[: min(32, len(texts))]
    if not sample:
        return {
            "batch_size": 0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "throughput_samples_per_second": 0.0,
        }

    checkpoint_path = model_dir / "model.pt"
    if checkpoint_path.is_file():
        from sentiment.inference import SentimentPredictor

        predictor = SentimentPredictor(checkpoint_path, device="cpu")

        def predict() -> None:
            predictor.predict_batch(sample)

    else:
        pipeline = joblib.load(model_dir / "model.joblib")

        def predict() -> None:
            pipeline.predict_proba(sample)

    for _ in range(3):
        predict()
    durations = []
    for _ in range(runs):
        started = time.perf_counter()
        predict()
        durations.append((time.perf_counter() - started) * 1_000)
    durations.sort()
    p50 = durations[len(durations) // 2]
    p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))]
    mean_seconds = sum(durations) / len(durations) / 1_000
    return {
        "batch_size": float(len(sample)),
        "latency_p50_ms": round(float(p50), 4),
        "latency_p95_ms": round(float(p95), 4),
        "throughput_samples_per_second": round(float(len(sample) / mean_seconds), 4),
    }


def evaluate_slices(
    texts: list[str],
    labels: list[int],
    probabilities: np.ndarray,
    threshold: float = 0.5,
    confidence_threshold: float = 0.6,
    vocabulary: Vocabulary | None = None,
    max_length: int = 256,
    truncation_strategy: str = "head_tail",
) -> dict[str, Any]:
    """Đo lát cắt độ dài, truncation/OOV và selective coverage."""
    preds = (probabilities >= threshold).astype(int)
    corrects = (preds == np.array(labels)).astype(int)

    # 1. Length Slices
    token_lens = [len(tokenize(t)) for t in texts]
    length_slices = {
        "0-64 tokens": [],
        "65-128 tokens": [],
        "129-256 tokens": [],
        ">256 tokens": [],
    }
    for i, t_len in enumerate(token_lens):
        if t_len <= 64:
            length_slices["0-64 tokens"].append(corrects[i])
        elif t_len <= 128:
            length_slices["65-128 tokens"].append(corrects[i])
        elif t_len <= 256:
            length_slices["129-256 tokens"].append(corrects[i])
        else:
            length_slices[">256 tokens"].append(corrects[i])

    length_report = {}
    for k, v in length_slices.items():
        length_report[k] = {
            "samples": len(v),
            "accuracy": round(float(np.mean(v)), 4) if v else 0.0,
        }

    report: dict[str, Any] = {"length_slices": length_report}

    confidence = np.maximum(probabilities, 1.0 - probabilities)
    accepted = confidence >= confidence_threshold
    report["selective"] = {
        "confidence_threshold": confidence_threshold,
        "coverage": round(float(np.mean(accepted)) if len(accepted) else 0.0, 4),
        "accepted_accuracy": round(
            float(np.mean((preds[accepted] == np.asarray(labels)[accepted])))
            if np.any(accepted)
            else 0.0,
        ),
        "review_rate": round(float(np.mean(~accepted)) if len(accepted) else 0.0, 4),
    }

    if vocabulary is not None:
        audits = [
            encode_with_audit(text, vocabulary, max_length, strategy=truncation_strategy)[2]
            for text in texts
        ]
        quality_masks = {
            "truncated": np.array([audit["is_truncated"] for audit in audits]),
            "not_truncated": np.array([not audit["is_truncated"] for audit in audits]),
            "low_used_oov": np.array([audit["used_oov_rate"] <= 0.20 for audit in audits]),
            "high_used_oov": np.array([audit["used_oov_rate"] > 0.20 for audit in audits]),
        }
        report["quality_slices"] = {
            name: {
                "samples": int(mask.sum()),
                "accuracy": round(float(np.mean(corrects[mask])) if np.any(mask) else 0.0, 4),
            }
            for name, mask in quality_masks.items()
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình Champion trên tập kiểm thử chính thức (Locked Official Test)"
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/releases/v1.0.0",
        help="Thư mục chứa checkpoint mô hình Champion (ví dụ: artifacts/bilstm hoặc artifacts/baseline).",
    )
    parser.add_argument("--train-data", default="data/raw/train.csv")
    parser.add_argument("--test-data", default="data/raw/test.csv")
    parser.add_argument("--output-report", default="artifacts/releases/v1.0.0/final_test_report.md")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print("=" * 65)
    print("*** KHOI CHAY DANH GIA OFFICIAL TEST CHO CHAMPION ***")
    print("=" * 65)
    print(f"-> Thu muc Champion: {model_dir}")
    print(f"-> Tap du lieu Test: {args.test_data}")
    print("-" * 65)

    # 1. Đọc nguyên vẹn hai official split và chỉ audit overlap.
    train_source = load_official_dataset(args.train_data)
    test_frame = load_official_dataset(args.test_data)
    overlap = validate_official_test_independence(train_source, test_frame)
    test_texts = test_frame["text"].tolist()
    test_labels = test_frame["label"].astype(int).tolist()
    print(f"-> Official Test samples: {len(test_frame):,}")
    print(f"-> Overlap audit       : {overlap['overlap_count']} mẫu")

    device = select_device(args.device)
    metrics: dict[str, Any] = {}
    threshold = 0.5
    confidence_threshold = 0.5
    temperature = 1.0
    evaluation_vocabulary: Vocabulary | None = None
    truncation_strategy = "head_tail"
    max_length = 256

    checkpoint_path = model_dir / "model.pt"
    baseline_path = model_dir / "model.joblib"

    if checkpoint_path.is_file():
        # Đánh giá PyTorch RNN Champion
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        config = ExperimentConfig(**checkpoint["config"])
        vocab = Vocabulary.from_dict(checkpoint["vocabulary"])
        temperature = float(checkpoint.get("temperature") or 1.0)
        threshold = float(checkpoint.get("decision_threshold") or 0.5)
        confidence_threshold = float(checkpoint.get("confidence_threshold") or 0.6)
        evaluation_vocabulary = vocab
        truncation_strategy = config.truncation_strategy
        max_length = config.max_length

        expected_test_hash = checkpoint.get("official_test_hash")
        if expected_test_hash not in {None, "", "unspecified"}:
            actual_test_hash = compute_dataset_hash(test_frame)
            if actual_test_hash != expected_test_hash:
                raise ValueError(
                    "Official Test hash không khớp artifact; dừng để bảo vệ locked benchmark."
                )

        model = SentimentRNN(len(vocab), vocab.pad_index, config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        test_dataset = IMDBDataset(
            test_frame,
            vocab,
            config.max_length,
            strategy=getattr(config, "truncation_strategy", "first"),
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        loss_function = torch.nn.BCEWithLogitsLoss()
        metrics = evaluate_model(
            model,
            test_loader,
            loss_function,
            device,
            temperature=temperature,
            decision_threshold=threshold,
            return_raw=True,
        )

        # Tính toán lát cắt lỗi
        raw_logits = metrics.pop("raw_logits")
        metrics.pop("raw_labels")
        scaled_logits = raw_logits / max(1e-4, temperature)
        scaled_logits = np.clip(scaled_logits, -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        slice_report = evaluate_slices(
            test_texts,
            test_labels,
            probs,
            threshold,
            confidence_threshold,
            vocab,
            config.max_length,
            config.truncation_strategy,
        )
        metrics["error_slices"] = slice_report

        model_name = config.model_type.upper()

    elif baseline_path.is_file():
        # Đánh giá Scikit-Learn Baseline Champion
        from baseline import evaluate_baseline

        baseline_metadata_path = model_dir / "baseline_metadata.json"
        if baseline_metadata_path.is_file():
            baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
            expected_test_hash = baseline_metadata.get("official_test_hash")
            if expected_test_hash not in {None, "", "unspecified"}:
                actual_test_hash = compute_dataset_hash(test_frame)
                if actual_test_hash != expected_test_hash:
                    raise ValueError(
                        "Official Test hash không khớp baseline artifact; "
                        "dừng để bảo vệ locked benchmark."
                    )
        pipeline = joblib.load(baseline_path)
        metrics = evaluate_baseline(pipeline, test_texts, test_labels)
        probs = pipeline.predict_proba(test_texts)[:, 1]
        slice_report = evaluate_slices(
            test_texts,
            test_labels,
            probs,
            threshold=0.5,
            confidence_threshold=0.5,
        )
        metrics["error_slices"] = slice_report
        model_name = "TF-IDF + LOGISTIC REGRESSION (BASELINE)"

    else:
        raise FileNotFoundError(f"Không tìm thấy model.pt hoặc model.joblib tại {model_dir}")

    evaluation_records = build_evaluation_records(
        test_texts,
        test_labels,
        probs,
        threshold,
        confidence_threshold,
        evaluation_vocabulary,
        max_length,
        truncation_strategy,
    )
    evaluation_records.to_csv(model_dir / "evaluation_records.csv", index=False)
    benchmark = benchmark_inference(model_dir, test_texts)
    metrics["benchmark"] = benchmark
    (model_dir / "benchmark.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Lưu test_metrics.json
    (model_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Tạo báo cáo Markdown final_test_report.md
    slices_table_rows = []
    for k, v in metrics["error_slices"]["length_slices"].items():
        slices_table_rows.append(f"| {k} | {v['samples']:,} | {v['accuracy']:.2%} |")

    slices_md = "\n".join(slices_table_rows)

    report_md = f"""# 🔒 Official Locked Final Test Report — {model_name}

> **Chính sách đánh giá (Frozen Test Policy):**
> Mô hình Champion được khóa chặt cấu hình và chỉ được mở tập Test chính thức đúng **MỘT LẦN DUY NHẤT**.
> Không có bất kỳ sự can thiệp hay tinh chỉnh siêu tham số nào trên tập dữ liệu này.

## 1. Các Chỉ Số Tổng Quan (Core Metrics)
- **Mô hình Champion:** `{model_name}`
    - **Test Samples:** {len(test_frame):,} mẫu từ Official Test (không chỉnh sửa)
- **Test Accuracy:** **{metrics["accuracy"]:.2%}**
- **Test Macro-F1:** **{metrics["macro_f1"]:.2%}**
- **ROC-AUC:** {metrics["roc_auc"]:.4f}
- **PR-AUC:** {metrics["pr_auc"]:.4f}
- **Brier Score (Calibration):** {metrics["brier_score"]:.4f}
- **Expected Calibration Error (ECE):** {metrics["ece"]:.4f}
- **Log Loss:** {metrics["log_loss"]:.4f}
- **Selective Coverage:** {metrics["error_slices"]["selective"]["coverage"]:.2%}
- **Accepted Accuracy:** {metrics["error_slices"]["selective"]["accepted_accuracy"]:.2%}
- **Review Rate:** {metrics["error_slices"]["selective"]["review_rate"]:.2%}

## 2. Ma Trận Nhầm Lẫn (Confusion Matrix)
```text
                  Predicted Negative    Predicted Positive
Actual Negative :        {metrics["confusion_matrix"][0][0]:<12}        {metrics["confusion_matrix"][0][1]:<12}
Actual Positive :        {metrics["confusion_matrix"][1][0]:<12}        {metrics["confusion_matrix"][1][1]:<12}
```

## 3. Lát Cắt Lỗi Theo Độ Dài Chuỗi (Length Error Slices)
| Token Length Range | Số Mẫu (Samples) | Test Accuracy |
|---|---:|---:|
{slices_md}

## 4. Facts để diễn giải
- **Official Test hash:** `{compute_dataset_hash(test_frame)}`
- **Overlap audit:** `{overlap["overlap_count"]}` mẫu
- **Temperature:** `{metrics.get("temperature", 1.0):.4f}`
- **Long-review slice accuracy:** `{metrics["error_slices"]["length_slices"][">256 tokens"]["accuracy"]:.2%}` trên `{metrics["error_slices"]["length_slices"][">256 tokens"]["samples"]:,}` mẫu
- **CPU latency p50/p95:** `{benchmark["latency_p50_ms"]:.2f} / {benchmark["latency_p95_ms"]:.2f}` ms
- **CPU throughput:** `{benchmark["throughput_samples_per_second"]:.2f}` samples/second

> Báo cáo chỉ xuất số liệu và facts. Kết luận chất lượng cần dựa trên ngưỡng
> được định trước, không được sinh tự động từ một câu khẳng định chung chung.
"""

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 65)
    print("🔒 KET QUA DANH GIA OFFICIAL TEST:")
    print(f"-> Accuracy      : {metrics['accuracy']:.2%}")
    print(f"-> Macro-F1      : {metrics['macro_f1']:.2%}")
    print(f"-> ROC-AUC       : {metrics['roc_auc']:.4f}")
    print(f"-> Brier Score   : {metrics['brier_score']:.4f}")
    print(f"-> ECE           : {metrics['ece']:.4f}")
    print(f"-> Da xuat bao cao tai: {report_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
