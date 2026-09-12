"""So sánh đối chiếu hiệu năng giữa Baseline (TF-IDF + LR) và Deep Model (BiLSTM).

Báo cáo so sánh gồm 4 chỉ số cốt lõi:
1. Accuracy: Độ chính xác tổng thể.
2. Macro-F1: Cân bằng giữa Precision và Recall trên cả hai lớp.
3. CPU Latency: Thời gian suy luận trung bình trên CPU (ms/sample).
4. Model Size / Parameters: Số lượng tham số và kích thước mô hình.
"""

import argparse
import json
import time
from pathlib import Path

import joblib

from sentiment.inference import SentimentPredictor
from sentiment.utils import configure_utf8_output, measure_latency


def get_baseline_stats(baseline_dir: Path) -> dict | None:
    """Đo số lượng hệ số, latency và đọc metrics của baseline."""
    model_path = baseline_dir / "model.joblib"
    metrics_path = baseline_dir / "validation_metrics.json"
    if not model_path.is_file() or not metrics_path.is_file():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(model_path)
    clf = pipeline.named_steps["classifier"]
    param_count = int(clf.coef_.size + clf.intercept_.size)

    sample = ["This movie has an outstanding storyline and wonderful performances!"]
    for _ in range(5):
        pipeline.predict_proba(sample)
    start = time.perf_counter()
    runs = 30
    for _ in range(runs):
        pipeline.predict_proba(sample)
    latency_ms = (time.perf_counter() - start) / runs * 1000

    return {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "brier_score": metrics.get("brier_score", 0.0),
        "ece": metrics.get("ece", 0.0),
        "params": param_count,
        "latency_ms": round(latency_ms, 2),
    }


def get_bilstm_stats(bilstm_dir: Path) -> dict | None:
    """Đo số lượng tham số, latency và đọc metrics của BiLSTM."""
    model_path = bilstm_dir / "model.pt"
    metrics_path = bilstm_dir / "validation_metrics.json"
    if not model_path.is_file() or not metrics_path.is_file():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictor = SentimentPredictor(model_path, device="cpu")
    param_count = predictor.model.count_parameters()

    sample = "This movie has an outstanding storyline and wonderful performances!"
    latency_ms = measure_latency(predictor, sample, runs=30)

    return {
        "model": "BiLSTM (Calibrated)",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "brier_score": metrics.get("brier_score", 0.0),
        "ece": metrics.get("ece", 0.0),
        "params": param_count,
        "latency_ms": round(latency_ms, 2),
    }


def format_table(rows: list[dict]) -> str:
    """Tạo bảng Markdown so sánh mô hình."""
    headers = ["Model", "Accuracy", "Macro-F1", "Brier Score", "ECE", "Parameters", "CPU Latency"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            f"| **{r['model']}** | {r['accuracy']:.2%} | {r['macro_f1']:.2%} | "
            f"{r['brier_score']:.4f} | {r['ece']:.4f} | {r['params']:,} | "
            f"{r['latency_ms']:.2f} ms |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh mô hình Baseline và BiLSTM")
    parser.add_argument("--baseline-dir", default="artifacts/baseline")
    parser.add_argument("--bilstm-dir", default="artifacts")
    parser.add_argument("--output", default="artifacts/model_comparison.md")
    return parser.parse_args()


def main() -> None:
    configure_utf8_output()
    args = parse_args()
    rows = []

    baseline_info = get_baseline_stats(Path(args.baseline_dir))
    if baseline_info:
        rows.append(baseline_info)

    bilstm_info = get_bilstm_stats(Path(args.bilstm_dir))
    if bilstm_info:
        rows.append(bilstm_info)

    if not rows:
        print("⚠️ Chưa tìm thấy kết quả của Baseline hoặc BiLSTM để so sánh.")
        print("Vui lòng chạy `python baseline.py` và `python train.py` trước.")
        return

    table_md = format_table(rows)
    print("\n" + "=" * 70)
    print(" BẢNG SO SÁNH ĐỐI CHIẾU MÔ HÌNH (MODEL COMPARISON)")
    print("=" * 70)
    print(table_md)
    print("=" * 70 + "\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_md + "\n", encoding="utf-8")
    print(f"✅ Bảng so sánh đã được lưu tại: {out_path.resolve()}")


if __name__ == "__main__":
    main()
