"""Tổng hợp và so sánh kết quả của các mô hình (LSTM, GRU, BiLSTM) thành bảng Markdown.

Script này tự động đọc các tệp `metrics.json` và `model.pt` từ thư mục artifacts,
đo lường số lượng tham số, độ trễ suy luận (latency) và các chỉ số (Accuracy,
Loss, Macro F1) để tạo bảng báo cáo so sánh chuẩn CV/Portfolio.
"""

import argparse
import json
import time
from pathlib import Path

import joblib

def read_metrics(artifact_root: Path, model_name: str) -> dict:
    """Đọc tệp JSON chứa kết quả đánh giá của mô hình."""
    path = artifact_root / model_name / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có tệp metrics.json cho mô hình '{model_name}' tại: {path}. "
            f"Hãy huấn luyện mô hình trước với lệnh: python train.py --model {model_name}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_model_info(artifact_root: Path, model_name: str) -> tuple[int, float]:
    """Lấy số tham số và đo độ trễ suy luận (Inference Latency) từ checkpoint nếu có."""
    # Trì hoãn import PyTorch để chức năng tạo bảng vẫn dùng được khi chưa cài model runtime.
    from sentiment.inference import SentimentPredictor
    from sentiment.utils import measure_latency

    checkpoint_path = artifact_root / model_name / "model.pt"
    if not checkpoint_path.is_file():
        return 0, 0.0

    predictor = SentimentPredictor(checkpoint_path, device="cpu")
    param_count = predictor.model.count_parameters()
    sample_text = "This movie has an outstanding storyline and wonderful performances!"
    latency_ms = measure_latency(predictor, sample_text, runs=30)
    return param_count, latency_ms


def get_baseline_info(artifact_root: Path) -> tuple[int, float]:
    """Đo số hệ số và latency của baseline scikit-learn trên CPU."""
    model_path = artifact_root / "baseline" / "model.joblib"
    if not model_path.is_file():
        return 0, 0.0
    pipeline = joblib.load(model_path)
    classifier = pipeline.named_steps["classifier"]
    coefficient_count = int(classifier.coef_.size + classifier.intercept_.size)
    sample = ["This movie has an outstanding storyline and wonderful performances!"]
    for _ in range(5):
        pipeline.predict_proba(sample)
    started = time.perf_counter()
    for _ in range(30):
        pipeline.predict_proba(sample)
    latency_ms = (time.perf_counter() - started) / 30 * 1_000
    return coefficient_count, latency_ms


def create_markdown(rows: list[dict]) -> str:
    """Tạo bảng Markdown so sánh các mô hình cho README và CV."""
    lines = [
        "| Model | Parameters | Test Loss | Accuracy | Macro F1 | Latency (CPU) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        params = row.get("params", 0)
        latency = row.get("latency", 0.0)
        params_str = f"{params:,}" if params > 0 else "N/A"
        latency_str = f"{latency:.2f} ms" if latency > 0 else "N/A"
        lines.append(
            f"| **{row['model']}** | {params_str} | {row['loss']:.4f} | "
            f"{row['accuracy']:.2%} | {row['macro_f1']:.2%} | {latency_str} |"
        )
    return "\n".join(lines)

def main() -> None:
    """Luồng tổng hợp kết quả chính."""
    parser = argparse.ArgumentParser(
        description="Tổng hợp và so sánh kết quả các mô hình LSTM, GRU và BiLSTM"
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Thư mục gốc chứa các kết quả thí nghiệm. Mặc định là 'artifacts'.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/model_comparison.md",
        help="Tệp đầu ra Markdown. Mặc định là 'artifacts/model_comparison.md'.",
    )
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    rows = []

    print("Đang đọc chỉ số đánh giá của các mô hình (LSTM, GRU, BiLSTM)...")
    for model_name in ("baseline", "lstm", "gru", "bilstm"):
        metrics = read_metrics(artifact_root, model_name)
        params, latency = (
            get_baseline_info(artifact_root)
            if model_name == "baseline"
            else get_model_info(artifact_root, model_name)
        )

        rows.append(
            {
                "model": model_name.upper(),
                "params": params,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"],
                "latency": latency,
            }
        )

    markdown_table = create_markdown(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_table + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("📊 BẢNG SO SÁNH BENCHMARK CÁC MÔ HÌNH")
    print("=" * 60)
    print(markdown_table)
    print("=" * 60)
    print(f"\nĐã xuất kết quả so sánh tại: {output_path.resolve()}")


if __name__ == "__main__":
    main()
