"""Tổng hợp và so sánh kết quả Validation của các mô hình thành Development Leaderboard.

Quy trình chuẩn hóa:
1. Đọc tệp `validation_metrics.json` từ các thư mục thí nghiệm
   (`baseline`, `lstm`, `gru`, `bilstm`).
2. Đo lường số tham số và độ trễ CPU latency (ms/sample).
3. Áp dụng Champion Selection Policy (Primary: Validation Macro-F1;
   Guardrails: Log-Loss, latency, simplicity).
4. Xuất báo cáo `artifacts/development_leaderboard.md`.
5. Đóng băng mô hình Champion để chuẩn bị đánh giá Official Test duy nhất 1 lần
   bằng `scripts.evaluate_release`.
"""

import argparse
import json
import time
from pathlib import Path

import joblib


def read_validation_metrics(artifact_root: Path, model_name: str) -> dict | None:
    """Đọc tệp validation_metrics.json (hoặc tương thích ngược với metrics.json nếu có)."""
    locations = [artifact_root / model_name]
    if model_name.lower() == "bilstm":
        locations.append(Path("runs/dev_bilstm"))
    for location in locations:
        val_path = location / "validation_metrics.json"
        if val_path.is_file():
            return json.loads(val_path.read_text(encoding="utf-8"))

        fallback_path = location / "metrics.json"
        if fallback_path.is_file():
            return json.loads(fallback_path.read_text(encoding="utf-8"))

    return None


def get_model_info(artifact_root: Path, model_name: str) -> tuple[int, float]:
    """Lấy số tham số và đo độ trễ suy luận trên CPU."""
    from sentiment.inference import SentimentPredictor
    from sentiment.utils import measure_latency

    checkpoint_path = artifact_root / model_name / "model.pt"
    if model_name.lower() == "bilstm" and not checkpoint_path.is_file():
        checkpoint_path = Path("runs/dev_bilstm/best_dev.ckpt")
    if not checkpoint_path.is_file():
        return 0, 0.0

    try:
        predictor = SentimentPredictor(checkpoint_path, device="cpu")
        param_count = predictor.model.count_parameters()
        sample_text = "This movie has an outstanding storyline and wonderful performances!"
        latency_ms = measure_latency(predictor, sample_text, runs=20)
        return param_count, latency_ms
    except Exception:
        return 0, 0.0


def get_baseline_info(artifact_root: Path) -> tuple[int, float]:
    """Đo số hệ số và độ trễ CPU của baseline TF-IDF + Logistic Regression."""
    model_path = artifact_root / "baseline" / "model.joblib"
    if not model_path.is_file():
        return 0, 0.0

    try:
        pipeline = joblib.load(model_path)
        classifier = pipeline.named_steps["classifier"]
        coef_count = int(classifier.coef_.size + classifier.intercept_.size)
        sample = ["This movie has an outstanding storyline and wonderful performances!"]
        for _ in range(3):
            pipeline.predict_proba(sample)
        started = time.perf_counter()
        runs = 20
        for _ in range(runs):
            pipeline.predict_proba(sample)
        latency_ms = (time.perf_counter() - started) / runs * 1_000
        return coef_count, latency_ms
    except Exception:
        return 0, 0.0


def select_champion(rows: list[dict], simplicity_margin: float = 0.005) -> tuple[dict, str]:
    """Áp dụng Champion Selection Policy.

    Tiêu chí:
    1. Primary: Validation Macro-F1 cao nhất.
    2. Simplicity Margin: Nếu model phức tạp chỉ hơn model đơn giản dưới 0.5% Macro-F1,
       ưu tiên mô hình đơn giản hơn (ít tham số và latency thấp hơn).
    """
    if not rows:
        raise ValueError("Không có mô hình nào để lựa chọn champion.")

    # Sắp xếp theo Macro-F1 giảm dần
    sorted_by_f1 = sorted(rows, key=lambda x: x["macro_f1"], reverse=True)
    best_candidate = sorted_by_f1[0]

    baseline = next((row for row in rows if row.get("model") == "BASELINE"), None)
    if baseline is not None and best_candidate.get("model") == "BILSTM":
        baseline_delta = best_candidate["macro_f1"] - baseline["macro_f1"]
        if baseline_delta <= 0 and best_candidate.get("latency", 0.0) > baseline.get(
            "latency", 0.0
        ):
            return (
                baseline,
                "BiLSTM không cải thiện Validation Macro-F1 so với baseline và có "
                "latency cao hơn; baseline giữ vai trò release gate.",
            )

    # Kiểm tra xem có mô hình nhẹ hơn nằm trong biên độ dung sai không
    complexity_rank = {"BASELINE": 1, "BILSTM": 2}

    chosen = best_candidate
    reason = f"Đạt Validation Macro-F1 cao nhất ({chosen['macro_f1']:.2%})."

    for candidate in sorted_by_f1[1:]:
        delta_f1 = best_candidate["macro_f1"] - candidate["macro_f1"]
        curr_rank = complexity_rank.get(candidate["model"], 99)
        best_rank = complexity_rank.get(best_candidate["model"], 99)

        # Nếu model nhẹ hơn đáng kể và delta_f1 < simplicity_margin
        if delta_f1 < simplicity_margin and curr_rank < best_rank:
            chosen = candidate
            reason = (
                f"Lựa chọn theo nguyên lý Occam's Razor: {candidate['model']} có Macro-F1 "
                f"({candidate['macro_f1']:.2%}) tiệm cận {best_candidate['model']} "
                f"({best_candidate['macro_f1']:.2%}, chênh lệch chỉ {delta_f1:.2%}), "
                f"nhưng có cấu trúc đơn giản và latency thấp hơn đáng kể."
            )
            break

    return chosen, reason


def create_markdown(rows: list[dict]) -> str:
    """Tạo bảng Markdown so sánh các mô hình (hỗ trợ tương thích ngược)."""
    lines = [
        "| Model | Parameters | Val Loss | Accuracy | Macro F1 | Latency (CPU) |",
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


def create_leaderboard_markdown(rows: list[dict], champion: dict, reason: str) -> str:
    """Tạo bảng Markdown báo cáo Development Leaderboard và Champion."""
    release_command = (
        "python -m scripts.package_baseline_release && python -m scripts.evaluate_release"
        if champion["model"] == "BASELINE"
        else (
            "python -m scripts.final_fit && python -m scripts.calibrate && "
            "python -m scripts.evaluate_release"
        )
    )
    lines = [
        "# 🏆 Development Validation Leaderboard",
        "",
        "> **Giao thức chuẩn (Leakage-Safe Protocol):** Bảng này được xây dựng "
        "**100% từ tập Validation**.",
        "> Tuyệt đối không sử dụng tập Test để so sánh hay chọn mô hình, nhằm "
        "tránh rò rỉ dữ liệu (Test Peeking).",
        "",
        "| Candidate Model | Parameters | Val Loss | Val Acc | Val Macro-F1 | "
        "Val ROC-AUC | Val ECE | CPU Latency | Champion Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for row in sorted(rows, key=lambda x: x["macro_f1"], reverse=True):
        is_champ = row["model"] == champion["model"]
        champ_tag = "🥇 **CHAMPION**" if is_champ else "Candidate"
        params = row.get("params", 0)
        latency = row.get("latency", 0.0)
        params_str = f"{params:,}" if params > 0 else "N/A"
        latency_str = f"{latency:.2f} ms" if latency > 0 else "N/A"
        ece_str = f"{row.get('ece', 0.0):.4f}" if "ece" in row else "N/A"
        auc_str = f"{row.get('roc_auc', 0.0):.4f}" if "roc_auc" in row else "N/A"

        lines.append(
            f"| **{row['model']}** | {params_str} | {row['loss']:.4f} | "
            f"{row['accuracy']:.2%} | **{row['macro_f1']:.2%}** | {auc_str} | {ece_str} | "
            f"{latency_str} | {champ_tag} |"
        )

    lines.extend(
        [
            "",
            "## 🎯 Quyết Định Lựa Chọn Champion (Champion Selection Policy)",
            f"- **Mô hình được chọn làm Champion:** `{champion['model']}`",
            f"- **Lý do lựa chọn:** {reason}",
            f"- **Bước tiếp theo:** Đóng băng toàn bộ checkpoint của "
            f"`{champion['model']}` và chỉ mở tập Official Test một lần duy nhất "
            "với lệnh:",
            "  ```bash",
            f"  {release_command}",
            "  ```",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tổng hợp Development Validation Leaderboard và lựa chọn Champion"
    )
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--output", default="artifacts/development_leaderboard.md")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    rows = []

    print("Đang đọc Validation Metrics từ các thư mục thí nghiệm...")
    for model_name in ("baseline", "bilstm"):
        metrics = read_validation_metrics(artifact_root, model_name)
        if metrics is None:
            continue

        params, latency = (
            get_baseline_info(artifact_root)
            if model_name == "baseline"
            else get_model_info(artifact_root, model_name)
        )

        macro_f1 = metrics.get("macro_f1")
        if macro_f1 is None and "classification_report" in metrics:
            macro_f1 = metrics["classification_report"].get("macro avg", {}).get("f1-score", 0.0)

        rows.append(
            {
                "model": model_name.upper(),
                "params": params,
                "loss": metrics.get("log_loss", metrics.get("loss", 0.0)),
                "accuracy": metrics.get("accuracy", 0.0),
                "macro_f1": float(macro_f1 or 0.0),
                "roc_auc": float(metrics.get("roc_auc", 0.0)),
                "ece": float(metrics.get("ece", 0.0)),
                "latency": latency,
            }
        )

    if not rows:
        print(f"[!] Chưa tìm thấy kết quả huấn luyện nào trong {artifact_root.resolve()}.")
        print(
            "Hãy huấn luyện baseline và các mô hình trước: "
            "python baseline.py && python train.py --model bilstm"
        )
        return

    champion, reason = select_champion(rows)
    md_content = create_leaderboard_markdown(rows, champion, reason)

    (artifact_root / "champion.json").write_text(
        json.dumps(
            {
                "model": champion["model"],
                "reason": reason,
                "validation_rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_content, encoding="utf-8")

    print("\n" + "=" * 65)
    print("📊 DEVELOPMENT VALIDATION LEADERBOARD")
    print("=" * 65)
    print(md_content)
    print("=" * 65)
    print(f"\nĐã ghi báo cáo leaderboard tại: {output_path.resolve()}")


if __name__ == "__main__":
    main()
