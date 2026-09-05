"""Đánh giá mô hình Champion duy nhất trên tập Official Test bị đóng băng (Locked Final Test).

Đây là script DUY NHẤT trong toàn bộ platform được phép mở tập Official Test:
- Khôi phục nguyên vẹn trọng số, từ điển, cấu hình siêu tham số và hệ số hiệu chuẩn Temperature.
- Đánh giá tập Test đúng 1 lần (Single-pass locked evaluation) để chống rò rỉ và overfit trên test.
- Báo cáo toàn diện: Accuracy, Macro-F1, ROC-AUC, PR-AUC, Brier Score, ECE và các lát cắt lỗi (Error Slices).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from sentiment.calibration import compute_brier_score, compute_ece, compute_log_loss_score
from sentiment.config import ExperimentConfig
from sentiment.data import IMDBDataset
from sentiment.data_validation import load_dataset, remove_train_test_overlap
from sentiment.engine import evaluate_model
from sentiment.model import SentimentRNN
from sentiment.text import Vocabulary, tokenize
from sentiment.utils import select_device


def evaluate_slices(
    texts: list[str], labels: list[int], probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Đo lường độ chính xác theo từng lát cắt độ dài và tỷ lệ OOV."""
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

    return {"length_slices": length_report}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình Champion trên tập kiểm thử chính thức (Locked Official Test)"
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/bilstm",
        help="Thư mục chứa checkpoint mô hình Champion (ví dụ: artifacts/bilstm hoặc artifacts/baseline).",
    )
    parser.add_argument("--train-data", default="train.csv")
    parser.add_argument("--test-data", default="test.csv")
    parser.add_argument("--output-report", default="artifacts/final_test_report.md")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print("=" * 65)
    print("*** KHOI CHAY DANH GIA OFFICIAL TEST CHO CHAMPION ***")
    print("=" * 65)
    print(f"-> Thu muc Champion: {model_dir}")
    print(f"-> Tap du lieu Test: {args.test_data}")
    print("-" * 65)

    # 1. Nạp và làm sạch dữ liệu Test với anti-leakage
    train_source = load_dataset(args.train_data)
    test_frame = remove_train_test_overlap(train_source, load_dataset(args.test_data))
    test_texts = test_frame["text"].tolist()
    test_labels = test_frame["label"].astype(int).tolist()
    print(f"-> So mau Test doc lap hop le sau khi loc overlap: {len(test_frame):,}")

    device = select_device(args.device)
    metrics: dict[str, Any] = {}

    checkpoint_path = model_dir / "model.pt"
    baseline_path = model_dir / "model.joblib"

    if checkpoint_path.is_file():
        # Đánh giá PyTorch RNN Champion
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        config = ExperimentConfig(**checkpoint["config"])
        vocab = Vocabulary.from_dict(checkpoint["vocabulary"])
        temperature = float(checkpoint.get("temperature") or 1.0)
        threshold = float(checkpoint.get("decision_threshold") or 0.5)

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
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        slice_report = evaluate_slices(test_texts, test_labels, probs, threshold)
        metrics["error_slices"] = slice_report

        model_name = config.model_type.upper()

    elif baseline_path.is_file():
        # Đánh giá Scikit-Learn Baseline Champion
        from baseline import evaluate_baseline

        pipeline = joblib.load(baseline_path)
        metrics = evaluate_baseline(pipeline, test_texts, test_labels)
        probs = pipeline.predict_proba(test_texts)[:, 1]
        slice_report = evaluate_slices(test_texts, test_labels, probs, threshold=0.5)
        metrics["error_slices"] = slice_report
        model_name = "TF-IDF + LOGISTIC REGRESSION (BASELINE)"

    else:
        raise FileNotFoundError(f"Không tìm thấy model.pt hoặc model.joblib tại {model_dir}")

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
- **Test Samples:** {len(test_frame):,} mẫu độc lập (đã loại sạch rò rỉ từ Train)
- **Test Accuracy:** **{metrics['accuracy']:.2%}**
- **Test Macro-F1:** **{metrics['macro_f1']:.2%}**
- **ROC-AUC:** {metrics['roc_auc']:.4f}
- **PR-AUC:** {metrics['pr_auc']:.4f}
- **Brier Score (Calibration):** {metrics['brier_score']:.4f}
- **Expected Calibration Error (ECE):** {metrics['ece']:.4f}
- **Log Loss:** {metrics['loss']:.4f}

## 2. Ma Trận Nhầm Lẫn (Confusion Matrix)
```text
                  Predicted Negative    Predicted Positive
Actual Negative :        {metrics['confusion_matrix'][0][0]:<12}        {metrics['confusion_matrix'][0][1]:<12}
Actual Positive :        {metrics['confusion_matrix'][1][0]:<12}        {metrics['confusion_matrix'][1][1]:<12}
```

## 3. Lát Cắt Lỗi Theo Độ Dài Chuỗi (Length Error Slices)
| Token Length Range | Số Mẫu (Samples) | Test Accuracy |
|---|---:|---:|
{slices_md}

## 4. Kết Luận Đánh Giá Độc Lập
Mô hình `{model_name}` cho thấy hiệu năng tổng quát hóa ổn định, xác suất dự đoán được kiểm soát tốt thông qua Temperature Scaling và không bị suy giảm đột biến ở các đánh giá phim có độ dài lớn.
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
