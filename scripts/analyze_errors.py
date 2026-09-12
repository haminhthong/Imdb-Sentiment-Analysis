"""Phân tích chuyên sâu lỗi dự đoán (Error Analysis) cho mô hình phân loại cảm xúc.

Phân tích các lát cắt lỗi cốt lõi:
1. Lỗi có độ tin cậy cao (High-confidence errors): các trường hợp model tự tin sai.
2. Lát cắt phủ định (Negation): chứa 'not', "n't", 'never', 'hardly'.
3. Lát cắt cảm xúc hỗn hợp (Mixed sentiment): chứa 'but', 'however', 'although'.
4. Lát cắt đảo chiều cảm xúc (Sentiment reversal): chứa 'at first', 'initially', 'turned out'.
5. Lát cắt theo độ dài câu (Length slices): short (<=100), medium (101-256), long (>256).
6. Lát cắt theo tỷ lệ từ ngoài từ điển (OOV slices): low (<=5%), medium (5-20%), high (>20%).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment.data_validation import load_dataset
from sentiment.inference import load_predictor
from sentiment.text import tokenize
from sentiment.utils import configure_utf8_output

NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no|hardly|barely|scarcely|without)\b|n't\b", re.IGNORECASE
)
MIXED_PATTERN = re.compile(
    r"\b(?:but|however|although|though|except|yet|despite|in spite of)\b", re.IGNORECASE
)
REVERSAL_PATTERN = re.compile(
    r"\b(?:at first|initially|started (?:out|well)|turned out|until the ending)\b", re.IGNORECASE
)
INTENSIFIER_PATTERN = re.compile(
    r"\b(?:absolutely|totally|utterly|completely|extremely|truly)\b", re.IGNORECASE
)


def analyze_errors_on_dataset(
    predictor,
    df: pd.DataFrame,
    max_samples: int = 2500,
) -> dict[str, Any]:
    """Chạy suy luận và tính toán các thống kê lỗi theo lát cắt ngôn ngữ học."""
    sample_df = df.head(max_samples).copy()
    texts = sample_df["text"].tolist()
    labels = sample_df["label"].astype(int).tolist()

    results = predictor.predict_batch(texts)

    predictions = [1 if r.label == "Positive" else 0 for r in results]
    probs = [r.probability for r in results]
    confidences = [max(p, 1.0 - p) for p in probs]
    oov_rates = [r.oov_rate for r in results]
    lengths = [len(tokenize(t)) for t in texts]
    is_errors = [p != y for p, y in zip(predictions, labels, strict=False)]

    records = pd.DataFrame(
        {
            "text": texts,
            "label": labels,
            "prediction": predictions,
            "probability": probs,
            "confidence": confidences,
            "length": lengths,
            "oov_rate": oov_rates,
            "is_error": is_errors,
        }
    )

    total_samples = len(records)
    total_errors = int(records["is_error"].sum())
    overall_error_rate = total_errors / total_samples if total_samples > 0 else 0.0

    # 1. Phân tích các lát cắt ngôn ngữ học
    patterns = {
        "negation": (NEGATION_PATTERN, "Negation (not, n't, never)"),
        "mixed_sentiment": (MIXED_PATTERN, "Mixed Sentiment (but, however, although)"),
        "sentiment_reversal": (REVERSAL_PATTERN, "Sentiment Reversal (at first, turned out)"),
        "intensifiers": (INTENSIFIER_PATTERN, "Intensifiers (absolutely, completely)"),
    }

    linguistic_slices = {}
    for key, (pattern, name) in patterns.items():
        matched = records["text"].str.contains(pattern, regex=True)
        count_matched = int(matched.sum())
        errors_in_slice = int((matched & records["is_error"]).sum())
        error_rate = errors_in_slice / count_matched if count_matched > 0 else 0.0
        linguistic_slices[key] = {
            "name": name,
            "total_count": count_matched,
            "error_count": errors_in_slice,
            "error_rate": round(error_rate, 4),
        }

    # 2. Phân tích lát cắt theo độ dài câu
    short_mask = records["length"] <= 100
    medium_mask = (records["length"] > 100) & (records["length"] <= 256)
    long_mask = records["length"] > 256

    length_slices = {
        "short_reviews (<=100)": {
            "total": int(short_mask.sum()),
            "errors": int((short_mask & records["is_error"]).sum()),
            "error_rate": round(
                float((short_mask & records["is_error"]).sum() / short_mask.sum())
                if short_mask.sum() > 0
                else 0.0,
                4,
            ),
        },
        "medium_reviews (101-256)": {
            "total": int(medium_mask.sum()),
            "errors": int((medium_mask & records["is_error"]).sum()),
            "error_rate": round(
                float((medium_mask & records["is_error"]).sum() / medium_mask.sum())
                if medium_mask.sum() > 0
                else 0.0,
                4,
            ),
        },
        "long_reviews (>256)": {
            "total": int(long_mask.sum()),
            "errors": int((long_mask & records["is_error"]).sum()),
            "error_rate": round(
                float((long_mask & records["is_error"]).sum() / long_mask.sum())
                if long_mask.sum() > 0
                else 0.0,
                4,
            ),
        },
    }

    # 3. Phân tích lát cắt theo OOV Rate
    low_oov = records["oov_rate"] <= 0.05
    med_oov = (records["oov_rate"] > 0.05) & (records["oov_rate"] <= 0.20)
    high_oov = records["oov_rate"] > 0.20

    oov_slices = {
        "low_oov (<=5%)": {
            "total": int(low_oov.sum()),
            "errors": int((low_oov & records["is_error"]).sum()),
            "error_rate": round(
                float((low_oov & records["is_error"]).sum() / low_oov.sum())
                if low_oov.sum() > 0
                else 0.0,
                4,
            ),
        },
        "medium_oov (5-20%)": {
            "total": int(med_oov.sum()),
            "errors": int((med_oov & records["is_error"]).sum()),
            "error_rate": round(
                float((med_oov & records["is_error"]).sum() / med_oov.sum())
                if med_oov.sum() > 0
                else 0.0,
                4,
            ),
        },
        "high_oov (>20%)": {
            "total": int(high_oov.sum()),
            "errors": int((high_oov & records["is_error"]).sum()),
            "error_rate": round(
                float((high_oov & records["is_error"]).sum() / high_oov.sum())
                if high_oov.sum() > 0
                else 0.0,
                4,
            ),
        },
    }

    # 4. Top High-confidence errors
    errors_only = records[records["is_error"]].sort_values("confidence", ascending=False)
    high_conf_examples = []
    for _, row in errors_only.head(5).iterrows():
        high_conf_examples.append(
            {
                "label": int(row["label"]),
                "predicted": int(row["prediction"]),
                "confidence": round(float(row["confidence"]), 4),
                "text_snippet": row["text"][:160] + "...",
            }
        )

    return {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "overall_error_rate": round(overall_error_rate, 4),
        "linguistic_slices": linguistic_slices,
        "length_slices": length_slices,
        "oov_slices": oov_slices,
        "high_confidence_examples": high_conf_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phân tích lỗi mô hình theo các lát cắt NLP")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    parser.add_argument("--data", default="data/raw/test.csv")
    parser.add_argument("--max-samples", type=int, default=2500)
    parser.add_argument("--output", default="artifacts/error_analysis.json")
    return parser.parse_args()


def main() -> None:
    configure_utf8_output()
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file() and Path("test.csv").is_file():
        data_path = Path("test.csv")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    predictor = load_predictor(checkpoint_path)
    df = load_dataset(data_path, deduplicate=False)

    print("=" * 65)
    print("*** PHÂN TÍCH LỖI MÔ HÌNH (ERROR ANALYSIS) ***")
    print("=" * 65)
    print(f"-> Checkpoint : {checkpoint_path}")
    print(f"-> Dữ liệu    : {data_path} ({len(df):,} mẫu)")
    print("-" * 65)

    report = analyze_errors_on_dataset(predictor, df, max_samples=args.max_samples)

    err_rate = report["overall_error_rate"]
    print(f"-> Tổng số mẫu đánh giá: {report['total_samples']:,}")
    print(f"-> Tổng số lỗi          : {report['total_errors']:,} ({err_rate:.2%})")
    print("\n--- Lát cắt Ngôn ngữ học ---")
    for _key, data in report["linguistic_slices"].items():
        name = data["name"]
        err_c = data["error_count"]
        tot_c = data["total_count"]
        rate = data["error_rate"]
        print(f"   • {name:<40}: {err_c:3d}/{tot_c:4d} ({rate:.2%})")

    print("\n--- Lát cắt Độ dài câu ---")
    for key, data in report["length_slices"].items():
        print(f"   • {key:<30}: {data['errors']:3d}/{data['total']:4d} ({data['error_rate']:.2%})")

    print("\n--- Lát cắt Tỷ lệ từ ngoài từ điển (OOV) ---")
    for key, data in report["oov_slices"].items():
        print(f"   • {key:<30}: {data['errors']:3d}/{data['total']:4d} ({data['error_rate']:.2%})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-" * 65)
    print(f"✅ Báo cáo chi tiết đã lưu tại: {out_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
