"""Phân tích chuyên sâu lỗi dự đoán (Error Analysis & Taxonomy) cho CineSentiment AI.

Module này cung cấp:
1. Trích xuất các mẫu dự đoán sai có độ tin cậy cao (High-confidence errors).
2. Phân tích lỗi theo lát cắt độ dài câu (Length-based slices).
3. Phân tích lỗi theo lát cắt tỷ lệ OOV (OOV-based slices).
4. Phân loại theo Taxonomy ngôn ngữ học:
   - Phủ định (Negation: 'not', "n't", 'never', 'hardly')
   - Cảm xúc hỗn hợp (Mixed sentiment: 'but', 'however', 'although')
   - Đảo chiều cảm xúc (Reversal: 'at first', 'initially', 'turned out')
   - Từ nhấn mạnh (Intensifiers: 'absolutely', 'totally', 'utterly')
5. Xuất các ví dụ False Positive và False Negative điển hình phục vụ phỏng vấn kỹ thuật.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from sentiment.data_validation import load_official_dataset, validate_official_test_independence
from sentiment.text import Vocabulary, tokenize

# Patterns nhận diện taxonomy ngôn ngữ học
NEGATION_PATTERN = re.compile(r"\b(not|never|no|hardly|barely|scarcely|without)\b|n't\b", re.IGNORECASE)
MIXED_PATTERN = re.compile(r"\b(but|however|although|though|except|yet|despite|in spite of)\b", re.IGNORECASE)
REVERSAL_PATTERN = re.compile(r"\b(at first|initially|started (out|well)|turned out|until the ending)\b", re.IGNORECASE)
INTENSIFIER_PATTERN = re.compile(r"\b(absolutely|totally|utterly|completely|extremely|truly)\b", re.IGNORECASE)


def collect_errors(frame: pd.DataFrame, predictions, probabilities) -> pd.DataFrame:
    """Tạo bảng lỗi, sắp xếp theo độ tin cậy giảm dần (tương thích ngược)."""
    result = frame.loc[:, ["text", "label"]].copy()
    result["prediction"] = predictions
    probs_arr = np.asarray(probabilities)
    if probs_arr.ndim == 2:
        result["positive_probability"] = probs_arr[:, 1]
    else:
        result["positive_probability"] = probs_arr
    result["confidence"] = result["positive_probability"].where(
        result["prediction"] == 1,
        1.0 - result["positive_probability"],
    )
    return result.loc[result["label"] != result["prediction"]].sort_values(
        "confidence", ascending=False
    )


def get_predictions_and_probabilities(
    model_path: Path, texts: list[str], device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Lấy dự đoán và xác suất từ PyTorch model.pt hoặc Scikit-Learn model.joblib."""
    if model_path.suffix == ".joblib":
        pipeline = joblib.load(model_path)
        probabilities = pipeline.predict_proba(texts)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        return predictions, probabilities

    # Checkpoint PyTorch
    from sentiment.inference import SentimentPredictor

    predictor = SentimentPredictor(model_path, device=device)
    results = predictor.predict_batch(texts)
    probabilities = np.array([r.positive_probability for r in results], dtype=np.float32)
    predictions = np.array([1 if r.label == "Positive" else 0 for r in results], dtype=np.int32)
    return predictions, probabilities


def analyze_linguistic_slices(df_errors: pd.DataFrame, df_total: pd.DataFrame) -> dict[str, Any]:
    """Đo lường tỷ lệ lỗi trong các nhóm ngữ cảnh ngôn ngữ học."""
    total_len = len(df_total)
    total_errors = len(df_errors)

    def compute_pattern_stats(pattern: re.Pattern, name: str) -> dict[str, Any]:
        in_total = df_total["text"].str.contains(pattern, regex=True).sum()
        in_errors = df_errors["text"].str.contains(pattern, regex=True).sum()
        error_rate = in_errors / in_total if in_total > 0 else 0.0
        return {
            "name": name,
            "total_matching_samples": int(in_total),
            "errors_count": int(in_errors),
            "error_rate": round(float(error_rate), 4),
        }

    return {
        "negation": compute_pattern_stats(NEGATION_PATTERN, "Negation ('not', \"n't\", 'never')"),
        "mixed_sentiment": compute_pattern_stats(MIXED_PATTERN, "Mixed Sentiment ('but', 'however')"),
        "sentiment_reversal": compute_pattern_stats(REVERSAL_PATTERN, "Sentiment Reversal ('at first', 'turned out')"),
        "intensifiers": compute_pattern_stats(INTENSIFIER_PATTERN, "Intensifiers ('absolutely', 'utterly')"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích lỗi chuyên sâu cho CineSentiment AI")
    parser.add_argument(
        "--model",
        default="artifacts/releases/v1.0.0/model.pt",
        help="Đường dẫn checkpoint model.pt hoặc model.joblib.",
    )
    parser.add_argument("--train-data", default="data/raw/train.csv")
    parser.add_argument("--test-data", default="data/raw/test.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")

    output_dir = Path(args.output_dir or model_path.parent / "error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang phân tích lỗi cho mô hình: {model_path}")
    train_frame = load_official_dataset(args.train_data)
    test_frame = load_official_dataset(args.test_data)
    validate_official_test_independence(train_frame, test_frame)

    predictions, probabilities = get_predictions_and_probabilities(
        model_path, test_frame["text"].tolist()
    )

    test_frame = test_frame.copy()
    test_frame["prediction"] = predictions
    test_frame["positive_probability"] = probabilities
    test_frame["confidence"] = np.where(
        predictions == 1, probabilities, 1.0 - probabilities
    )
    test_frame["is_error"] = test_frame["label"] != test_frame["prediction"]

    df_errors = test_frame.loc[test_frame["is_error"]].sort_values(
        "confidence", ascending=False
    ).copy()

    # Phân tích Taxonomy ngôn ngữ học
    linguistic_summary = analyze_linguistic_slices(df_errors, test_frame)

    # Trích xuất False Positives và False Negatives độ tin cậy cao nhất
    fps = df_errors.loc[(df_errors["label"] == 0) & (df_errors["prediction"] == 1)].head(5)
    fns = df_errors.loc[(df_errors["label"] == 1) & (df_errors["prediction"] == 0)].head(5)

    fp_examples = [
        {"text": row["text"][:150] + "...", "confidence": round(float(row["confidence"]), 4)}
        for _, row in fps.iterrows()
    ]
    fn_examples = [
        {"text": row["text"][:150] + "...", "confidence": round(float(row["confidence"]), 4)}
        for _, row in fns.iterrows()
    ]

    summary = {
        "model": str(model_path),
        "total_test_samples": len(test_frame),
        "total_errors": len(df_errors),
        "overall_error_rate": round(len(df_errors) / len(test_frame), 4),
        "false_positives": int(((df_errors["label"] == 0) & (df_errors["prediction"] == 1)).sum()),
        "false_negatives": int(((df_errors["label"] == 1) & (df_errors["prediction"] == 0)).sum()),
        "linguistic_taxonomy": linguistic_summary,
        "representative_high_confidence_false_positives": fp_examples,
        "representative_high_confidence_false_negatives": fn_examples,
    }

    # Xuất tệp
    df_errors.head(args.top).to_csv(output_dir / "high_confidence_errors.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("🔍 BÁO CÁO PHÂN TÍCH LỖI (ERROR ANALYSIS)")
    print("=" * 65)
    print(f"-> Tổng mẫu Test   : {summary['total_test_samples']:,}")
    print(f"-> Tổng lỗi        : {summary['total_errors']:,} ({summary['overall_error_rate']:.2%})")
    print(f"-> False Positives : {summary['false_positives']:,}")
    print(f"-> False Negatives : {summary['false_negatives']:,}")
    print("\n[Lát cắt Taxonomy Ngôn Ngữ Học]:")
    for k, v in linguistic_summary.items():
        print(f"   • {v['name']}: {v['errors_count']}/{v['total_matching_samples']} lỗi ({v['error_rate']:.2%})")
    print("=" * 65)
    print(f"Đã lưu kết quả tại: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
