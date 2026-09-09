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

import numpy as np
import pandas as pd

# Patterns nhận diện taxonomy ngôn ngữ học
NEGATION_PATTERN = re.compile(
    r"\b(not|never|no|hardly|barely|scarcely|without)\b|n't\b", re.IGNORECASE
)
MIXED_PATTERN = re.compile(
    r"\b(but|however|although|though|except|yet|despite|in spite of)\b", re.IGNORECASE
)
REVERSAL_PATTERN = re.compile(
    r"\b(at first|initially|started (out|well)|turned out|until the ending)\b", re.IGNORECASE
)
INTENSIFIER_PATTERN = re.compile(
    r"\b(absolutely|totally|utterly|completely|extremely|truly)\b", re.IGNORECASE
)


def _coerce_bool(values: pd.Series) -> pd.Series:
    """Đọc bool ổn định dù CSV lưu True/False dạng bool, số hoặc chuỗi."""
    if pd.api.types.is_bool_dtype(values):
        return values
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(int).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


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


def analyze_linguistic_slices(df_errors: pd.DataFrame, df_total: pd.DataFrame) -> dict[str, Any]:
    """Đo lường tỷ lệ lỗi trong các nhóm ngữ cảnh ngôn ngữ học."""
    patterns = {
        "has_negation": (NEGATION_PATTERN, "Negation ('not', \"n't\", 'never')"),
        "has_mixed_sentiment": (MIXED_PATTERN, "Mixed Sentiment ('but', 'however')"),
        "has_sentiment_reversal": (
            REVERSAL_PATTERN,
            "Sentiment Reversal ('at first', 'turned out')",
        ),
        "has_intensifier": (INTENSIFIER_PATTERN, "Intensifiers ('absolutely', 'utterly')"),
    }

    def compute_pattern_stats(column: str, pattern: re.Pattern, name: str) -> dict[str, Any]:
        if column in df_total.columns:
            in_total = _coerce_bool(df_total[column]).sum()
            in_errors = _coerce_bool(df_errors[column]).sum()
        elif "text" in df_total.columns:
            in_total = df_total["text"].str.contains(pattern, regex=True).sum()
            in_errors = df_errors["text"].str.contains(pattern, regex=True).sum()
        else:
            in_total = in_errors = 0
        error_rate = in_errors / in_total if in_total > 0 else 0.0
        return {
            "name": name,
            "total_matching_samples": int(in_total),
            "errors_count": int(in_errors),
            "error_rate": round(float(error_rate), 4),
        }

    return {
        column.removeprefix("has_"): compute_pattern_stats(column, pattern, name)
        for column, (pattern, name) in patterns.items()
    }


def main() -> None:
    """Đọc prediction facts đã khóa và không mở Official Test lần hai."""
    parser = argparse.ArgumentParser(
        description="Phân tích lỗi từ locked evaluation records của CineSentiment"
    )
    parser.add_argument(
        "--evaluation-records",
        default="artifacts/releases/v1.0.0/evaluation_records.csv",
        help="Bảng facts do evaluate_release tạo ra.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    records_path = Path(args.evaluation_records)
    if not records_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy evaluation records tại: {records_path}")

    output_dir = Path(args.output_dir or records_path.parent / "error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(records_path)
    required = {"label", "prediction", "positive_probability", "confidence", "is_error"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Evaluation records thiếu cột: {', '.join(sorted(missing))}")
    records["is_error"] = _coerce_bool(records["is_error"])
    df_errors = records.loc[records["is_error"]].sort_values("confidence", ascending=False).copy()

    linguistic_summary = analyze_linguistic_slices(df_errors, records)
    representatives = []
    for row in df_errors.head(10).itertuples(index=False):
        representatives.append(
            {
                "row_id": int(getattr(row, "row_id", -1)),
                "label": int(row.label),
                "prediction": int(row.prediction),
                "positive_probability": round(float(row.positive_probability), 4),
                "confidence": round(float(row.confidence), 4),
            }
        )

    summary = {
        "evaluation_records": str(records_path),
        "total_evaluated_samples": len(records),
        "total_errors": len(df_errors),
        "overall_error_rate": round(len(df_errors) / len(records), 4) if len(records) else 0.0,
        "false_positives": int(((df_errors["label"] == 0) & (df_errors["prediction"] == 1)).sum()),
        "false_negatives": int(((df_errors["label"] == 1) & (df_errors["prediction"] == 0)).sum()),
        "linguistic_taxonomy": linguistic_summary,
        "representative_high_confidence_errors": representatives,
    }

    df_errors.head(max(0, args.top)).to_csv(output_dir / "high_confidence_errors.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Đã phân tích {len(records):,} prediction facts từ {records_path}.")
    print(f"Tổng lỗi: {len(df_errors):,} ({summary['overall_error_rate']:.2%})")
    print(f"Đã lưu kết quả tại: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
