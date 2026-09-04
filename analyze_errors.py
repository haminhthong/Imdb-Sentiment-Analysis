"""Xuất các dự đoán sai có độ tin cậy cao để phân tích lỗi baseline."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sentiment.data_validation import load_dataset, remove_train_test_overlap


def collect_errors(frame: pd.DataFrame, predictions, probabilities) -> pd.DataFrame:
    """Tạo bảng lỗi, sắp xếp theo độ tin cậy giảm dần."""
    result = frame.loc[:, ["text", "label"]].copy()
    result["prediction"] = predictions
    result["positive_probability"] = probabilities[:, 1]
    result["confidence"] = result["positive_probability"].where(
        result["prediction"] == 1,
        1 - result["positive_probability"],
    )
    return result.loc[result["label"] != result["prediction"]].sort_values(
        "confidence", ascending=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích dự đoán sai của baseline")
    parser.add_argument("--model", default="artifacts/baseline/model.joblib")
    parser.add_argument("--train-data", default="train.csv")
    parser.add_argument("--test-data", default="test.csv")
    parser.add_argument("--output-dir", default="artifacts/baseline/error_analysis")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()
    if args.top <= 0:
        raise ValueError("top phải lớn hơn 0.")

    train_frame = load_dataset(args.train_data)
    test_frame = remove_train_test_overlap(train_frame, load_dataset(args.test_data))
    pipeline = joblib.load(args.model)
    predictions = pipeline.predict(test_frame["text"])
    probabilities = pipeline.predict_proba(test_frame["text"])
    errors = collect_errors(test_frame, predictions, probabilities)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors.head(args.top).to_csv(output_dir / "high_confidence_errors.csv", index=False)
    summary = {
        "test_samples": len(test_frame),
        "total_errors": len(errors),
        "error_rate": len(errors) / len(test_frame),
        "false_positives": int(((errors["label"] == 0) & (errors["prediction"] == 1)).sum()),
        "false_negatives": int(((errors["label"] == 1) & (errors["prediction"] == 0)).sum()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved error analysis to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

