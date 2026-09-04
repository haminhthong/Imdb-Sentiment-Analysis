"""Huấn luyện baseline TF-IDF + Logistic Regression để đối chiếu mô hình RNN."""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sentiment.data_validation import load_dataset, remove_train_test_overlap
from sentiment.text import tokenize


def create_pipeline(max_features: int = 50_000) -> Pipeline:
    """Tạo baseline đơn giản và dễ diễn giải cho phân loại văn bản."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, texts, labels) -> dict:
    """Đánh giá baseline bằng cùng nhóm metric với mô hình PyTorch."""
    predictions = pipeline.predict(texts)
    probabilities = pipeline.predict_proba(texts)
    return {
        "loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, predictions)),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=["Negative", "Positive"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện baseline TF-IDF + Logistic Regression"
    )
    parser.add_argument("--train-data", default="train.csv")
    parser.add_argument("--test-data", default="test.csv")
    parser.add_argument("--output-dir", default="artifacts/baseline")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_size < 1:
        raise ValueError("validation-size phải nằm trong khoảng (0, 1).")
    if args.max_features <= 0:
        raise ValueError("max-features phải lớn hơn 0.")

    train_source = load_dataset(args.train_data)
    test_frame = remove_train_test_overlap(train_source, load_dataset(args.test_data))
    train_frame, validation_frame = train_test_split(
        train_source,
        test_size=args.validation_size,
        random_state=42,
        stratify=train_source["label"],
    )

    pipeline = create_pipeline(args.max_features)
    pipeline.fit(train_frame["text"], train_frame["label"].astype(int))
    validation_metrics = evaluate(
        pipeline, validation_frame["text"], validation_frame["label"].astype(int)
    )
    test_metrics = evaluate(pipeline, test_frame["text"], test_frame["label"].astype(int))
    test_metrics["validation_accuracy"] = validation_metrics["accuracy"]
    test_metrics["validation_macro_f1"] = validation_metrics["classification_report"][
        "macro avg"
    ]["f1-score"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Validation accuracy: {validation_metrics['accuracy']:.2%}")
    print(f"Test accuracy: {test_metrics['accuracy']:.2%}")
    # Dùng ASCII để tương thích terminal Windows chưa bật UTF-8.
    print(f"Saved baseline to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
