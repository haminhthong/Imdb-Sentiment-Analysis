"""Huấn luyện baseline TF-IDF + Logistic Regression đối chiếu mô hình Deep Learning.

TF-IDF + Logistic Regression là một first-class model quan trọng cho bài toán sentiment:
- Đo lường xem mô hình RNN có thực sự đem lại cải tiến vượt bậc so với sparse-text hay không.
- Cung cấp tính giải thích (Explainability) thông qua trọng số của các n-gram đặc trưng.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from sentiment.calibration import compute_brier_score, compute_ece
from sentiment.config import ExperimentConfig
from sentiment.data import split_development_frame
from sentiment.data_validation import compute_dataset_hash, load_dataset
from sentiment.text import tokenize


def create_pipeline(max_features: int = 50_000) -> Pipeline:
    """Tạo pipeline TF-IDF unigram + bigram kết hợp Logistic Regression."""
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


def evaluate_baseline(pipeline: Pipeline, texts, labels) -> dict:
    """Đánh giá toàn diện mô hình baseline bằng cùng bộ metric chuẩn."""
    y_true = np.asarray(labels, dtype=int)
    predictions = pipeline.predict(texts)
    probabilities = pipeline.predict_proba(texts)[:, 1]

    unique_labels = set(y_true)
    if len(unique_labels) > 1:
        roc_auc = float(roc_auc_score(y_true, probabilities))
        pr_auc = float(average_precision_score(y_true, probabilities))
    else:
        roc_auc = 0.5
        pr_auc = 0.5

    report = classification_report(
        y_true,
        predictions,
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        # Alias để bảng/consumer cũ không vỡ; report mới dùng log_loss.
        "loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": compute_brier_score(y_true, probabilities),
        "ece": compute_ece(y_true, probabilities, n_bins=10),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


# Alias tương thích ngược cho unit test
evaluate = evaluate_baseline


def extract_top_features(
    pipeline: Pipeline, top_k: int = 15
) -> dict[str, list[dict[str, float | str]]]:
    """Trích xuất các n-gram có trọng số dương và âm cao nhất (Feature Importance).

    Lưu ý kỹ thuật: Trọng số hồi quy biểu thị mức độ tương quan đặc trưng trong tập huấn luyện,
    không đồng nghĩa với mối quan hệ nhân quả (Correlation != Causation).
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]

    top_pos_idx = np.argsort(coefs)[-top_k:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_k]

    return {
        "top_positive_ngrams": [
            {"ngram": str(feature_names[i]), "weight": float(coefs[i])} for i in top_pos_idx
        ],
        "top_negative_ngrams": [
            {"ngram": str(feature_names[i]), "weight": float(coefs[i])} for i in top_neg_idx
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện baseline TF-IDF + Logistic Regression cho CineSentiment Platform"
    )
    parser.add_argument("--train-data", default="data/raw/train.csv")
    parser.add_argument("--output-dir", default="artifacts/baseline")
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--calibration-size", type=float, default=0.1)
    parser.add_argument("--max-features", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_size < 1:
        raise ValueError("validation-size phải nằm trong khoảng (0, 1).")
    if args.max_features <= 0:
        raise ValueError("max-features phải lớn hơn 0.")

    config = ExperimentConfig(
        validation_size=args.validation_size,
        calibration_size=args.calibration_size,
        min_frequency=2,
        max_vocabulary_size=args.max_features,
    )
    train_source = load_dataset(args.train_data)
    train_frame, validation_frame, calibration_frame = split_development_frame(train_source, config)

    print("=" * 60)
    print("*** HUAN LUYEN BASELINE TF-IDF + LOGISTIC REGRESSION ***")
    print("=" * 60)
    pipeline = create_pipeline(args.max_features)
    pipeline.fit(train_frame["text"], train_frame["label"].astype(int))

    # Đánh giá trên Validation split
    validation_metrics = evaluate_baseline(
        pipeline, validation_frame["text"], validation_frame["label"].astype(int)
    )
    explainability = extract_top_features(pipeline, top_k=15)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")

    # Lưu validation metrics và báo cáo giải thích tính năng
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "explainability.json").write_text(
        json.dumps(explainability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "data_audit.json").write_text(
        json.dumps(
            {
                "source_train_hash": compute_dataset_hash(train_source),
                "train_split_hash": compute_dataset_hash(train_frame),
                "validation_split_hash": compute_dataset_hash(validation_frame),
                "calibration_split_hash": compute_dataset_hash(calibration_frame),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"-> Validation Accuracy : {validation_metrics['accuracy']:.2%}")
    print(f"-> Validation Macro-F1 : {validation_metrics['macro_f1']:.2%}")
    print(f"-> Validation ROC-AUC  : {validation_metrics['roc_auc']:.4f}")
    print(f"-> Validation ECE      : {validation_metrics['ece']:.4f}")
    print(f"-> Da luu baseline tai : {output_dir.resolve()}")

    print("=" * 60)


if __name__ == "__main__":
    main()
