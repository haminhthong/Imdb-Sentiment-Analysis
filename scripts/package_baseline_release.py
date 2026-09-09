"""Đóng gói TF-IDF baseline khi validation gate chọn baseline làm champion."""

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Đọc thư mục artifact baseline và thư mục release đích."""
    parser = argparse.ArgumentParser(description="Package baseline champion release")
    parser.add_argument("--baseline-dir", default="artifacts/baseline")
    parser.add_argument("--output-dir", default="artifacts/releases/v1.0.0")
    return parser.parse_args()


def main() -> None:
    """Copy model baseline và metadata validation vào release bundle."""
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    model_path = baseline_dir / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy baseline artifact: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, output_dir / "model.joblib")
    metadata = {
        "artifact_schema_version": 3,
        "checkpoint_kind": "release",
        "model_version": "1.0.0-baseline",
        "model_type": "tfidf_logistic_regression",
        "decision_threshold": 0.5,
        "confidence_threshold": 0.5,
        "validation_metrics": {},
    }
    for name in ("validation_metrics.json", "data_audit.json", "explainability.json"):
        source = baseline_dir / name
        if source.is_file():
            metadata[name.removesuffix(".json")] = json.loads(source.read_text(encoding="utf-8"))
    manifest_path = Path("data/manifest.json")
    if manifest_path.is_file():
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata["source_manifest"] = source_manifest
        metadata["official_test_hash"] = str(
            source_manifest.get("official_test_hash", "unspecified")
        )
    (output_dir / "baseline_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = Path("MODEL_CARD.md")
    if model_card.is_file():
        shutil.copyfile(model_card, output_dir / "model_card.md")
    # Dùng thông báo ASCII để CLI không lỗi khi chạy trong Windows console cp1252.
    print(f"Baseline release packaged at: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
