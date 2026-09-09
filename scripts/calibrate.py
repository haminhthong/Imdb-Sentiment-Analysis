"""Calibrate final raw model trên Calibration Set và tạo release bundle."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from sentiment.artifacts import save_checkpoint, save_json, save_reliability_diagram
from sentiment.calibration import TemperatureScaler, tune_confidence_threshold
from sentiment.config import ExperimentConfig
from sentiment.data import create_final_fit_bundle
from sentiment.data_validation import compute_dataset_hash
from sentiment.engine import evaluate_model
from sentiment.model import SentimentRNN
from sentiment.utils import select_device


def parse_args() -> argparse.Namespace:
    """Đọc tham số calibration; không nhận hoặc đọc Official Test."""
    parser = argparse.ArgumentParser(description="Calibration CineSentiment trên Calibration Set")
    parser.add_argument("--raw-model", default="runs/final_bilstm/final_raw_model.pt")
    parser.add_argument("--train-data", default="data/raw/train.csv")
    parser.add_argument("--output-dir", default="artifacts/releases/v1.0.0")
    parser.add_argument(
        "--manifest",
        default="data/manifest.json",
        help="Manifest đã tạo từ bước download; script chỉ đọc hash, không đọc Official Test.",
    )
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    """Fit temperature/policy rồi đóng gói model deployable."""
    args = parse_args()
    raw_path = Path(args.raw_model)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy final raw model: {raw_path}")

    raw: dict[str, Any] = torch.load(raw_path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**raw["config"])
    device = select_device(args.device)
    bundle, train_frame, validation_frame, calibration_frame = create_final_fit_bundle(
        args.train_data, config
    )
    if raw.get("vocabulary_hash") != bundle.vocabulary.compute_hash():
        raise ValueError("Vocabulary của raw model không khớp final-fit data; dừng calibration.")

    model = SentimentRNN(len(bundle.vocabulary), bundle.vocabulary.pad_index, config).to(device)
    model.load_state_dict(raw["model_state"])
    model.eval()
    raw_metrics = evaluate_model(
        model,
        bundle.calibration,
        torch.nn.BCEWithLogitsLoss(),
        device,
        return_raw=True,
    )
    scaler = TemperatureScaler()
    temperature = scaler.fit(raw_metrics["raw_logits"], raw_metrics["raw_labels"])
    calibrated_probs = scaler.calibrate(raw_metrics["raw_logits"])
    confidence_threshold, policy_curve = tune_confidence_threshold(
        raw_metrics["raw_labels"], calibrated_probs, min_coverage=args.min_coverage
    )
    config.temperature = temperature
    config.confidence_threshold = confidence_threshold

    official_test_hash = "unspecified"
    manifest_path = Path(args.manifest)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        official_test_hash = str(manifest.get("official_test_hash", "unspecified"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_metrics = evaluate_model(
        model,
        bundle.calibration,
        torch.nn.BCEWithLogitsLoss(),
        device,
        temperature=temperature,
        decision_threshold=config.decision_threshold,
    )
    calibration_metrics.pop("raw_logits", None)
    calibration_metrics.pop("raw_labels", None)
    save_checkpoint(
        output_dir / "model.pt",
        model,
        bundle.vocabulary,
        config,
        checkpoint_kind="release",
        model_version="1.0.0",
        source_dataset_hash=bundle.audit["source_train_hash"],
        train_split_hash=compute_dataset_hash(train_frame),
        validation_split_hash=compute_dataset_hash(validation_frame),
        calibration_split_hash=compute_dataset_hash(calibration_frame),
        official_test_hash=official_test_hash,
        final_fit_epoch=raw.get("final_fit_epoch"),
        final_metrics=calibration_metrics,
    )
    save_json(
        output_dir / "calibration.json",
        {
            "temperature": temperature,
            "confidence_threshold": confidence_threshold,
            "min_coverage": args.min_coverage,
            "policy_curve": policy_curve,
            "calibration_split_hash": compute_dataset_hash(calibration_frame),
            "metrics": calibration_metrics,
        },
    )
    release_manifest = dict(bundle.audit)
    release_manifest["official_test_hash"] = official_test_hash
    if manifest_path.is_file():
        release_manifest["source_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    save_json(output_dir / "data_manifest.json", release_manifest)
    model_card = Path("MODEL_CARD.md")
    if model_card.is_file():
        shutil.copyfile(model_card, output_dir / "model_card.md")
    save_reliability_diagram(
        output_dir,
        raw_metrics["raw_labels"],
        calibrated_probs,
        filename="calibration_reliability.png",
    )
    print(f"Đã tạo release bundle tại: {output_dir.resolve()}")
    print(f"Temperature={temperature:.4f}, confidence_threshold={confidence_threshold:.2f}")


if __name__ == "__main__":
    main()
