"""Fit lại BiLSTM trên Train + Validation sau khi epoch đã được khóa."""

import argparse
from pathlib import Path
from typing import Any

import torch

from sentiment.artifacts import save_checkpoint, save_json
from sentiment.config import ExperimentConfig
from sentiment.data import create_final_fit_bundle
from sentiment.data_validation import compute_dataset_hash
from sentiment.engine import train_fixed_epochs
from sentiment.model import SentimentRNN
from sentiment.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    """Đọc tham số final fit; script không nhận Official Test."""
    parser = argparse.ArgumentParser(
        description="Final fit CineSentiment trên Train + Validation đã được khóa"
    )
    parser.add_argument("--dev-checkpoint", default="runs/dev_bilstm/best_dev.ckpt")
    parser.add_argument("--train-data", default="data/raw/train.csv")
    parser.add_argument("--output-dir", default="runs/final_bilstm")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    """Phục hồi config/epoch development và huấn luyện lại không early stopping."""
    args = parse_args()
    dev_path = Path(args.dev_checkpoint)
    if not dev_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy development checkpoint: {dev_path}")

    dev_checkpoint: dict[str, Any] = torch.load(dev_path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**dev_checkpoint["config"])
    best_epoch = int(
        dev_checkpoint.get("best_dev_epoch") or dev_checkpoint.get("final_fit_epoch") or 0
    )
    if best_epoch <= 0:
        raise ValueError("Development checkpoint thiếu best_epoch hợp lệ để final fit.")

    seed_everything(config.seed)
    device = select_device(args.device)
    bundle, train_frame, validation_frame, calibration_frame = create_final_fit_bundle(
        args.train_data, config
    )
    model = SentimentRNN(len(bundle.vocabulary), bundle.vocabulary.pad_index, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = train_fixed_epochs(
        model,
        bundle.train,
        optimizer,
        torch.nn.BCEWithLogitsLoss(),
        device,
        best_epoch,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / "final_raw_model.pt",
        model,
        bundle.vocabulary,
        config,
        checkpoint_kind="final_raw",
        model_version="1.0.0-raw",
        source_dataset_hash=bundle.audit["source_train_hash"],
        train_split_hash=compute_dataset_hash(train_frame),
        validation_split_hash=compute_dataset_hash(validation_frame),
        calibration_split_hash=compute_dataset_hash(calibration_frame),
        final_fit_epoch=best_epoch,
    )
    save_json(output_dir / "final_fit_history.json", history)
    save_json(output_dir / "data_audit.json", bundle.audit)
    print(f"Đã lưu final raw model tại: {(output_dir / 'final_raw_model.pt').resolve()}")


if __name__ == "__main__":
    main()
