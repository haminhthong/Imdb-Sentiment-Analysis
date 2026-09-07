"""Điểm chạy chính (CLI) để huấn luyện development BiLSTM của CineSentiment.

Quy trình chuẩn hóa (Leakage-Safe Validation Protocol):
1. Đọc dữ liệu, kiểm tra schema, deduplication theo raw & normalized hash.
2. Huấn luyện mô hình và Early Stopping dựa trên Validation Loss.
3. Early stopping chỉ nhìn Validation Loss.
4. Lưu validation metrics và best development checkpoint.
5. Final fit, calibration và Official Test được tách thành release steps riêng.

Ví dụ sử dụng:
    python train.py --epochs 15 --batch-size 128
"""

import argparse
from pathlib import Path

import torch

from sentiment.artifacts import save_checkpoint, save_json, save_plots
from sentiment.config import ExperimentConfig
from sentiment.data import create_data_bundle
from sentiment.engine import evaluate_model, train_model
from sentiment.model import SentimentRNN
from sentiment.utils import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    """Cấu hình các tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình phân loại cảm xúc CineSentiment AI dưới giao thức Validation an toàn"
    )
    parser.add_argument(
        "--model",
        choices=["bilstm"],
        default="bilstm",
        help="Mô hình production duy nhất của lifecycle v3: BiLSTM.",
    )
    parser.add_argument(
        "--train-data",
        default="data/raw/train.csv",
        help="Đường dẫn Official Train. Mặc định là 'data/raw/train.csv'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Thư mục xuất checkpoint development. Mặc định là 'runs/dev_<model>'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Số lượng epoch huấn luyện tối đa. Mặc định là 15.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Kích thước lô (batch size). Mặc định là 128.",
    )
    parser.add_argument(
        "--truncation-strategy",
        choices=["first", "head_tail"],
        default="head_tail",
        help="Chiến lược cắt chuỗi ('first' hoặc 'head_tail'). Mặc định là 'first'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên (Random seed). Mặc định là 42.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Thiết bị tính toán ('auto', 'cpu', 'cuda'). Mặc định là 'auto'.",
    )
    return parser.parse_args()


def main() -> None:
    """Luồng thực thi huấn luyện mô hình từ dòng lệnh."""
    args = parse_args()
    config = ExperimentConfig(
        model_type=args.model,
        validation_size=0.1,
        calibration_size=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        truncation_strategy=args.truncation_strategy,
        seed=args.seed,
    )

    seed_everything(config.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir or f"runs/dev_{args.model}")

    print("=" * 65)
    print(f"*** HUAN LUYEN CINESENTIMENT PLATFORM - [{config.model_type.upper()}] ***")
    print("=" * 65)
    print(f"-> Thiet bi tinh toan   : {device}")
    print(f"-> Thu muc dau ra       : {output_dir}")
    print(f"-> Batch size / Epochs  : {config.batch_size} / {config.epochs}")
    print(f"-> Truncation Strategy  : {config.truncation_strategy}")
    print(f"-> Seed                 : {config.seed}")
    print("-" * 65)

    # 1. Nạp và kiểm toán chất lượng dữ liệu
    print("1/4. Dang nap Official Train va tao ba development split...")
    data = create_data_bundle(args.train_data, config)
    print(f"     -> Phan bo mau     : {data.sizes}")
    print(f"     -> Kich thuoc tu dien: {len(data.vocabulary):,} tokens")
    print(
        f"     -> OOV Rates       : Train={data.audit['oov_rates']['train']:.2%}, "
        f"Val={data.audit['oov_rates']['validation']:.2%}, "
        f"Calibration={data.audit['oov_rates']['calibration']:.2%}"
    )

    # 2. Khởi tạo mô hình
    print("2/4. Dang khoi tao mo hinh...")
    model = SentimentRNN(
        vocabulary_size=len(data.vocabulary),
        padding_index=data.vocabulary.pad_index,
        config=config,
    ).to(device)

    total_params = model.count_parameters()
    print(f"     -> Tong tham so mo hinh (Parameters): {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = torch.nn.BCEWithLogitsLoss()

    # 3. Huấn luyện mô hình
    print("3/4. Tien hanh huan luyen voi Early Stopping...")
    history = train_model(
        model=model,
        train_loader=data.train,
        validation_loader=data.validation,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        epochs=config.epochs,
        patience=config.patience,
    )

    # 4. Chỉ đánh giá Validation để chọn model/epoch; calibration để release step.
    print("4/4. Danh gia Validation va luu development checkpoint...")
    val_metrics = evaluate_model(
        model,
        data.validation,
        loss_function,
        device,
        decision_threshold=config.decision_threshold,
    )

    # Đây là checkpoint huấn luyện, không phải artifact deploy.
    save_checkpoint(
        output_dir / "best_dev.ckpt",
        model,
        data.vocabulary,
        config,
        training_data_hash=data.audit.get("train_data_hash"),
        checkpoint_kind="best_dev",
        model_version="1.0.0-dev",
        source_dataset_hash=data.audit.get("source_train_hash"),
        train_split_hash=data.audit.get("train_data_hash"),
        validation_split_hash=data.audit.get("validation_data_hash"),
        calibration_split_hash=data.audit.get("calibration_data_hash"),
        best_dev_epoch=history.get("best_epoch"),
    )
    save_json(output_dir / "validation_metrics.json", val_metrics)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "data_audit.json", data.audit)
    save_plots(output_dir, history, val_metrics["confusion_matrix"])

    print("=" * 65)
    print("[SUCCESS] HOAN THANH HUAN LUYEN & VALIDATION!")
    print(f"-> Val Accuracy      : {val_metrics['accuracy']:.2%}")
    print(f"-> Val Macro F1      : {val_metrics['macro_f1']:.2%}")
    print(f"-> Val ROC-AUC       : {val_metrics['roc_auc']:.4f}")
    print(f"-> Val Brier Score   : {val_metrics['brier_score']:.4f}")
    print(f"-> Val ECE           : {val_metrics['ece']:.4f}")
    print(f"-> Best epoch         : {history['best_epoch']}")
    print(f"-> Checkpoint Schema : v3 development ({output_dir / 'best_dev.ckpt'})")
    print(f"-> Validation Report : {output_dir / 'validation_metrics.json'}")

    print("=" * 65)


if __name__ == "__main__":
    main()
