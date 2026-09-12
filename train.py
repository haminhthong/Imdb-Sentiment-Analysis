"""Huấn luyện mô hình BiLSTM cho phân loại cảm xúc IMDB.

Quy trình chuẩn mực:
1. Đọc dữ liệu, kiểm tra schema, deduplicate chuẩn hóa.
2. Tách Train (80%), Validation (10%), Calibration (10%).
3. Xây dựng bộ từ điển CHỈ trên tập Train (Train-only vocabulary).
4. Huấn luyện BiLSTM với pack_padded_sequence và Early Stopping theo Validation Loss.
5. Học hệ số Temperature Scaling trên Calibration split để hiệu chuẩn xác suất.
6. Lưu checkpoint tinh gọn (model.pt), metrics và biểu đồ trực quan.
"""

import argparse
from pathlib import Path

import torch

from sentiment.artifacts import save_checkpoint, save_json, save_plots, save_reliability_diagram
from sentiment.calibration import TemperatureScaler
from sentiment.config import ExperimentConfig
from sentiment.data import create_data_bundle
from sentiment.engine import evaluate_model, train_model
from sentiment.model import BiLSTMSentimentClassifier
from sentiment.utils import configure_utf8_output, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình phân loại cảm xúc BiLSTM cho CineSentiment"
    )
    parser.add_argument(
        "--train-data",
        default="data/raw/train.csv",
        help="Đường dẫn tệp dữ liệu Train (mặc định: data/raw/train.csv hoặc train.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Thư mục xuất checkpoint và metrics (mặc định: artifacts)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Số lượng epoch tối đa (mặc định: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Kích thước batch (mặc định: 128)",
    )
    parser.add_argument(
        "--truncation-strategy",
        choices=["first", "head_tail"],
        default="head_tail",
        help="Chiến lược cắt ngắn chuỗi ('first' hoặc 'head_tail', mặc định: head_tail)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Độ dài chuỗi tối đa (mặc định: 256)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn số lượng mẫu để huấn luyện nhanh (mặc định: None)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=3,
        help="Tần suất tối thiểu của từ trong từ điển (mặc định: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (mặc định: 42)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Thiết bị tính toán (mặc định: auto)",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_output()
    args = parse_args()
    config = ExperimentConfig(
        model_type="bilstm",
        validation_size=0.1,
        calibration_size=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        min_frequency=args.min_frequency,
        truncation_strategy=args.truncation_strategy,
        seed=args.seed,
    )

    seed_everything(config.seed)
    device = select_device(args.device)

    train_path = Path(args.train_data)
    if not train_path.is_file() and Path("train.csv").is_file():
        train_path = Path("train.csv")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("*** HUẤN LUYỆN MÔ HÌNH BiLSTM (CINESENTIMENT) ***")
    print("=" * 65)
    print(f"-> Thiết bị tính toán  : {device}")
    print(f"-> Thư mục đầu ra      : {output_dir.resolve()}")
    print(f"-> Batch size / Epochs : {config.batch_size} / {config.epochs}")
    print(f"-> Max length          : {config.max_length}")
    print(f"-> Truncation Strategy : {config.truncation_strategy}")
    print(f"-> Seed                : {config.seed}")
    print("-" * 65)

    # 1. Nạp dữ liệu và xây dựng Vocabulary
    print("1/4. Đang nạp dữ liệu và phân chia Train/Val/Calibration...")
    data = create_data_bundle(train_path, config, max_samples=args.max_samples)
    print(f"     -> Phân bổ mẫu    : {data.sizes}")
    print(f"     -> Từ điển Train  : {len(data.vocabulary):,} tokens")
    print(
        f"     -> Tỷ lệ OOV      : Train={data.audit['oov_rates']['train']:.2%}, "
        f"Val={data.audit['oov_rates']['validation']:.2%}, "
        f"Cal={data.audit['oov_rates']['calibration']:.2%}"
    )

    # 2. Khởi tạo mô hình
    print("2/4. Đang khởi tạo mô hình BiLSTM...")
    model = BiLSTMSentimentClassifier(
        vocabulary_size=len(data.vocabulary),
        padding_index=data.vocabulary.pad_index,
        config=config,
    ).to(device)
    print(f"     -> Tổng tham số   : {model.count_parameters():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = torch.nn.BCEWithLogitsLoss()

    # 3. Huấn luyện với Early Stopping
    print("3/4. Đang huấn luyện mô hình (Early Stopping trên Validation Loss)...")
    history, best_epoch = train_model(
        model=model,
        train_loader=data.train,
        validation_loader=data.validation,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        epochs=config.epochs,
        patience=config.patience,
    )
    print(f"     -> Best Epoch đạt được: {best_epoch}")

    # 4. Hiệu chuẩn Temperature Scaling trên Calibration Set
    print("4/4. Đang hiệu chuẩn xác suất (Temperature Scaling trên Calibration Set)...")
    cal_raw = evaluate_model(model, data.calibration, loss_function, device, return_raw=True)
    scaler = TemperatureScaler()
    temperature = scaler.fit(cal_raw["raw_logits"], cal_raw["raw_labels"])
    config.temperature = temperature
    print(f"     -> Hệ số Temperature học được: T = {temperature:.4f}")

    # Đánh giá cuối trên Validation với Temperature đã hiệu chuẩn
    val_metrics = evaluate_model(
        model,
        data.validation,
        loss_function,
        device,
        temperature=temperature,
    )
    print(f"     -> Validation Accuracy : {val_metrics['accuracy']:.2%}")
    print(f"     -> Validation Macro-F1 : {val_metrics['macro_f1']:.2%}")
    print(f"     -> Validation Brier    : {val_metrics['brier_score']:.4f}")
    print(f"     -> Validation ECE      : {val_metrics['ece']:.4f}")

    # Lưu artifacts
    save_checkpoint(
        output_dir / "model.pt",
        model,
        data.vocabulary,
        config,
        temperature=temperature,
    )
    save_json(output_dir / "validation_metrics.json", val_metrics)
    save_plots(output_dir, history, val_metrics["confusion_matrix"])

    calibrated_val_eval = evaluate_model(
        model, data.validation, loss_function, device, temperature=temperature, return_raw=True
    )
    save_reliability_diagram(
        output_dir,
        calibrated_val_eval["raw_labels"],
        calibrated_val_eval["probabilities"],
        filename="reliability_diagram.png",
    )

    print("-" * 65)
    print(f"✅ Hoàn tất! Model checkpoint đã lưu tại: {output_dir / 'model.pt'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
