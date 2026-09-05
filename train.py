"""Điểm chạy chính (CLI) để huấn luyện mô hình CineSentiment AI (LSTM, GRU, BiLSTM).

Quy trình chuẩn hóa (Leakage-Safe Validation Protocol):
1. Đọc dữ liệu, kiểm tra schema, deduplication theo raw & normalized hash.
2. Huấn luyện mô hình và Early Stopping dựa trên Validation Loss.
3. Học tham số Temperature Scaling DUY NHẤT từ Validation Logits.
4. Đánh giá toàn diện và lưu `validation_metrics.json`.
5. KHÔNG đánh giá tập Official Test trong giai đoạn lựa chọn mô hình ứng viên (chống Test Peeking).

Ví dụ sử dụng:
    python train.py --model bilstm --epochs 15 --batch-size 128
    python train.py --model gru --epochs 5 --device cpu --output-dir artifacts/gru
"""

import argparse
from pathlib import Path

import torch

from sentiment.artifacts import (
    save_checkpoint,
    save_json,
    save_plots,
    save_reliability_diagram,
)
from sentiment.calibration import TemperatureScaler
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
        choices=["lstm", "gru", "bilstm"],
        default="bilstm",
        help="Kiến trúc mô hình RNN ('lstm', 'gru', 'bilstm'). Mặc định là 'bilstm' (default implementation).",
    )
    parser.add_argument(
        "--train-data",
        default="train.csv",
        help="Đường dẫn tệp CSV dữ liệu huấn luyện. Mặc định là 'train.csv'.",
    )
    parser.add_argument(
        "--test-data",
        default="test.csv",
        help="Đường dẫn tệp CSV dữ liệu kiểm thử. Mặc định là 'test.csv'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Thư mục xuất lưu checkpoint và kết quả. Mặc định là 'artifacts/<model>'.",
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
        default="first",
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
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Cờ tùy chọn đánh giá ngay trên tập Test (Lưu ý: Chỉ khuyến nghị cho Champion đã chọn).",
    )
    return parser.parse_args()


def main() -> None:
    """Luồng thực thi huấn luyện mô hình từ dòng lệnh."""
    args = parse_args()
    config = ExperimentConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        truncation_strategy=args.truncation_strategy,
        seed=args.seed,
    )

    seed_everything(config.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir or f"artifacts/{args.model}")

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
    print("1/4. Dang nap du lieu, kiem tra anti-leakage va tao DataBundle...")
    data = create_data_bundle(args.train_data, args.test_data, config)
    print(f"     -> Phan bo mau     : {data.sizes}")
    print(f"     -> Kich thuoc tu dien: {len(data.vocabulary):,} tokens")
    print(f"     -> OOV Rates       : Train={data.audit['oov_rates']['train']:.2%}, "
          f"Val={data.audit['oov_rates']['validation']:.2%}")

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

    # 4. Hiệu chuẩn xác suất (Temperature Scaling) & Đánh giá Validation
    print("4/4. Hieu chuan Temperature Scaling & Danh gia Validation Leaderboard...")
    val_raw = evaluate_model(
        model, data.validation, loss_function, device, return_raw=True
    )

    # Học nhiệt độ T tối ưu trên Validation Logits
    scaler = TemperatureScaler()
    best_temperature = scaler.fit(val_raw["raw_logits"], val_raw["raw_labels"])
    config.temperature = best_temperature

    # Đánh giá lại Validation metrics với xác suất đã hiệu chuẩn
    val_metrics = evaluate_model(
        model,
        data.validation,
        loss_function,
        device,
        temperature=best_temperature,
        decision_threshold=config.decision_threshold,
    )

    # Lưu checkpoint Schema v2 và các báo cáo Validation
    save_checkpoint(
        output_dir / "model.pt",
        model,
        data.vocabulary,
        config,
        training_data_hash=data.audit.get("train_data_hash"),
    )
    save_json(output_dir / "validation_metrics.json", val_metrics)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "data_audit.json", data.audit)
    save_plots(output_dir, history, val_metrics["confusion_matrix"])

    # Vẽ biểu đồ hiệu chuẩn
    calibrated_val_probs = scaler.calibrate(val_raw["raw_logits"])
    save_reliability_diagram(
        output_dir,
        val_raw["raw_labels"],
        calibrated_val_probs,
        filename="validation_reliability.png",
    )

    print("=" * 65)
    print("[SUCCESS] HOAN THANH HUAN LUYEN & VALIDATION!")
    print(f"-> Val Accuracy      : {val_metrics['accuracy']:.2%}")
    print(f"-> Val Macro F1      : {val_metrics['macro_f1']:.2%}")
    print(f"-> Val ROC-AUC       : {val_metrics['roc_auc']:.4f}")
    print(f"-> Val Brier Score   : {val_metrics['brier_score']:.4f}")
    print(f"-> Val ECE           : {val_metrics['ece']:.4f}")
    print(f"-> Temperature (T)   : {best_temperature:.4f}")
    print(f"-> Checkpoint Schema : v2 ({output_dir / 'model.pt'})")
    print(f"-> Validation Report : {output_dir / 'validation_metrics.json'}")

    # Xử lý cờ ngoại lệ nếu người dùng muốn mở test sớm
    if args.evaluate_test:
        print("\n[WARNING] Dang mo tap Test (Khuyen nghi chi dung cho Champion thong qua evaluate_final.py)!")
        test_metrics = evaluate_model(
            model,
            data.test,
            loss_function,
            device,
            temperature=best_temperature,
            decision_threshold=config.decision_threshold,
        )
        save_json(output_dir / "test_metrics.json", test_metrics)
        print(f"-> Test Accuracy     : {test_metrics['accuracy']:.2%}")
        print(f"-> Test Macro F1     : {test_metrics['macro_f1']:.2%}")

    print("=" * 65)


if __name__ == "__main__":
    main()
