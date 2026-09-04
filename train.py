"""Điểm chạy chính (CLI) để huấn luyện mô hình CineSentiment AI (LSTM, GRU, BiLSTM).

Ví dụ sử dụng từ dòng lệnh:
    python train.py --model bilstm --epochs 15 --batch-size 128
    python train.py --model gru --epochs 5 --device cpu --output-dir artifacts/gru
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
        description="Huấn luyện mô hình phân loại cảm xúc đánh giá phim CineSentiment AI"
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "gru", "bilstm"],
        default="bilstm",
        help="Kiến trúc mô hình RNN ('lstm', 'gru', 'bilstm'). Mặc định là 'bilstm'.",
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
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Đảm bảo khả năng tái lập kết quả
    seed_everything(config.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir or f"artifacts/{args.model}")

    print("=" * 60)
    print(f"*** BAT DAU HUAN LUYEN CINESENTIMENT AI - [{config.model_type.upper()}] ***")
    print("=" * 60)
    print(f"-> Thiet bi tinh toan : {device}")
    print(f"-> Thu muc dau ra     : {output_dir}")
    print(f"-> Batch size / Epochs: {config.batch_size} / {config.epochs}")
    print("-" * 60)

    # 1. Nạp và tiền xử lý dữ liệu
    print("1/4. Dang nap du lieu va xay dung bo tu vung (Vocabulary)...")
    data = create_data_bundle(args.train_data, args.test_data, config)
    print(f"     -> Phan bo mau : {data.sizes}")
    print(f"     -> Kich thuoc tu dien: {len(data.vocabulary):,} token")

    # 2. Khởi tạo mô hình & Loss / Optimizer
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
    print("3/4. Tien hanh huan luyen mo hinh voi Early Stopping...")
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

    # 4. Đánh giá mô hình trên tập Test độc lập
    print("4/4. Danh gia ket qua tren tap kiem thu (Test Set)...")
    metrics = evaluate_model(model, data.test, loss_function, device)

    # 5. Lưu checkpoint và biểu đồ kết quả
    save_checkpoint(output_dir / "model.pt", model, data.vocabulary, config)
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "history.json", history)
    save_plots(output_dir, history, metrics["confusion_matrix"])

    print("=" * 60)
    print("[SUCCESS] HOAN THANH HUAN LUYEN!")
    print(f"-> Test Loss    : {metrics['loss']:.4f}")
    print(f"-> Test Accuracy: {metrics['accuracy']:.2%}")
    print(f"-> Macro F1     : {metrics['classification_report']['macro avg']['f1-score']:.2%}")
    print(f"-> Da luu ket qua tai: {output_dir.resolve()}")
    print("=" * 60)



if __name__ == "__main__":
    main()

