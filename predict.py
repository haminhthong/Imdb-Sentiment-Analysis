"""Script CLI dự đoán cảm xúc (Positive / Negative) cho câu văn bản hoặc tệp tin.

Ví dụ sử dụng:
    python predict.py "This movie has an amazing storyline and breathtaking visuals!"
    python predict.py "Terrible plot and awful acting." --checkpoint artifacts/bilstm/model.pt
"""

import argparse
from pathlib import Path

from sentiment.inference import load_predictor
from sentiment.utils import select_device


def parse_args() -> argparse.Namespace:
    """Cấu hình tham số dòng lệnh cho dự đoán."""
    parser = argparse.ArgumentParser(
        description="Dự đoán cảm xúc của một câu đánh giá phim bằng CineSentiment AI"
    )
    parser.add_argument(
        "text",
        help="Nội dung đánh giá phim (hoặc đường dẫn đến tệp txt chứa nội dung cần phân tích).",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/releases/v1.0.0/model.pt",
        help="Đường dẫn release model.pt. Mặc định là 'artifacts/releases/v1.0.0/model.pt'.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Thiết bị suy luận ('auto', 'cpu', 'cuda'). Mặc định là 'auto'.",
    )
    return parser.parse_args()


def main() -> None:
    """Luồng suy luận chính."""
    args = parse_args()

    # Kiểm tra xem input có phải là file text không
    input_path = Path(args.text)
    if input_path.is_file():
        text = input_path.read_text(encoding="utf-8").strip()
    else:
        text = args.text

    device = select_device(args.device)
    predictor = load_predictor(args.checkpoint, str(device))
    result = predictor.predict(text)

    print("=" * 60)
    print("🎬 KẾT QUẢ PHÂN TÍCH CẢM XÚC - CINESENTIMENT AI")
    print("=" * 60)
    print(f'• Nội dung nhập vào : "{text[:80]}{"..." if len(text) > 80 else ""}"')
    print(f"• Cảm xúc dự đoán   : [{result.label.upper()}]")
    print(f"• Xác suất Positive : {result.positive_probability:.2%}")
    print(f"• Độ tin cậy (Conf) : {result.confidence:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
