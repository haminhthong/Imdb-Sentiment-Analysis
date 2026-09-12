"""CLI dự đoán cảm xúc (Positive / Negative) cho câu văn bản hoặc tệp tin.

Ví dụ:
    python predict.py "This movie has an amazing storyline and breathtaking visuals!"
    python predict.py "Terrible plot and awful acting." --checkpoint artifacts/model.pt
"""

import argparse
from pathlib import Path

from sentiment.inference import load_predictor
from sentiment.utils import configure_utf8_output, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dự đoán cảm xúc của một câu đánh giá phim bằng CineSentiment"
    )
    parser.add_argument(
        "text",
        help="Nội dung đánh giá phim (hoặc đường dẫn đến tệp txt chứa nội dung cần phân tích).",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/model.pt",
        help="Đường dẫn model.pt hoặc model.joblib (mặc định: artifacts/model.pt).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Thiết bị suy luận ('auto', 'cpu', 'cuda', mặc định: auto).",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_output()
    args = parse_args()

    input_path = Path(args.text)
    text = input_path.read_text(encoding="utf-8").strip() if input_path.is_file() else args.text

    device = select_device(args.device)
    predictor = load_predictor(args.checkpoint, str(device))
    result = predictor.predict(text)

    print("=" * 60)
    print("🎬 KẾT QUẢ PHÂN TÍCH CẢM XÚC - CINESENTIMENT")
    print("=" * 60)
    print(f'• Nội dung          : "{text[:80]}{"..." if len(text) > 80 else ""}"')
    print(f"• Cảm xúc dự đoán   : [{result.label.upper()}]")
    print(f"• Xác suất Positive : {result.probability:.2%}")
    print(f"• Số lượng token    : {result.token_count}")
    print(f"• Tỷ lệ OOV         : {result.oov_rate:.2%}")
    if result.truncated:
        print("• Cảnh báo          : Văn bản bị cắt ngắn do vượt quá độ dài tối đa.")
    if result.warnings:
        print(f"• Cảnh báo khác     : {', '.join(result.warnings)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
