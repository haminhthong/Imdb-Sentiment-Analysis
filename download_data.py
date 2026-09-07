"""Compatibility checker cho dữ liệu IMDB chính thức.

Script này không tự tạo dữ liệu giả. Dữ liệu smoke phải được tạo riêng bằng
``python -m scripts.create_smoke_dataset`` và không được đặt tên như official split.
"""

from pathlib import Path

def ensure_dataset(
    train_path: str = "data/raw/train.csv", test_path: str = "data/raw/test.csv"
) -> None:
    """Kiểm tra file official và fail-fast nếu dữ liệu chưa được tải."""
    train_file = Path(train_path)
    test_file = Path(test_path)

    missing = [str(path) for path in (train_file, test_file) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Thiếu dữ liệu IMDB official: " + ", ".join(missing) + ". "
            "Hãy chạy python -m scripts.download_imdb; không dùng dữ liệu giả thay thế."
        )
    print(f"Official train/test khả dụng: {train_file}, {test_file}")


if __name__ == "__main__":
    ensure_dataset()
