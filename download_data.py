"""Script tiện ích thiết lập và tạo dữ liệu thử nghiệm mẫu cho CineSentiment AI.

Nếu chưa có tệp `train.csv` và `test.csv`, script này sẽ tự động khởi tạo dữ liệu
đánh giá phim mẫu hợp lệ để người dùng và nhà tuyển dụng có thể trải nghiệm chạy
pipeline huấn luyện ngay lập tức mà không gặp lỗi thiếu dữ liệu.
"""

from pathlib import Path
import pandas as pd

SAMPLE_TRAIN_DATA = [
    {"text": "This movie was absolutely fantastic! Great acting and brilliant plot.", "label": 1},
    {"text": "Terrible film. The story was boring and actors were awful.", "label": 0},
    {"text": "A true masterpiece of modern cinema. Highly recommended!", "label": 1},
    {"text": "I hated every minute of this movie. Waste of time and money.", "label": 0},
    {"text": "Wonderful performance by the lead actor. Visually impressive.", "label": 1},
    {"text": "Dull, slow, and completely uninspiring story.", "label": 0},
    {"text": "An emotional roller coaster with incredible character development.", "label": 1},
    {"text": "Worst movie I have ever watched in my entire life.", "label": 0},
    {"text": "Captivating visuals and stunning soundtrack. Loved it!", "label": 1},
    {"text": "Poor directing and cheesy dialogues. Do not recommend.", "label": 0},
]

SAMPLE_TEST_DATA = [
    {"text": "Exceptionally good movie with stunning visual effects.", "label": 1},
    {"text": "Horrible movie with weak script and terrible pacing.", "label": 0},
    {"text": "Brilliant storytelling and magnificent performances overall.", "label": 1},
    {"text": "Total disaster of a film. Utterly boring from start to finish.", "label": 0},
]


def ensure_dataset(train_path: str = "train.csv", test_path: str = "test.csv") -> None:
    """Kiểm tra và khởi tạo dữ liệu mẫu nếu chưa có dữ liệu chính thức."""
    train_file = Path(train_path)
    test_file = Path(test_path)

    if not train_file.exists():
        print(f"Không tìm thấy {train_file}, đang khởi tạo tệp mẫu train...")
        df_train = pd.DataFrame(SAMPLE_TRAIN_DATA)
        df_train.to_csv(train_file, index=False, encoding="utf-8")
        print(f"Đã khởi tạo {train_file} với {len(df_train)} mẫu.")
    else:
        print(f"Tệp dữ liệu train khả dụng: {train_file}")

    if not test_file.exists():
        print(f"Không tìm thấy {test_file}, đang khởi tạo tệp mẫu test...")
        df_test = pd.DataFrame(SAMPLE_TEST_DATA)
        df_test.to_csv(test_file, index=False, encoding="utf-8")
        print(f"Đã khởi tạo {test_file} với {len(df_test)} mẫu.")
    else:
        print(f"Tệp dữ liệu test khả dụng: {test_file}")


if __name__ == "__main__":
    ensure_dataset()
