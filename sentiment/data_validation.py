"""Kiểm tra và làm sạch dữ liệu dùng chung cho baseline và mô hình PyTorch."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"text", "label"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Đọc CSV, kiểm tra schema và loại bản ghi không hợp lệ hoặc trùng lặp."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu: {path}")

    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Thiếu cột bắt buộc trong tệp CSV: {columns}")

    frame = frame.loc[:, ["text", "label"]].dropna().copy()
    frame["text"] = frame["text"].astype(str)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Cột label chỉ được chứa giá trị 0 hoặc 1.")

    # Nhãn mâu thuẫn cho cùng nội dung là lỗi dữ liệu, không được âm thầm giữ bản đầu.
    label_counts = frame.groupby("text")["label"].nunique()
    if (label_counts > 1).any():
        raise ValueError("Phát hiện nội dung trùng lặp nhưng có nhãn mâu thuẫn.")

    frame["label"] = frame["label"].astype("float32")
    frame = frame.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Tập dữ liệu không chứa mẫu hợp lệ nào sau khi làm sạch.")
    return frame


def remove_train_test_overlap(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame
) -> pd.DataFrame:
    """Loại khỏi test những nội dung đã xuất hiện trong dữ liệu train nguồn."""
    clean_test = test_frame.loc[
        ~test_frame["text"].isin(set(train_frame["text"]))
    ].copy()
    if clean_test.empty:
        raise ValueError(
            "Tập test không còn mẫu độc lập nào sau khi loại dữ liệu trùng với train."
        )
    return clean_test.reset_index(drop=True)

