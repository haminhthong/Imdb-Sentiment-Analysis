"""Kiểm tra tính hợp lệ của dữ liệu và rà soát trùng lặp / rò rỉ.

Chức năng:
1. Xác thực schema cột ('text', 'label') và kiểu nhãn nhị phân {0, 1}.
2. Phát hiện dữ liệu mâu thuẫn nhãn (cùng một nội dung nhưng có cả nhãn 0 và 1).
3. Loại bỏ bản ghi trùng lặp nội bộ (cả trùng lặp thô và trùng lặp chuẩn hóa).
4. Kiểm toán chồng lấn (overlap) giữa tập Train và tập Test.
5. Thống kê phân phối độ dài token (p50, p90, p95, truncation rate).
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .text import normalize_text, tokenize

REQUIRED_COLUMNS = {"text", "label"}


def load_dataset(path: str | Path, *, deduplicate: bool = True) -> pd.DataFrame:
    """Đọc tệp CSV và kiểm tra tính hợp lệ dữ liệu.

    Args:
        path: Đường dẫn tới tệp CSV.
        deduplicate: Nếu True, loại bỏ các bản ghi trùng lặp chuẩn hóa.

    Returns:
        pd.DataFrame sạch với 2 cột 'text' và 'label' (float32).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu: {path}")

    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Thiếu cột bắt buộc trong tệp CSV: {columns}")

    frame = frame.loc[:, ["text", "label"]].copy()
    if frame.isna().any().any():
        raise ValueError("Dữ liệu chứa giá trị khuyết thiếu ở cột text hoặc label.")

    frame["text"] = frame["text"].astype(str)
    if (frame["text"].str.strip() == "").any():
        raise ValueError("Cột text không được chứa văn bản rỗng.")

    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Cột label chỉ được chứa giá trị 0 hoặc 1.")

    # 1. Kiểm tra nhãn mâu thuẫn trên văn bản gốc
    raw_label_counts = frame.groupby("text")["label"].nunique()
    if (raw_label_counts > 1).any():
        raise ValueError("Phát hiện nội dung trùng lặp nhưng có nhãn mâu thuẫn trên văn bản gốc.")

    # 2. Kiểm tra nhãn mâu thuẫn trên văn bản chuẩn hóa
    norm_texts = frame["text"].apply(normalize_text)
    frame["_norm_text"] = norm_texts

    norm_label_counts = frame.groupby("_norm_text")["label"].nunique()
    if (norm_label_counts > 1).any():
        raise ValueError("Phát hiện trùng lặp nội dung nhưng có nhãn mâu thuẫn.")

    frame["label"] = frame["label"].astype("float32")

    if deduplicate:
        frame = frame.drop_duplicates(subset=["_norm_text"], keep="first")

    frame = frame.drop(columns=["_norm_text"]).reset_index(drop=True)

    if frame.empty:
        raise ValueError("Tập dữ liệu không chứa mẫu hợp lệ nào sau khi làm sạch.")

    return frame


def load_official_dataset(path: str | Path) -> pd.DataFrame:
    """Đọc tập dữ liệu chính thức mà không tự động loại bỏ duplicate."""
    return load_dataset(path, deduplicate=False)


def compute_split_overlap(left_frame: pd.DataFrame, right_frame: pd.DataFrame) -> dict[str, int]:
    """Đếm số lượng văn bản bị trùng lặp giữa hai split."""
    left_raw = set(left_frame["text"])
    right_raw = set(right_frame["text"])
    raw_exact_overlap = len(left_raw & right_raw)

    left_norm = set(left_frame["text"].apply(normalize_text))
    right_norm = set(right_frame["text"].apply(normalize_text))
    normalized_exact_overlap = len(left_norm & right_norm)

    return {
        "raw_exact_overlap": raw_exact_overlap,
        "normalized_exact_overlap": normalized_exact_overlap,
        "overlap_count": normalized_exact_overlap,
    }


def validate_official_test_independence(
    train_frame: pd.DataFrame, official_test_frame: pd.DataFrame
) -> dict[str, int]:
    """Kiểm toán tập Test và dừng chương trình nếu phát hiện rò rỉ từ tập Train."""
    overlap = compute_split_overlap(train_frame, official_test_frame)
    if overlap["overlap_count"] > 0:
        raise ValueError(
            "DATASET AUDIT FAILED: Phát hiện trùng lặp giữa Train và Test "
            f"({overlap['overlap_count']} mẫu; raw={overlap['raw_exact_overlap']}, "
            f"normalized={overlap['normalized_exact_overlap']})."
        )
    return overlap


def compute_token_length_distribution(
    texts: Iterable[str], max_length: int = 256
) -> dict[str, float | int]:
    """Phân tích phân phối độ dài token của tập văn bản."""
    lengths = [len(tokenize(text)) for text in texts]
    if not lengths:
        return {
            "count": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "max": 0,
            "truncation_rate": 0.0,
        }

    arr = np.array(lengths)
    return {
        "count": int(len(arr)),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(np.max(arr)),
        "truncation_rate": round(float(np.mean(arr > max_length)), 4),
    }
