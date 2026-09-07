"""Kiểm tra dữ liệu và bảo vệ ranh giới giữa các split.

Module này thực hiện:
1. Kiểm tra cấu trúc Schema và nhãn nhị phân {0, 1}.
2. Phát hiện dữ liệu mâu thuẫn nhãn trên cả văn bản thô (raw text) và chuẩn hóa (normalized text).
3. Phát hiện trùng lặp nội bộ (raw exact và normalized exact).
4. Audit overlap giữa các split; Official Test không bao giờ bị sửa tự động.
5. Thống kê phân phối độ dài token (p50, p90, p95, truncation rate).
"""

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .text import compute_normalized_text_hash, tokenize

REQUIRED_COLUMNS = {"text", "label"}


def load_dataset(path: str | Path, *, deduplicate: bool = True) -> pd.DataFrame:
    """Đọc CSV và kiểm tra schema, nhãn mâu thuẫn.

    Quy trình kiểm tra chất lượng dữ liệu:
    - Xác thực sự tồn tại của tệp và các cột bắt buộc ('text', 'label').
    - Loại bỏ giá trị khuyết thiếu (NaN) và ép kiểu văn bản.
    - Kiểm tra nhãn nhị phân hợp lệ {0, 1}.
    - Bắt lỗi nhãn mâu thuẫn trên raw text và canonical normalized text.
    - Với ``deduplicate=True`` (mặc định), loại duplicate nội bộ theo normalized
      exact hash. Official split nên dùng ``load_official_dataset`` để giữ nguyên
      dữ liệu nguồn và chỉ audit, không sửa benchmark.

    Args:
        path (str | Path): Đường dẫn đến tệp CSV dữ liệu.

    Returns:
        pd.DataFrame: DataFrame sạch gồm 2 cột 'text' và 'label' (float32).
    """
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

    # 1. Kiểm tra nhãn mâu thuẫn trên văn bản thô (Exact Raw Text)
    raw_label_counts = frame.groupby("text")["label"].nunique()
    if (raw_label_counts > 1).any():
        raise ValueError("Phát hiện nội dung trùng lặp nhưng có nhãn mâu thuẫn trên văn bản gốc.")

    # 2. Kiểm tra nhãn mâu thuẫn trên văn bản chuẩn hóa (Normalized Text Hash)
    normalized_hashes = frame["text"].apply(compute_normalized_text_hash)
    frame["_norm_hash"] = normalized_hashes

    norm_label_counts = frame.groupby("_norm_hash")["label"].nunique()
    if (norm_label_counts > 1).any():
        raise ValueError(
            "Phát hiện normalized exact duplicate nhưng có nhãn mâu thuẫn."
        )

    frame["label"] = frame["label"].astype("float32")

    # Chỉ deduplicate ở bước chuẩn bị development data. Official Test phải giữ
    # nguyên từng dòng để hash và benchmark phản ánh đúng nguồn chính thức.
    if deduplicate:
        frame = frame.drop_duplicates(subset=["_norm_hash"], keep="first")
    frame = frame.drop(columns=["_norm_hash"]).reset_index(drop=True)

    if frame.empty:
        raise ValueError("Tập dữ liệu không chứa mẫu hợp lệ nào sau khi làm sạch.")

    return frame


def load_official_dataset(path: str | Path) -> pd.DataFrame:
    """Đọc một official split theo chế độ bất biến (không drop hoặc sửa dòng).

    Hàm vẫn kiểm tra schema, giá trị thiếu và nhãn mâu thuẫn; nó chỉ không loại
    duplicate nội bộ. Nhờ vậy caller có thể phát hiện dataset bất thường thay vì
    âm thầm thay đổi Official Test.
    """
    return load_dataset(path, deduplicate=False)


def compute_split_overlap(
    left_frame: pd.DataFrame, right_frame: pd.DataFrame
) -> dict[str, int]:
    """Đếm raw exact và normalized exact overlap giữa hai split.

    ``normalized_exact`` chỉ là duplicate sau canonical tokenizer, không phải
    semantic near-duplicate. Near-duplicate thật cần MinHash/SimHash/embedding
    similarity và nằm ngoài phạm vi v1.
    """
    left_raw = set(left_frame["text"])
    right_raw = set(right_frame["text"])
    left_norm = set(left_frame["text"].map(compute_normalized_text_hash))
    right_norm = set(right_frame["text"].map(compute_normalized_text_hash))
    raw_overlap = right_frame["text"].isin(left_raw)
    norm_overlap = right_frame["text"].map(compute_normalized_text_hash).isin(left_norm)
    return {
        "raw_exact_overlap": int(raw_overlap.sum()),
        "normalized_exact_overlap": int(norm_overlap.sum()),
        "overlap_count": int((raw_overlap | norm_overlap).sum()),
    }


def validate_official_test_independence(
    train_frame: pd.DataFrame, official_test_frame: pd.DataFrame
) -> dict[str, int]:
    """Audit Official Test và dừng pipeline nếu phát hiện overlap.

    Không trả về một test frame đã lọc. Dataset quality issue phải được xử lý ở
    nguồn dữ liệu, không được biến thành một benchmark mới nhưng vẫn gọi là official.
    """
    overlap = compute_split_overlap(train_frame, official_test_frame)
    if overlap["overlap_count"]:
        raise ValueError(
            "DATASET AUDIT FAILED: Official Test có overlap với Official Train "
            f"({overlap['overlap_count']} mẫu; raw_exact={overlap['raw_exact_overlap']}, "
            f"normalized_exact={overlap['normalized_exact_overlap']}). "
            "Không tự động xóa mẫu khỏi Official Test."
        )
    return overlap


def remove_train_test_overlap(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame
) -> pd.DataFrame:
    """Compatibility helper cũ để xử lý dữ liệu không phải Official Test.

    Hàm này giữ lại để notebook cũ không hỏng. Release evaluator không gọi hàm
    này; nó dùng ``validate_official_test_independence`` và fail-fast.
    """
    train_raw_set = set(train_frame["text"])
    train_norm_hashes = set(train_frame["text"].apply(compute_normalized_text_hash))

    test_norm_hashes = test_frame["text"].apply(compute_normalized_text_hash)

    # Lọc mẫu test không thuộc train_raw_set và normalized hash không thuộc train_norm_hashes
    is_raw_leak = test_frame["text"].isin(train_raw_set)
    is_norm_leak = test_norm_hashes.isin(train_norm_hashes)
    is_clean = ~(is_raw_leak | is_norm_leak)

    clean_test = test_frame.loc[is_clean].copy()
    if clean_test.empty:
        raise ValueError(
            "Tập test không còn mẫu độc lập nào sau khi loại dữ liệu trùng với train."
        )

    return clean_test.reset_index(drop=True)


def compute_token_length_distribution(
    texts: Iterable[str], max_length: int = 256
) -> dict[str, float | int]:
    """Phân tích phân phối độ dài token của tập văn bản.

    Returns:
        dict[str, float | int]: Gồm count, p50, p90, p95, max_tokens, truncation_rate.
    """
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
    truncated_count = int(np.sum(arr > max_length))
    return {
        "count": len(lengths),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(np.max(arr)),
        "truncation_rate": round(truncated_count / len(lengths), 4),
    }


def compute_dataset_hash(frame: pd.DataFrame) -> str:
    """Tạo fingerprint SHA256 cho tập dữ liệu phục vụ artifact tracking."""
    # Chuẩn hóa label về int để hash không đổi giữa DataFrame vừa tạo (1) và
    # DataFrame đọc CSV (1.0/float32).
    labels = frame["label"].astype(int).astype(str)
    combined = frame["text"].str.cat(labels, sep=":::").str.cat(sep="\n")
    return hashlib.sha256(combined.encode("utf-8", errors="replace")).hexdigest()
