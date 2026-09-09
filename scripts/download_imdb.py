"""Tải Large Movie Review Dataset và chuẩn hóa thành data/raw/*.csv.

Checksum phải được truyền bởi caller hoặc được ghi trong manifest của dự án.
Script không tạo fallback synthetic và không ghi đè file đã tồn tại nếu chưa
được yêu cầu rõ ràng.
"""

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from sentiment.manifest import build_dataset_manifest, save_manifest

DEFAULT_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def sha256_file(path: Path) -> str:
    """Tính SHA256 theo block để không nạp toàn bộ archive vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(root: Path, split: str) -> pd.DataFrame:
    """Đọc pos/neg review từ archive thành DataFrame hai cột."""
    rows: list[dict[str, int | str]] = []
    for label_name, label in (("pos", 1), ("neg", 0)):
        directory = root / split / label_name
        for path in sorted(directory.glob("*.txt")):
            rows.append(
                {"text": path.read_text(encoding="utf-8", errors="replace"), "label": label}
            )
    if not rows:
        raise ValueError(f"Không tìm thấy review trong {root / split}.")
    return pd.DataFrame(rows)


def main() -> None:
    """Tải, kiểm checksum nếu có và xuất official train/test."""
    parser = argparse.ArgumentParser(description="Tải Large Movie Review Dataset official")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", required=True, help="SHA256 archive do nguồn tin cậy cung cấp")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--force", action="store_true", help="Cho phép ghi đè file CSV đã có")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "aclImdb_v1.tar.gz"
    train_csv = raw_dir / "train.csv"
    test_csv = raw_dir / "test.csv"
    if not args.force and (train_csv.exists() or test_csv.exists()):
        raise FileExistsError(
            "data/raw/train.csv hoặc test.csv đã tồn tại; dùng --force nếu muốn ghi đè."
        )

    print(f"Đang tải {args.url} ...")
    urllib.request.urlretrieve(args.url, archive)
    actual_sha256 = sha256_file(archive)
    if actual_sha256.lower() != args.sha256.lower():
        archive.unlink(missing_ok=True)
        raise ValueError(f"Checksum không khớp: expected={args.sha256}, actual={actual_sha256}")

    extract_dir = raw_dir / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            # Python 3.12+ có filter để chặn path nguy hiểm trong archive.
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            # Docker hiện dùng Python 3.11; archive chính thức có cấu trúc cố định.
            tar.extractall(extract_dir)
    dataset_root = extract_dir / "aclImdb"
    train_frame = read_split(dataset_root, "train")
    test_frame = read_split(dataset_root, "test")
    train_frame.to_csv(train_csv, index=False, encoding="utf-8")
    test_frame.to_csv(test_csv, index=False, encoding="utf-8")
    save_manifest(
        raw_dir.parent / "manifest.json",
        build_dataset_manifest(
            train_frame,
            test_frame,
            source_checksum=actual_sha256,
        ),
    )
    print(f"Đã xác minh archive SHA256={actual_sha256}")
    print(f"Đã tạo {train_csv} và {test_csv}; source archive được giữ tại {archive}.")


if __name__ == "__main__":
    main()
