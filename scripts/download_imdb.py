"""Tải Large Movie Review Dataset (IMDB) và chuẩn hóa thành train.csv và test.csv."""

import argparse
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def read_split(root: Path, split: str) -> pd.DataFrame:
    """Đọc pos/neg review từ archive thành DataFrame hai cột text và label."""
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
    """Tải và xuất train/test CSV."""
    parser = argparse.ArgumentParser(description="Tải Large Movie Review Dataset (IMDB)")
    parser.add_argument("--url", default=DEFAULT_URL)
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

    extract_dir = raw_dir / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    print("Đang giải nén dữ liệu...")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            tar.extractall(extract_dir)

    dataset_root = extract_dir / "aclImdb"
    train_frame = read_split(dataset_root, "train")
    test_frame = read_split(dataset_root, "test")
    train_frame.to_csv(train_csv, index=False, encoding="utf-8")
    test_frame.to_csv(test_csv, index=False, encoding="utf-8")
    print(f"Đã tạo {train_csv} ({len(train_frame)} mẫu) và {test_csv} ({len(test_frame)} mẫu).")


if __name__ == "__main__":
    main()
