"""Manifest bất biến cho dataset và các split development/release."""

import json
from pathlib import Path
from typing import Any

from .data_validation import compute_dataset_hash


def build_dataset_manifest(
    source_train: Any,
    official_test: Any | None = None,
    *,
    source: str = "Stanford Large Movie Review Dataset v1.0",
    source_checksum: str = "unspecified",
    split_frames: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tạo manifest JSON từ các frame đã được audit.

    Hàm không tự đọc hay sửa file. Caller chịu trách nhiệm gọi audit overlap
    trước khi truyền Official Test.
    """
    manifest: dict[str, Any] = {
        "dataset": "Large Movie Review Dataset v1.0",
        "source": source,
        "source_checksum": source_checksum,
        "train_samples": int(len(source_train)),
        "source_train_hash": compute_dataset_hash(source_train),
        "splits": {},
    }
    if official_test is not None:
        manifest["test_samples"] = int(len(official_test))
        manifest["official_test_hash"] = compute_dataset_hash(official_test)
    if split_frames:
        manifest["splits"] = {
            name: {
                "samples": int(len(frame)),
                "hash": compute_dataset_hash(frame),
            }
            for name, frame in split_frames.items()
        }
    return manifest


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Ghi manifest UTF-8 ổn định để có thể kiểm tra trong release review."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
