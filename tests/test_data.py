"""Kiểm thử tự động cho module data.py (nạp dữ liệu, chống rò rỉ, DataBundle)."""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sentiment.config import ExperimentConfig
from sentiment.data import IMDBDataset, create_data_bundle, load_dataset
from sentiment.text import build_vocabulary


def test_load_dataset_loai_bo_trung_lap_va_nan(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "text": ["Great movie!", "Great movie!", "Terrible plot.", None],
        "label": [1, 1, 0, 1]
    })
    df.to_csv(csv_file, index=False)

    loaded = load_dataset(csv_file)
    assert len(loaded) == 2  # 1 dòng NaN bị xóa, 1 dòng trùng bị xóa
    assert set(loaded["label"].unique()) == {0.0, 1.0}


def test_load_dataset_bao_loi_khi_thieu_cot(tmp_path):
    csv_file = tmp_path / "invalid.csv"
    pd.DataFrame({"invalid_col": [1, 2]}).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Thiếu cột bắt buộc"):
        load_dataset(csv_file)


def test_imdb_dataset_tra_tensor_dung_kieu_va_shape():
    df = pd.DataFrame({"text": ["Good movie", "Bad plot"], "label": [1, 0]})
    vocab = build_vocabulary(df["text"], min_frequency=1)
    dataset = IMDBDataset(df, vocab, max_length=10)

    tokens, length, label = dataset[0]
    assert isinstance(tokens, torch.Tensor)
    assert tokens.dtype == torch.long
    assert len(tokens) == 10
    assert length.item() == 2
    assert label.item() == 1.0


def test_create_data_bundle_chong_ro_ri_du_lieu(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    pd.DataFrame({
        "text": [f"Movie {index}" for index in range(1, 11)],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }).to_csv(train_csv, index=False)

    pd.DataFrame({
        "text": ["Movie 1", "Movie 11", "Movie 12"],  # Movie 1 đã xuất hiện ở train!
        "label": [1, 0, 1]
    }).to_csv(test_csv, index=False)

    config = ExperimentConfig(validation_size=0.2, max_length=8, batch_size=2)
    bundle = create_data_bundle(train_csv, test_csv, config)

    assert bundle.sizes["train"] == 8
    assert bundle.sizes["validation"] == 2
    assert bundle.sizes["test"] == 2  # Movie 1 bị loại bỏ khỏi test!
