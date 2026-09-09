"""Kiểm thử tự động cho module data.py (nạp dữ liệu, chống rò rỉ, DataBundle)."""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sentiment.config import ExperimentConfig
from sentiment.data import IMDBDataset, create_data_bundle
from sentiment.data_validation import load_dataset
from sentiment.text import build_vocabulary


def test_load_dataset_tu_choi_gia_tri_khuyet_thieu(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {"text": ["Great movie!", "Great movie!", "Terrible plot.", None], "label": [1, 1, 0, 1]}
    )
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="giá trị khuyết thiếu"):
        load_dataset(csv_file)


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
    pd.DataFrame(
        {
            "text": [f"Movie {index}" for index in range(1, 11)],
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    ).to_csv(train_csv, index=False)

    config = ExperimentConfig(
        validation_size=0.2,
        calibration_size=0.1,
        max_length=8,
        batch_size=2,
    )
    bundle = create_data_bundle(train_csv, config)

    assert bundle.sizes["train"] == 7
    assert bundle.sizes["validation"] == 2
    assert bundle.sizes["calibration"] == 1
    assert not hasattr(bundle, "test")
