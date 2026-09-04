import pandas as pd
import pytest

from sentiment.data_validation import load_dataset, remove_train_test_overlap


def test_load_dataset_tu_choi_nhan_mau_thuan(tmp_path):
    path = tmp_path / "conflict.csv"
    pd.DataFrame({"text": ["Same review", "Same review"], "label": [0, 1]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="nhãn mâu thuẫn"):
        load_dataset(path)


def test_remove_overlap_bao_loi_khi_test_khong_con_mau():
    train = pd.DataFrame({"text": ["Same"], "label": [1]})
    test = pd.DataFrame({"text": ["Same"], "label": [1]})
    with pytest.raises(ValueError, match="không còn mẫu độc lập"):
        remove_train_test_overlap(train, test)

