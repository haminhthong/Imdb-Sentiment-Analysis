import pandas as pd
import pytest

from sentiment.data_validation import load_dataset, validate_official_test_independence


def test_load_dataset_tu_choi_nhan_mau_thuan(tmp_path):
    path = tmp_path / "conflict.csv"
    pd.DataFrame({"text": ["Same review", "Same review"], "label": [0, 1]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="nhãn mâu thuẫn"):
        load_dataset(path)


def test_official_test_overlap_fail_fast_and_keeps_frame_immutable():
    train = pd.DataFrame({"text": ["Same"], "label": [1]})
    test = pd.DataFrame({"text": ["Same"], "label": [1]})
    original = test.copy(deep=True)
    with pytest.raises(ValueError, match="DATASET AUDIT FAILED"):
        validate_official_test_independence(train, test)
    pd.testing.assert_frame_equal(test, original)
