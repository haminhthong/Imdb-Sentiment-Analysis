import pytest

from sentiment.config import ExperimentConfig


def test_config_tu_choi_ty_le_validation_khong_hop_le():
    with pytest.raises(ValueError, match="validation_size"):
        ExperimentConfig(validation_size=1.0)


def test_config_tu_choi_batch_size_am():
    with pytest.raises(ValueError, match="batch_size"):
        ExperimentConfig(batch_size=-1)
