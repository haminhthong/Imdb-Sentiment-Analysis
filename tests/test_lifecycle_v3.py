"""Regression tests cho contract lifecycle v3."""

import pandas as pd
import pytest

from sentiment.config import ExperimentConfig
from sentiment.data import create_data_bundle
from sentiment.data_validation import validate_official_test_independence
from sentiment.text import build_vocabulary, encode_with_audit
from sentiment.calibration import tune_confidence_threshold


def _frame(size: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"review {i} {'good' if i % 2 else 'bad'}" for i in range(size)],
            "label": [i % 2 for i in range(size)],
        }
    )


def test_development_bundle_has_three_roles_and_no_test(tmp_path):
    """Training bundle không đọc/không tạo Official Test loader."""
    path = tmp_path / "official_train.csv"
    _frame().to_csv(path, index=False)
    bundle = create_data_bundle(path, ExperimentConfig(min_frequency=1))

    assert bundle.sizes == {"train": 16, "validation": 2, "calibration": 2}
    assert not hasattr(bundle, "test")
    assert "test" not in bundle.audit
    assert bundle.audit["vocabulary_scope"] == "train_only"


def test_small_development_dataset_still_has_three_nonempty_roles(tmp_path):
    """Smoke dataset nhỏ không làm hỏng invariant ba vai trò dữ liệu."""
    path = tmp_path / "small_train.csv"
    _frame(3).to_csv(path, index=False)
    bundle = create_data_bundle(path, ExperimentConfig(min_frequency=1))

    assert bundle.sizes == {"train": 1, "validation": 1, "calibration": 1}


def test_official_test_overlap_fails_without_modifying_frame():
    train = pd.DataFrame({"text": ["Same review"], "label": [1]})
    official_test = pd.DataFrame({"text": ["same review", "Independent review"], "label": [1, 0]})
    original = official_test.copy(deep=True)
    with pytest.raises(ValueError, match="DATASET AUDIT FAILED"):
        validate_official_test_independence(train, official_test)
    pd.testing.assert_frame_equal(official_test, original)


def test_oov_audit_distinguishes_input_and_used_tokens():
    vocabulary = build_vocabulary(["known token"], min_frequency=1)
    _, _, audit = encode_with_audit(
        "known unknown " + "extra " * 20,
        vocabulary,
        max_length=4,
        strategy="first",
    )
    assert audit["input_tokens"] > audit["used_tokens"]
    assert audit["input_oov_rate"] > audit["used_oov_rate"]


def test_confidence_threshold_is_tuned_on_calibration_curve():
    threshold, curve = tune_confidence_threshold(
        [0, 1, 1, 0], [0.05, 0.95, 0.55, 0.45], min_coverage=0.5
    )
    assert 0.5 <= threshold <= 1.0
    assert {"coverage", "accepted_accuracy", "review_rate"} <= curve[0].keys()
