"""Kiểm thử tự động cho module calibration.py (Temperature Scaling, Brier, ECE)."""

import numpy as np

from sentiment.calibration import TemperatureScaler, compute_brier_score, compute_ece


def test_temperature_scaler_fit_and_calibrate():
    logits = np.array([10.0, -10.0, 8.0, -8.0, 9.0, -9.0], dtype=np.float32)
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)

    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, labels)

    assert temperature > 1.0  # Vì overconfident nên T phải lớn hơn 1
    calibrated = scaler.calibrate(logits)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_compute_brier_score():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert compute_brier_score(y_true, y_prob) == 0.0

    y_worst = np.array([0.0, 1.0, 0.0, 1.0])
    assert compute_brier_score(y_true, y_worst) == 1.0


def test_compute_ece():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
