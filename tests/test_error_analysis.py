import numpy as np
import pandas as pd

from analyze_errors import analyze_linguistic_slices, collect_errors


def test_collect_errors_chi_giu_du_doan_sai_va_sap_xep_confidence():
    frame = pd.DataFrame({"text": ["a", "b", "c"], "label": [1, 0, 1]})
    predictions = np.array([0, 1, 1])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    errors = collect_errors(frame, predictions, probabilities)
    assert errors["text"].tolist() == ["b", "a"]
    assert errors["confidence"].tolist() == [0.9, 0.8]


def test_linguistic_slices_doc_bool_tu_csv():
    total = pd.DataFrame({"has_negation": ["False", "True", "False"]})
    errors = pd.DataFrame({"has_negation": ["True"]})

    result = analyze_linguistic_slices(errors, total)

    assert result["negation"]["total_matching_samples"] == 1
    assert result["negation"]["errors_count"] == 1
