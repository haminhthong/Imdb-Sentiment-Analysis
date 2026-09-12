import pandas as pd

from scripts.analyze_errors import NEGATION_PATTERN, analyze_errors_on_dataset
from sentiment.inference import PredictionResult


class MockPredictor:
    def predict_batch(self, texts):
        results = []
        for t in texts:
            is_pos = "good" in t
            results.append(
                PredictionResult(
                    label="Positive" if is_pos else "Negative",
                    probability=0.9 if is_pos else 0.1,
                    token_count=len(t.split()),
                    oov_rate=0.0,
                )
            )
        return results


def test_analyze_errors_on_dataset():
    df = pd.DataFrame(
        {
            "text": [
                "a good movie",
                "not a good movie but bad",
                "a bad movie",
                "another good film",
            ],
            "label": [1, 0, 0, 1],
        }
    )
    predictor = MockPredictor()
    report = analyze_errors_on_dataset(predictor, df)

    assert report["total_samples"] == 4
    assert "linguistic_slices" in report
    assert "length_slices" in report
    assert "oov_slices" in report


def test_negation_pattern():
    assert NEGATION_PATTERN.search("I do not like it")
    assert NEGATION_PATTERN.search("It isn't good")
    assert not NEGATION_PATTERN.search("This is good")
