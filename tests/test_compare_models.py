from compare_models import format_table


def test_format_table_hien_thi_dung_metric():
    rows = [
        {
            "model": "BiLSTM (Calibrated)",
            "accuracy": 0.88,
            "macro_f1": 0.87,
            "brier_score": 0.08,
            "ece": 0.03,
            "params": 6000000,
            "latency_ms": 12.5,
        }
    ]
    table = format_table(rows)
    assert "BiLSTM" in table
    assert "88.00%" in table
    assert "87.00%" in table
    assert "12.50 ms" in table
