from compare_models import create_markdown


def test_create_markdown_hien_thi_dung_metric():
    table = create_markdown(
        [{"model": "GRU", "loss": 0.321, "accuracy": 0.88, "macro_f1": 0.87}]
    )
    assert "GRU" in table
    assert "88.00%" in table
    assert "87.00%" in table

