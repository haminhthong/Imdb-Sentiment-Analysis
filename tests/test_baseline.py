from baseline import create_pipeline, evaluate


def test_baseline_huan_luyen_va_du_doan_duoc():
    texts = [
        "great movie",
        "great acting",
        "bad movie",
        "bad acting",
    ]
    labels = [1, 1, 0, 0]
    pipeline = create_pipeline(max_features=100)
    pipeline.fit(texts, labels)
    metrics = evaluate(pipeline, texts, labels)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
