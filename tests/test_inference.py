"""Kiểm thử tự động cho module suy luận inference.py."""

import joblib
import pytest

from baseline import create_pipeline
from sentiment.artifacts import save_checkpoint
from sentiment.config import ExperimentConfig
from sentiment.inference import SentimentPredictor, load_predictor
from sentiment.model import SentimentRNN
from sentiment.text import build_vocabulary


def test_predictor_bao_loi_khi_checkpoint_khong_ton_tai(tmp_path):
    with pytest.raises(FileNotFoundError, match="Không tìm thấy tệp checkpoint"):
        SentimentPredictor(tmp_path / "missing.pt")


def test_predictor_du_doan_don_va_batch(tmp_path):
    checkpoint_file = tmp_path / "model.pt"
    config = ExperimentConfig(model_type="bilstm", embedding_dim=8, hidden_dim=8, num_layers=1)
    vocab = build_vocabulary(["good movie", "bad story"], min_frequency=1)
    model = SentimentRNN(len(vocab), vocab.pad_index, config)

    save_checkpoint(checkpoint_file, model, vocab, config)

    predictor = SentimentPredictor(checkpoint_file, device="cpu")
    result = predictor.predict("A good movie!")
    assert result.label in {"Positive", "Negative"}
    assert 0.0 <= result.positive_probability <= 1.0

    batch_results = predictor.predict_batch(["Great movie", "Terrible film"])
    assert len(batch_results) == 2


def test_load_predictor_ho_tro_baseline_joblib(tmp_path):
    pipeline = create_pipeline(max_features=100)
    pipeline.fit(["good movie", "bad movie"], [1, 0])
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)

    predictor = load_predictor(model_path)
    result = predictor.predict("good movie")

    assert predictor.config.model_type == "tfidf_logistic_regression"
    assert predictor.count_parameters() > 0
    assert result.label in {"Positive", "Negative"}
    assert result.truncated is False
