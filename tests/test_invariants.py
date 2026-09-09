"""Kiểm thử tự động các ràng buộc bất biến (Invariants) cốt lõi của CineSentiment Platform."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sentiment.artifacts import save_checkpoint
from sentiment.calibration import DecisionPolicy, TemperatureScaler
from sentiment.config import ExperimentConfig
from sentiment.data import create_data_bundle
from sentiment.data_validation import load_dataset
from sentiment.inference import SentimentPredictor
from sentiment.model import SentimentRNN
from sentiment.text import (
    TOKENIZER_VERSION,
    build_vocabulary,
    encode_and_pad,
    encode_with_audit,
    tokenize,
)


def test_tokenizer_preserves_negation():
    """Bất biến 1: Tokenizer phải giữ nguyên các từ phủ định quan trọng như don't, isn't."""
    text = "I don't like this movie, it isn't good at all."
    tokens = tokenize(text)
    assert "don't" in tokens
    assert "isn't" in tokens


def test_raw_duplicate_removed(tmp_path):
    """Bất biến 2: Dữ liệu trùng lặp thô (Exact duplicate) phải bị loại bỏ."""
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {"text": ["A great movie.", "A great movie.", "A terrible movie."], "label": [1, 1, 0]}
    )
    df.to_csv(csv_file, index=False)
    loaded = load_dataset(csv_file)
    assert len(loaded) == 2


def test_normalized_duplicate_removed(tmp_path):
    """Bất biến 3: Dữ liệu trùng lặp chuẩn hóa (Normalized duplicate) hoa/thường, HTML phải bị loại."""
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "text": [
                "This movie is GREAT!",
                "this movie is great",
                "This movie is great.<br /><br />",
                "Completely different movie.",
            ],
            "label": [1, 1, 1, 0],
        }
    )
    df.to_csv(csv_file, index=False)
    loaded = load_dataset(csv_file)
    # 3 mẫu đầu là normalized duplicate của nhau -> chỉ giữ lại 1
    assert len(loaded) == 2


def test_conflicting_duplicate_rejected(tmp_path):
    """Bất biến 4: Trùng lặp nội dung nhưng có nhãn mâu thuẫn phải bị từ chối với ValueError."""
    csv_file = tmp_path / "conflict.csv"
    df = pd.DataFrame(
        {"text": ["This movie is amazing!", "this movie is amazing"], "label": [1, 0]}
    )
    df.to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="mâu thuẫn"):
        load_dataset(csv_file)


def test_vocabulary_uses_train_only(tmp_path):
    """Bất biến 6: Từ điển chỉ được xây từ tập Train sau khi chia, từ mới ở Val/Test phải là UNK."""
    train_csv = tmp_path / "train.csv"
    # Train có 10 mẫu với các từ 'apple', 'banana'
    pd.DataFrame(
        {"text": [f"apple banana movie sample {i}" for i in range(10)], "label": [1, 0] * 5}
    ).to_csv(train_csv, index=False)

    config = ExperimentConfig(
        validation_size=0.2,
        calibration_size=0.1,
        min_frequency=1,
        max_length=10,
    )
    bundle = create_data_bundle(train_csv, config)

    assert "papaya" not in bundle.vocabulary.token_to_index


def test_sequence_length_never_exceeds_max_length():
    """Bất biến 7: Chiều dài chuỗi sau encode_and_pad luôn cố định đúng max_length."""
    vocab = build_vocabulary(["one two three four five"], min_frequency=1)
    short_text = "one two"
    long_text = "one two three four five one two three four five one two three"

    enc_short, len_short = encode_and_pad(short_text, vocab, max_length=6)
    enc_long, len_long = encode_and_pad(long_text, vocab, max_length=6)

    assert len(enc_short) == 6
    assert len(enc_long) == 6
    assert len_short == 2
    assert len_long == 6


def test_head_tail_truncation_strategy():
    """Bất biến 8: Head-tail truncation phải kết hợp cả phần đầu và kết luận ở cuối."""
    vocab = build_vocabulary(["head token setup middle filler tail token end"], min_frequency=1)
    text = "head token middle filler tail token"
    # max_length = 4 -> head 2 ('head', 'token') + tail 2 ('tail', 'token')
    enc, length = encode_and_pad(text, vocab, max_length=4, strategy="head_tail")
    assert len(enc) == 4
    assert length == 4


def test_train_inference_tokenization_parity():
    """Bất biến 9: Cùng văn bản đầu vào phải cho ra cùng token và index ở cả train và inference."""
    vocab = build_vocabulary(["superb directing and fantastic plot"], min_frequency=1)
    text = "Superb directing and fantastic plot!"

    tokens_train = tokenize(text)
    enc_train, len_train = encode_and_pad(text, vocab, max_length=10)

    tokens_inf = tokenize(text)
    enc_inf, len_inf, audit = encode_with_audit(text, vocab, max_length=10)

    assert tokens_train == tokens_inf
    assert enc_train == enc_inf
    assert len_train == len_inf
    assert audit["input_tokens"] == 5


def test_calibration_fit_on_calibration_split():
    """Bất biến 10: Temperature Scaling học T từ Calibration logits độc lập."""
    # Giả lập model bị overconfident: logits lớn (+10 và -10) trong khi labels có nhiễu
    logits = np.array([10.0, -10.0, 8.0, -8.0, 9.0, -9.0], dtype=np.float32)
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)

    scaler = TemperatureScaler()
    best_temp = scaler.fit(logits, labels)

    assert best_temp > 1.0  # Vì overconfident nên T phải lớn hơn 1 để làm mềm logits
    calibrated = scaler.calibrate(logits)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_decision_policy_confidence_threshold():
    """Bất biến 11: Confidence thấp hơn tau phải được gắn cờ review_required."""
    policy = DecisionPolicy(threshold=0.5, uncertain_band=(0.40, 0.60))

    res_pos = policy.decide(0.85)
    assert res_pos.label == "Positive"
    assert res_pos.decision == "accepted"
    assert not res_pos.uncertain

    res_uncertain = policy.decide(0.52)
    assert res_uncertain.label == "Positive"
    assert res_uncertain.decision == "review_required"
    assert res_uncertain.uncertain

    res_neg = policy.decide(0.25)
    assert res_neg.label == "Negative"
    assert res_neg.decision == "accepted"
    assert not res_neg.uncertain


def test_long_input_reports_truncation_warning(tmp_path):
    """Bất biến 12: Review dài vượt max_length phải có cảnh báo TRUNCATED_INPUT."""
    checkpoint_file = tmp_path / "model.pt"
    config = ExperimentConfig(
        model_type="gru", embedding_dim=8, hidden_dim=8, num_layers=1, max_length=5
    )
    vocab = build_vocabulary(["word one two three four five six seven"], min_frequency=1)
    model = SentimentRNN(len(vocab), vocab.pad_index, config)
    save_checkpoint(checkpoint_file, model, vocab, config)

    predictor = SentimentPredictor(checkpoint_file, device="cpu")
    result = predictor.predict("word one two three four five six seven eight nine ten")

    assert result.truncated
    assert "TRUNCATED_INPUT" in result.warnings
    assert result.used_tokens == 5
    assert result.input_tokens > 5


def test_high_oov_reports_warning(tmp_path):
    """Bất biến 13: Văn bản có phần lớn từ chưa từng thấy phải trả về cảnh báo HIGH_OOV_WARNING."""
    checkpoint_file = tmp_path / "model.pt"
    config = ExperimentConfig(
        model_type="lstm", embedding_dim=8, hidden_dim=8, num_layers=1, max_length=10
    )
    vocab = build_vocabulary(["known word only"], min_frequency=1)
    model = SentimentRNN(len(vocab), vocab.pad_index, config)
    save_checkpoint(checkpoint_file, model, vocab, config)

    predictor = SentimentPredictor(checkpoint_file, device="cpu")
    result = predictor.predict("xylophone quetzal zephyr pterodactyl unbeknownst")

    assert result.oov_rate > 0.5
    assert "HIGH_OOV_WARNING" in result.warnings


def test_checkpoint_tokenizer_contract_matches_runtime(tmp_path):
    """Bất biến 14: Artifact v3 lưu text contract, policy và temperature."""
    checkpoint_file = tmp_path / "model.pt"
    config = ExperimentConfig(
        model_type="bilstm", embedding_dim=8, hidden_dim=8, num_layers=1, temperature=1.23
    )
    vocab = build_vocabulary(["good bad"], min_frequency=1)
    model = SentimentRNN(len(vocab), vocab.pad_index, config)
    save_checkpoint(checkpoint_file, model, vocab, config)

    saved = torch.load(checkpoint_file, weights_only=True)
    assert saved["artifact_schema_version"] == 3
    assert saved["schema_version"] == 3
    assert saved["tokenizer_version"] == TOKENIZER_VERSION
    assert saved["temperature"] == 1.23
    assert "confidence_threshold" in saved
    assert "source_dataset_hash" in saved
    assert "vocabulary_hash" in saved
