import torch

from sentiment.config import ExperimentConfig
from sentiment.model import BiLSTMSentimentClassifier, SentimentRNN


def test_bilstm_tra_mot_logit_cho_moi_mau():
    config = ExperimentConfig(model_type="bilstm", embedding_dim=8, hidden_dim=8, num_layers=1)
    model = BiLSTMSentimentClassifier(vocabulary_size=20, padding_index=0, config=config)
    tokens = torch.tensor([[2, 3, 0], [4, 5, 6]])
    lengths = torch.tensor([2, 3])
    logits = model(tokens, lengths)
    assert logits.shape == (2,)
    assert model.count_parameters() > 0


def test_sentiment_rnn_alias():
    assert SentimentRNN is BiLSTMSentimentClassifier
