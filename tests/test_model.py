import pytest

torch = pytest.importorskip("torch")

from sentiment.config import ExperimentConfig
from sentiment.model import SentimentRNN


@pytest.mark.parametrize("model_type", ["lstm", "gru", "bilstm"])
def test_model_tra_mot_logit_cho_moi_mau(model_type):
    config = ExperimentConfig(model_type=model_type, embedding_dim=8, hidden_dim=8, num_layers=1)
    model = SentimentRNN(vocabulary_size=20, padding_index=0, config=config)
    tokens = torch.tensor([[2, 3, 0], [4, 5, 6]])
    lengths = torch.tensor([2, 3])
    assert model(tokens, lengths).shape == (2,)

