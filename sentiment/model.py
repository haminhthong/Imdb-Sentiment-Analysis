"""Mô hình Bidirectional LSTM (BiLSTM) cho bài toán phân loại cảm xúc.

Đặc điểm kỹ thuật:
- Lớp Embedding có padding_idx để bỏ qua cập nhật gradient cho token <PAD>.
- Sử dụng PyTorch `pack_padded_sequence` để RNN không lãng phí tính toán trên vùng padding.
- Ghép hidden state của chiều tiến và chiều lùi ở layer cuối: concat(h_forward, h_backward).
- Áp dụng Dropout để chống overfitting trước khi qua Linear layer chiếu ra 1 logit duy nhất.
- Trả về raw logit (chưa qua Sigmoid) để tương thích tối ưu với `nn.BCEWithLogitsLoss`.
"""

import torch
from torch import nn

from .config import ExperimentConfig


class BiLSTMSentimentClassifier(nn.Module):
    """Mô hình phân loại nhị phân dựa trên mạng BiLSTM với pack_padded_sequence."""

    def __init__(self, vocabulary_size: int, padding_index: int, config: ExperimentConfig) -> None:
        """Khởi tạo các lớp mạng neural.

        Args:
            vocabulary_size: Kích thước bộ từ vựng.
            padding_index: Chỉ số của token `<PAD>`.
            config: Cấu hình siêu tham số của mô hình.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size, config.embedding_dim, padding_idx=padding_index
        )
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, 1),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Thực hiện luồng lan truyền tiến (Forward Pass).

        Args:
            tokens: Batch các chuỗi chỉ số từ [batch_size, max_length].
            lengths: Độ dài thực tế của từng chuỗi [batch_size].

        Returns:
            torch.Tensor: Logits dự đoán [batch_size].
        """
        embedded = self.embedding(tokens)

        # Nén chuỗi để LSTM bỏ qua các bước thời gian thuộc vùng padding
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # Ghép hidden state của hướng tiến (hidden[-2]) và hướng lùi (hidden[-1]) ở layer cuối
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)

        return self.classifier(features).squeeze(1)

    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể huấn luyện (Trainable Parameters)."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


# Alias tương thích
SentimentRNN = BiLSTMSentimentClassifier
