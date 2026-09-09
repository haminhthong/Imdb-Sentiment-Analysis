"""Mô hình Recurrent Neural Network (RNN) thống nhất cho phân loại cảm xúc.

Module này định nghĩa lớp SentimentRNN hỗ trợ cả 3 kiến trúc:
- Single-direction LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- BiLSTM (Bidirectional LSTM)

Đặc điểm kỹ thuật:
- Sử dụng Embedding layer có padding_idx để bỏ qua gradient padding.
- Sử dụng PyTorch `pack_padded_sequence` để tối ưu hóa hiệu năng tính toán.
- Lấy hidden state ở bước thời gian cuối cùng làm biểu diễn vector cho toàn bộ văn bản.
- Trả về raw logits (chưa qua Sigmoid) để tương thích với `nn.BCEWithLogitsLoss`.
"""

import torch
from torch import nn

from .config import ExperimentConfig


class SentimentRNN(nn.Module):
    """Mô hình phân loại nhị phân dựa trên RNN với tính năng bỏ qua padding.

    Attributes:
        embedding (nn.Embedding): Lớp nhúng từ (Vocabulary Size -> Embedding Dim).
        rnn (nn.LSTM | nn.GRU): Lớp RNN cốt lõi (LSTM hoặc GRU).
        classifier (nn.Sequential): Lớp Dropout và Linear chiếu ra 1 logit duy nhất.
        bidirectional (bool): Cờ đánh dấu mô hình có phải hai chiều (BiLSTM) hay không.
    """

    def __init__(self, vocabulary_size: int, padding_index: int, config: ExperimentConfig) -> None:
        """Khởi tạo các lớp mạng neural theo cấu hình ExperimentConfig.

        Args:
            vocabulary_size (int): Kích thước bộ từ vựng.
            padding_index (int): Chỉ số của token `<PAD>` để không cập nhật gradient.
            config (ExperimentConfig): Siêu tham số mô hình.
        """
        super().__init__()
        bidirectional = config.model_type == "bilstm"
        rnn_class = nn.GRU if config.model_type == "gru" else nn.LSTM

        self.embedding = nn.Embedding(
            vocabulary_size, config.embedding_dim, padding_idx=padding_index
        )
        self.rnn = rnn_class(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_features = config.hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(output_features, 1),
        )
        self.bidirectional = bidirectional

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Thực hiện luồng lan truyền tiến (Forward Pass).

        Args:
            tokens (torch.Tensor): Batch các chuỗi chỉ số từ [batch_size, max_length].
            lengths (torch.Tensor): Độ dài thực tế của từng chuỗi [batch_size].

        Returns:
            torch.Tensor: Logits dự đoán [batch_size]. Giá trị > 0 tương ứng Positive.
        """
        embedded = self.embedding(tokens)
        # Nén chuỗi để RNN bỏ qua các bước thời gian thuộc vùng padding
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output = self.rnn(packed)

        # Lấy hidden state cuối cùng (LSTM trả về tuple (h, c), GRU trả về h)
        hidden = output[1][0] if isinstance(output[1], tuple) else output[1]

        if self.bidirectional:
            # Ghép hidden state của chiều tiến (layer cuối) và chiều lùi (layer cuối)
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # Lấy hidden state của layer cuối cùng
            features = hidden[-1]

        return self.classifier(features).squeeze(1)

    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể huấn luyện (Trainable Parameters) của mô hình.

        Returns:
            int: Số lượng tham số cần tối ưu hóa.
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
