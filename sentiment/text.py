"""Tiền xử lý văn bản và quản lý bộ từ vựng (Vocabulary).

Module này cung cấp các chức năng làm sạch văn bản, tách token (tokenization),
xây dựng từ điển chỉ từ tập train để chống rò rỉ dữ liệu, và chuyển đổi chuỗi
thành chuỗi chỉ số (encoding & padding).
"""

import html
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

# Token đặc biệt cho Padding và Out-Of-Vocabulary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Regex giữ lại chữ cái, chữ số và dấu nháy trong các từ phủ định (ví dụ: "don't", "isn't")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
# Regex loại bỏ tất cả các thẻ HTML (ví dụ: <br>, <p>, ...)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def tokenize(text: str) -> list[str]:
    """Chuẩn hóa văn bản, xóa thẻ HTML, chuyển chữ thường và tách token.

    Hàm giải mã các ký tự HTML entity (ví dụ: &amp; -> &), loại bỏ các thẻ HTML,
    chuyển thành chữ thường và trích xuất danh sách token theo biểu thức chính quy.

    Args:
        text (str): Văn bản đánh giá cần tách token.

    Returns:
        list[str]: Danh sách các token từ văn bản đầu vào.
    """
    if not isinstance(text, str):
        return []
    normalized = html.unescape(HTML_TAG_PATTERN.sub(" ", text)).lower()
    return TOKEN_PATTERN.findall(normalized)


@dataclass(frozen=True)
class Vocabulary:
    """Quản lý ánh xạ hai chiều giữa từ (token) và chỉ số số nguyên (index).

    Attributes:
        token_to_index (dict[str, int]): Ánh xạ từ token sang chỉ số số nguyên.
    """

    token_to_index: dict[str, int]

    @property
    def pad_index(self) -> int:
        """Chỉ số của token padding `<PAD>`."""
        return self.token_to_index[PAD_TOKEN]

    @property
    def unknown_index(self) -> int:
        """Chỉ số của token chưa biết `<UNK>`."""
        return self.token_to_index[UNK_TOKEN]

    def __len__(self) -> int:
        """Kích thước tổng cộng của bộ từ vựng."""
        return len(self.token_to_index)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Mã hóa danh sách các token thành chuỗi các chỉ số số nguyên.

        Args:
            tokens (Iterable[str]): Danh sách các từ cần mã hóa.

        Returns:
            list[int]: Danh sách chỉ số tương ứng. Nếu từ không có trong từ điển,
                sẽ dùng chỉ số của `<UNK>`.
        """
        return [self.token_to_index.get(token, self.unknown_index) for token in tokens]

    def to_dict(self) -> dict[str, int]:
        """Xuất từ điển thành dictionary chuẩn."""
        return dict(self.token_to_index)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Vocabulary":
        """Khởi tạo Vocabulary từ dictionary lưu trong checkpoint.

        Args:
            data (dict[str, int]): Dictionary chứa ánh xạ token -> index.

        Returns:
            Vocabulary: Đối tượng Vocabulary đã phục hồi.
        """
        return cls({str(token): int(index) for token, index in data.items()})


def build_vocabulary(
    texts: Iterable[str], min_frequency: int = 5, max_size: int = 50_000
) -> Vocabulary:
    """Tạo bộ từ vựng từ tập dữ liệu huấn luyện (Train Split).

    Để tránh rò rỉ dữ liệu (Data Leakage), bộ từ vựng CHỈ được xây dựng từ tập
    train. Các từ có tần suất xuất hiện nhỏ hơn `min_frequency` hoặc vượt quá
    `max_size` sẽ bị loại bỏ.

    Args:
        texts (Iterable[str]): Danh sách các câu đánh giá thuộc tập train.
        min_frequency (int): Tần suất xuất hiện tối thiểu để giữ lại từ. Mặc định là 5.
        max_size (int): Kích thước tối đa của bộ từ vựng. Mặc định là 50,000.

    Returns:
        Vocabulary: Bộ từ vựng đã được khởi tạo kèm các token đặc biệt `<PAD>` và `<UNK>`.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # Dành 2 vị trí cho token đặc biệt <PAD> (index 0) và <UNK> (index 1)
    capacity = max(0, max_size - 2)
    tokens = [
        token
        for token, count in counter.most_common()
        if count >= min_frequency
    ][:capacity]

    mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    mapping.update({token: index for index, token in enumerate(tokens, start=2)})
    return Vocabulary(mapping)


def encode_and_pad(
    text: str, vocabulary: Vocabulary, max_length: int
) -> tuple[list[int], int]:
    """Mã hóa văn bản thành chuỗi chỉ số có độ dài cố định và trả về độ dài thực.

    Nếu văn bản dài hơn `max_length`, câu sẽ bị cắt ngắn (truncate).
    Nếu ngắn hơn, câu sẽ được chèn thêm token `<PAD>` ở cuối.

    Args:
        text (str): Văn bản đánh giá đầu vào.
        vocabulary (Vocabulary): Bộ từ vựng để mã hóa.
        max_length (int): Độ dài tối đa sau khi pad/truncate.

    Returns:
        tuple[list[int], int]:
            - list[int]: Danh sách chỉ số có độ dài chính xác bằng `max_length`.
            - int: Độ dài thực tế (đã bị giới hạn bởi `max_length`) trước khi pad.
              Dùng cho PyTorch PackedSequence để bỏ qua tính toán với vùng padding.
    """
    encoded = vocabulary.encode(tokenize(text))[:max_length]
    length = max(1, len(encoded))
    if not encoded:
        encoded = [vocabulary.unknown_index]
    encoded.extend([vocabulary.pad_index] * (max_length - len(encoded)))
    return encoded, length

