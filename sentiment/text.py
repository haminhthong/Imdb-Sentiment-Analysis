"""Tiền xử lý văn bản và quản lý bộ từ vựng (Vocabulary).

Module này cung cấp các chức năng làm sạch văn bản, tách token (tokenization),
băm văn bản (exact & normalized hash) để chống rò rỉ dữ liệu, xây dựng từ điển
chỉ từ tập train, và chuyển đổi chuỗi thành chuỗi chỉ số (encoding & padding)
với các chiến lược cắt chuỗi (first, head_tail).
"""

import hashlib
import html
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal

TOKENIZER_VERSION = "word-regex-v1"

# Token đặc biệt cho Padding và Out-Of-Vocabulary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Regex giữ lại chữ cái, chữ số và dấu nháy trong các từ phủ định (ví dụ: "don't", "isn't")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
# Regex loại bỏ tất cả các thẻ HTML (ví dụ: <br>, <p>, ...)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Tập từ vựng tiếng Anh phổ biến để kiểm tra cảnh báo ngôn ngữ ngoài miền
COMMON_ENGLISH_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "it", "this",
    "that", "movie", "film", "actor", "actors", "story", "plot", "good",
    "bad", "great", "one", "all", "see", "watch", "time", "very", "not",
}


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


def compute_raw_hash(text: str) -> str:
    """Tính mã băm SHA256 của chuỗi văn bản gốc (Raw Text Hash)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def compute_normalized_text_hash(text: str) -> str:
    """Tính mã băm SHA256 sau khi chuẩn hóa canonical tokenization.

    Hai văn bản chỉ khác nhau về chữ hoa/thường, thẻ HTML (như <br />),
    khoảng trắng thừa hoặc dấu câu ngoại lai sẽ có cùng normalized hash.
    """
    tokens = tokenize(text)
    canonical_representation = " ".join(tokens)
    return hashlib.sha256(
        canonical_representation.encode("utf-8", errors="replace")
    ).hexdigest()


def detect_language_warning(text: str) -> list[str]:
    """Kiểm tra sơ bộ tính phù hợp ngôn ngữ tiếng Anh của văn bản đánh giá.

    Returns:
        list[str]: Danh sách các cảnh báo (ví dụ ['NON_ENGLISH_WARNING']) nếu phát hiện.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    tokens = tokenize(text)
    if not tokens:
        return []

    # Kiểm tra tỷ lệ ký tự non-ASCII
    total_chars = len(text)
    non_ascii_chars = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = non_ascii_chars / total_chars if total_chars > 0 else 0.0

    # Nếu câu có độ dài tương đối (>= 5 tokens) nhưng không chứa từ tiếng Anh quen thuộc nào
    # hoặc tỷ lệ ký tự lạ vượt 25%
    has_english_token = any(token in COMMON_ENGLISH_WORDS for token in tokens)
    if (len(tokens) >= 5 and not has_english_token) or non_ascii_ratio > 0.25:
        return ["NON_ENGLISH_WARNING"]

    return []


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

    def compute_oov_stats(self, tokens: Iterable[str]) -> tuple[int, float]:
        """Tính số lượng từ OOV và tỷ lệ OOV trên danh sách token.

        Returns:
            tuple[int, float]: (oov_count, oov_rate).
        """
        token_list = list(tokens)
        if not token_list:
            return 0, 0.0
        oov_count = sum(1 for t in token_list if t not in self.token_to_index)
        return oov_count, oov_count / len(token_list)

    def compute_hash(self) -> str:
        """Tạo mã băm SHA256 cho toàn bộ bộ từ điển để đảm bảo version contract."""
        sorted_pairs = sorted(self.token_to_index.items())
        serialized = ";".join(f"{token}:{idx}" for token, idx in sorted_pairs)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

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


TruncationStrategy = Literal["first", "head_tail"]


def encode_and_pad(
    text: str,
    vocabulary: Vocabulary,
    max_length: int,
    strategy: TruncationStrategy = "first",
) -> tuple[list[int], int]:
    """Mã hóa văn bản thành chuỗi chỉ số có độ dài cố định và trả về độ dài thực.

    Args:
        text (str): Văn bản đánh giá đầu vào.
        vocabulary (Vocabulary): Bộ từ vựng để mã hóa.
        max_length (int): Độ dài tối đa sau khi pad/truncate.
        strategy (TruncationStrategy): 'first' (lấy các từ đầu) hoặc 'head_tail'
            (ghép nửa đầu và nửa cuối văn bản). Mặc định là 'first'.

    Returns:
        tuple[list[int], int]:
            - list[int]: Danh sách chỉ số có độ dài chính xác bằng `max_length`.
            - int: Độ dài thực tế (đã bị giới hạn bởi `max_length`) trước khi pad.
    """
    encoded, length, _ = encode_with_audit(text, vocabulary, max_length, strategy)
    return encoded, length


def encode_with_audit(
    text: str,
    vocabulary: Vocabulary,
    max_length: int,
    strategy: TruncationStrategy = "first",
) -> tuple[list[int], int, dict[str, Any]]:
    """Mã hóa văn bản kèm báo cáo kiểm tra độ tin cậy (token audit metadata).

    Returns:
        tuple[list[int], int, dict[str, Any]]:
            - encoded: Chuỗi chỉ số tokens [max_length].
            - length: Độ dài thực tế chuỗi dùng trong pack_padded_sequence.
            - audit: Dictionary chứa 'input_tokens', 'used_tokens', 'is_truncated',
                     'oov_count', 'oov_rate'.
    """
    tokens = tokenize(text)
    total_tokens = len(tokens)
    oov_count, oov_rate = vocabulary.compute_oov_stats(tokens)

    if not tokens:
        selected_tokens: list[str] = []
    elif len(tokens) <= max_length:
        selected_tokens = tokens
    elif strategy == "head_tail":
        head_len = max_length // 2
        tail_len = max_length - head_len
        selected_tokens = tokens[:head_len] + tokens[-tail_len:]
    else:  # strategy == "first"
        selected_tokens = tokens[:max_length]

    encoded = vocabulary.encode(selected_tokens)
    length = max(1, len(encoded))
    if not encoded:
        encoded = [vocabulary.unknown_index]

    is_truncated = total_tokens > max_length
    used_tokens = len(selected_tokens)

    # Thêm padding về độ dài chuẩn max_length
    if len(encoded) < max_length:
        encoded.extend([vocabulary.pad_index] * (max_length - len(encoded)))

    audit = {
        "input_tokens": total_tokens,
        "used_tokens": used_tokens,
        "is_truncated": is_truncated,
        "oov_count": oov_count,
        "oov_rate": round(oov_rate, 4),
    }

    return encoded, length, audit
