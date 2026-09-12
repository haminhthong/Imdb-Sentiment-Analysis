"""Xử lý văn bản và quản lý từ điển cho bài toán phân loại cảm xúc IMDB.

Các tính năng chính:
1. Chuẩn hóa văn bản: giải mã thực thể HTML, loại bỏ thẻ HTML, chuyển chữ thường.
2. Tokenizer: Sử dụng biểu thức chính quy bảo tồn các từ phủ định quan trọng
   như "don't", "isn't", "can't", "won't".
3. Từ điển (Vocabulary): Quản lý ánh xạ token sang index, hỗ trợ <PAD> và <UNK>.
4. Mã hóa & Đệm (Padding/Truncation): Hỗ trợ cắt chuỗi theo 'first' hoặc 'head_tail'.
"""

import html
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Regex giữ lại chữ cái, chữ số và dấu nháy trong các từ phủ định (ví dụ: "don't", "isn't")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

COMMON_ENGLISH_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "was",
    "are",
    "were",
    "it",
    "this",
    "that",
    "movie",
    "film",
    "actor",
    "actors",
    "story",
    "plot",
    "good",
    "bad",
    "great",
    "one",
    "all",
    "see",
    "watch",
    "time",
    "very",
    "not",
}

TruncationStrategy = Literal["first", "head_tail"]


def tokenize(text: str) -> list[str]:
    """Chuẩn hóa văn bản, xóa thẻ HTML, chuyển chữ thường và tách token.

    Giữ nguyên các dạng viết tắt phủ định như "don't", "isn't" để phục vụ
    phân tích cảm xúc.
    """
    if not isinstance(text, str):
        return []
    normalized = html.unescape(HTML_TAG_PATTERN.sub(" ", text)).lower()
    return TOKEN_PATTERN.findall(normalized)


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản thành chuỗi các token cách nhau bởi khoảng trắng.

    Dùng để phát hiện các câu trùng lặp nội dung dù khác nhau về thẻ HTML,
    khoảng trắng thừa hoặc chữ hoa/thường.
    """
    return " ".join(tokenize(text))


def detect_language_warning(text: str) -> list[str]:
    """Gắn cờ cảnh báo nếu văn bản đầu vào có dấu hiệu ngoài miền tiếng Anh."""
    if not isinstance(text, str) or not text.strip():
        return []

    tokens = tokenize(text)
    if not tokens:
        return []

    total_chars = len(text)
    non_ascii_chars = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = non_ascii_chars / total_chars if total_chars > 0 else 0.0

    has_english_token = any(token in COMMON_ENGLISH_WORDS for token in tokens)
    if (len(tokens) >= 5 and not has_english_token) or non_ascii_ratio > 0.25:
        return ["OUT_OF_DOMAIN_LANGUAGE_HEURISTIC"]

    return []


@dataclass(frozen=True)
class Vocabulary:
    """Quản lý ánh xạ hai chiều giữa từ (token) và chỉ số số nguyên (index)."""

    token_to_index: dict[str, int]

    @property
    def pad_index(self) -> int:
        return self.token_to_index[PAD_TOKEN]

    @property
    def unk_index(self) -> int:
        return self.token_to_index[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_index)

    def to_dict(self) -> dict[str, Any]:
        return {"token_to_index": self.token_to_index}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vocabulary":
        return cls(token_to_index=data["token_to_index"])


def build_vocabulary(
    texts: Iterable[str],
    min_frequency: int = 5,
    max_vocabulary_size: int = 50_000,
) -> Vocabulary:
    """Xây dựng bộ từ vựng từ tập văn bản (chỉ nên gọi trên tập Train)."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    token_to_index: dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    sorted_tokens = [
        token
        for token, count in counter.most_common()
        if count >= min_frequency and token not in token_to_index
    ]

    available_slots = max_vocabulary_size - len(token_to_index)
    for token in sorted_tokens[:available_slots]:
        token_to_index[token] = len(token_to_index)

    return Vocabulary(token_to_index=token_to_index)


def truncate_tokens(
    tokens: list[str], max_length: int, strategy: TruncationStrategy = "head_tail"
) -> list[str]:
    """Cắt ngắn danh sách token nếu vượt quá max_length."""
    if len(tokens) <= max_length:
        return tokens

    if strategy == "head_tail":
        head_length = max_length // 2
        tail_length = max_length - head_length
        return tokens[:head_length] + tokens[-tail_length:]

    return tokens[:max_length]


def encode_and_pad(
    text: str,
    vocabulary: Vocabulary,
    max_length: int,
    strategy: TruncationStrategy = "head_tail",
) -> tuple[list[int], int]:
    """Mã hóa văn bản thành danh sách chỉ số và đệm cố định đúng max_length."""
    raw_tokens = tokenize(text)
    used_tokens = truncate_tokens(raw_tokens, max_length, strategy=strategy)
    actual_length = len(used_tokens)

    encoded = [vocabulary.token_to_index.get(token, vocabulary.unk_index) for token in used_tokens]

    if len(encoded) < max_length:
        encoded.extend([vocabulary.pad_index] * (max_length - len(encoded)))

    return encoded, actual_length


def encode_with_audit(
    text: str,
    vocabulary: Vocabulary,
    max_length: int,
    strategy: TruncationStrategy = "head_tail",
) -> tuple[list[int], int, dict[str, Any]]:
    """Mã hóa văn bản và trả về báo cáo kiểm toán thông tin token."""
    raw_tokens = tokenize(text)
    total_tokens = len(raw_tokens)
    used_tokens = truncate_tokens(raw_tokens, max_length, strategy=strategy)
    actual_length = len(used_tokens)

    encoded = [vocabulary.token_to_index.get(token, vocabulary.unk_index) for token in used_tokens]

    input_oov_count = sum(1 for t in raw_tokens if t not in vocabulary.token_to_index)
    used_oov_count = sum(1 for t in used_tokens if t not in vocabulary.token_to_index)

    input_oov_rate = input_oov_count / total_tokens if total_tokens > 0 else 0.0
    used_oov_rate = used_oov_count / actual_length if actual_length > 0 else 0.0

    if len(encoded) < max_length:
        encoded.extend([vocabulary.pad_index] * (max_length - len(encoded)))

    audit = {
        "input_tokens": total_tokens,
        "used_tokens": actual_length,
        "is_truncated": total_tokens > max_length,
        "input_oov_rate": round(input_oov_rate, 4),
        "used_oov_rate": round(used_oov_rate, 4),
    }

    return encoded, actual_length, audit
