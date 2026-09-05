# Dataset Card — Large Movie Review Dataset (IMDB)

## 1. Nguồn Dữ Liệu
Dự án sử dụng **Large Movie Review Dataset v1.0** của Maas và cộng sự (Stanford AI):
- Trang chủ chính thức: [Stanford AI — Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- Bài báo nguồn: [Learning Word Vectors for Sentiment Analysis (ACL-HLT 2011)](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
- Kích thước: 50.000 đánh giá phim có nhãn, chia thành 25.000 mẫu train và 25.000 mẫu test cân bằng 50/50 giữa positive và negative.

## 2. Cấu Trúc Schema
| Cột | Kiểu | Mô tả |
|---|---|---|
| `text` | string | Nội dung câu đánh giá phim bằng tiếng Anh |
| `label` | float32 / integer | Nhãn nhị phân: `0` = Negative (<= 4/10), `1` = Positive (>= 7/10) |

*Ghi chú: Dữ liệu không chứa các đánh giá trung tính (Neutral) nên bài toán được định nghĩa chuẩn là phân loại nhị phân (Binary Classification).*

## 3. Kiểm Toán Dữ Liệu & Chống Rò Rỉ Đa Tầng (Data Audit & Anti-Leakage)

### A. Kiểm tra mâu thuẫn nhãn (Conflicting Labels)
- Bắt buộc kiểm tra cả văn bản gốc lẫn văn bản chuẩn hóa (lowercase + strip tags + canonical tokenization).
- Nếu phát hiện 2 bản ghi có nội dung tương đương nhưng mang nhãn mâu thuẫn (ví dụ một bản ghi nhãn 0, một bản ghi nhãn 1), pipeline lập tức báo lỗi `ValueError` để kỹ sư xử lý nguồn cấp dữ liệu, không âm thầm giữ bản ghi đầu tiên.

### B. Loại bỏ trùng lặp chuẩn hóa (Normalized Dedup)
- Ngoài việc loại bỏ exact duplicate (`drop_duplicates`), pipeline áp dụng `compute_normalized_text_hash` (SHA256 của canonical tokens).
- Điều này loại bỏ hoàn toàn các mẫu trùng lặp biến thể hoa/thường, thẻ HTML thừa `<br />` hoặc dấu câu ngoại lai.

### C. Triệt tiêu rò rỉ Train-Test Overlap
- Mọi mẫu trong tập Test có raw text hoặc normalized hash xuất hiện trong tập Train nguồn đều bị loại bỏ hoàn toàn trước khi tiến hành chia tập và huấn luyện.
- Đảm bảo tập Test độc lập 100% về mặt phân phối ngữ nghĩa.

## 4. Hợp Đồng Từ Điển Chỉ Trên Train (Train-Only Vocabulary Contract)
- Tập Train được chia tách trước (Stratified Split 80/20) thành Train Split và Validation Split.
- `Vocabulary` và `TfidfVectorizer` **CHỈ** được fit trên phần Train Split.
- Tỷ lệ token ngoài từ điển (OOV Rate) được theo dõi chặt chẽ:
  - Train OOV: ~0.0%
  - Validation OOV: Thống kê định lượng
  - Test OOV: Thống kê định lượng

## 5. Thống Kê Phân Phối Độ Dài Chuỗi (Token Length Distribution)
- Độ dài trung vị ($p_{50}$): ~170 tokens
- Phân vị $p_{90}$: ~385 tokens
- Phân vị $p_{95}$: ~510 tokens
- Độ dài cực đại ($\max$): > 2.000 tokens
- Tỷ lệ cắt chuỗi ($\text{Truncation Rate}$ tại `max_length=256`): ~20-25% mẫu.
- Để giảm thiểu rủi ro mất mát thông tin kết luận ở cuối câu đánh giá, pipeline hỗ trợ chiến lược cắt chuỗi `head_tail` (kết hợp nửa đầu và nửa cuối văn bản).

## 6. Trích Dẫn Nguồn
```bibtex
@inproceedings{maas2011learning,
  title={Learning Word Vectors for Sentiment Analysis},
  author={Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and
          Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  booktitle={Proceedings of ACL-HLT 2011},
  pages={142--150},
  year={2011}
}
```
