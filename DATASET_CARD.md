# Dataset Card — Large Movie Review Dataset (IMDB)

## Nguồn dữ liệu

Dự án sử dụng **Large Movie Review Dataset v1.0** của Maas và cộng sự. Bộ dữ liệu gồm 50.000 đánh giá phim có nhãn, chia sẵn thành 25.000 mẫu train và 25.000 mẫu test; mỗi tập cân bằng giữa positive và negative.

- Trang dữ liệu chính thức: [Stanford AI — Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- Bài báo nguồn: [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)

## Schema

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `text` | string | Nội dung đánh giá phim tiếng Anh |
| `label` | integer | `0` = Negative, `1` = Positive |

Theo bài báo nguồn, negative có điểm không quá 4/10 và positive có điểm ít nhất 7/10. Neutral không nằm trong dữ liệu nên dự án chỉ giải quyết phân loại nhị phân.

## Kiểm tra dữ liệu trong repository

| Kiểm tra | Train nguồn | Test nguồn |
|---|---:|---:|
| Số mẫu | 25.000 | 25.000 |
| Negative | 12.500 | 12.500 |
| Positive | 12.500 | 12.500 |
| Missing | 0 | 0 |
| Duplicate text nội bộ | 96 | 199 |

Có 123 nội dung xuất hiện ở cả train và test, không có trường hợp cùng nội dung nhưng khác nhãn. Pipeline loại phần giao trước khi đánh giá, còn lại 24.678 mẫu test độc lập.

## Chống rò rỉ dữ liệu

1. Loại missing và duplicate trong từng file.
2. Dừng pipeline nếu cùng nội dung có nhãn mâu thuẫn.
3. Loại khỏi test mọi nội dung đã xuất hiện trong train nguồn.
4. Chia train/validation bằng stratified split.
5. Chỉ fit vocabulary hoặc TF-IDF trên train split.
6. Chỉ dùng validation để lựa chọn mô hình; test dành cho đánh giá cuối.

## Hạn chế và quyền sử dụng

- Chỉ có hai cực cảm xúc, không có neutral hoặc mixed sentiment.
- Dữ liệu thuộc miền đánh giá phim tiếng Anh và có thể không tổng quát sang miền khác.
- Review có thể chứa ngôn ngữ thô tục, định kiến hoặc thông tin đã được người dùng đăng công khai.
- Cần kiểm tra quyền phân phối lại file CSV trước khi public repository; phương án an toàn là cung cấp đường dẫn/script tải từ nguồn chính thức.

## Trích dẫn

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

