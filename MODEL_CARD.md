# Model Card — CineSentiment Platform

## 1. Tổng Quan Mô Hình (Model Overview)
- **Tên mô hình:** CineSentiment — Calibrated BiLSTM Sentiment Service.
- **Kiến trúc production:** TF-IDF + Logistic Regression là baseline; BiLSTM là target model. LSTM/GRU chỉ nằm trong legacy experiments.
- **Cấu hình thực thi:** Release bundle `artifacts/releases/v1.0.0/model.pt` với `word-regex-v2` tokenizer contract.
- **Nhiệm vụ:** Phân loại cảm xúc nhị phân của câu đánh giá phim tiếng Anh thành `Positive` hoặc `Negative`.

## 2. Mục Đích Sử Dụng (Intended Use)
### Phù hợp:
- Minh họa chuẩn mực quy trình NLP end-to-end: anti-leakage, calibration, locked test, serving.
- Phân tích xu hướng cảm xúc tổng quan của các văn bản đánh giá phim tiếng Anh.
- Portfolio kỹ thuật chuyên sâu cho vị trí AI / NLP / MLOps Engineer.

### Không phù hợp:
- Tự động ra quyết định ảnh hưởng trực tiếp đến người dùng mà không có con người giám sát (Human-in-the-loop).
- Xử lý ngôn ngữ ngoài tiếng Anh hoặc các văn bản ngoài miền đánh giá điện ảnh.
- Sử dụng trực tiếp điểm sigmoid thô mà không thông qua Temperature Scaling hiệu chuẩn.

## 3. Dữ Liệu Huấn Luyện & Giao Thức Chống Rò Rỉ (Data & Anti-Leakage)
- **Nguồn dữ liệu:** Large Movie Review Dataset (IMDB v1.0, 50.000 mẫu).
- **Kiểm soát rò rỉ:**
  - Băm SHA256 văn bản gốc (Raw Hash).
  - Băm SHA256 văn bản chuẩn hóa canonical tokenization (`normalized_exact_hash`); đây không phải semantic near-duplicate.
  - Official Test chỉ được audit overlap với Train; overlap làm pipeline fail, không sửa benchmark.
  - Train/Validation/Calibration là 20.000/2.500/2.500; từ điển development fit **CHỈ** trên Train.

## 4. Hiệu Chuẩn Xác Suất & Chính Sách Quyết Định (Probability Calibration & Decision Policy)
- **Temperature Scaling:** Mạng nơ-ron được tối ưu hóa hệ số $T > 0$ trên Calibration Logits để tránh overconfidence:
  $$\hat{p} = \sigma\left(\frac{z}{T}\right)$$
- **Đo lường hiệu chuẩn:** Brier Score và Expected Calibration Error (ECE qua 10 bins).
- **Chính sách selective classification:** nhãn dùng threshold 0.5; `confidence = max(p, 1-p)`. Calibration chọn `confidence_threshold = τ`; nếu confidence < τ thì `review_required`.

## 5. Giao Thức Đánh Giá Độc Lập (Evaluation Protocol)
- **Loại bỏ Test Peeking:** Quá trình ứng viên chỉ lưu `validation_metrics.json`. `compare_models.py` tổng hợp Development Leaderboard từ Validation để chọn Champion.
- **Locked Final Test:** Chỉ có mô hình Champion được mở tập Test chính thức đúng **1 LẦN DUY NHẤT** thông qua `evaluate_final.py`.

## 6. Hạn Chế & Giảm Thiểu Rủi Ro (Limitations & Mitigations)
- **Cấu trúc ngôn ngữ phức tạp:** Có thể hiểu nhầm câu mỉa mai (sarcasm) hoặc câu có cảm xúc hỗn hợp/đảo chiều ở đoạn kết.
- **Ranh giới miền:** Gắn cờ `OUT_OF_DOMAIN_LANGUAGE_HEURISTIC`; đây chỉ là heuristic, không phải language detector.
- **Cắt ngắn chuỗi:** Tự động phát hiện và đính kèm cờ `TRUNCATED_INPUT` nếu câu vượt quá `max_length`.
- **Độ tin cậy từ vựng:** Cung cấp cảnh báo `HIGH_OOV_WARNING` khi tỷ lệ từ ngoài từ điển vượt quá 20%.

## 7. Khả Năng Tái Lập (Reproducibility)
- Release checkpoint tuân thủ **Artifact Schema Version 3**, tích hợp: `model_state`, `vocabulary`, `config`, `temperature`, `confidence_threshold`, tokenizer contract, vocabulary hash, source/split hashes, version và final-fit metadata.
- Cố định seed ngẫu nhiên cho Python, NumPy và PyTorch cuDNN deterministic.
