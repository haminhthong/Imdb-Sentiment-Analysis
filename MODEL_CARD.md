# Model Card — CineSentiment Platform

## 1. Tổng Quan Mô Hình (Model Overview)
- **Tên mô hình:** CineSentiment Model (Mạng nơ-ron hồi quy Recurrent Neural Network & TF-IDF Logistic Baseline).
- **Kiến trúc hỗ trợ:** TF-IDF + Logistic Regression, LSTM, GRU, Bidirectional LSTM (BiLSTM).
- **Cấu hình thực thi mặc định:** BiLSTM (`artifacts/bilstm/model.pt`) với `word-regex-v1` tokenizer contract.
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
- **Kiểm soát rò rỉ 2 tầng:**
  - Băm SHA256 văn bản gốc (Raw Hash).
  - Băm SHA256 văn bản chuẩn hóa canonical tokenization (Normalized Text Hash) để loại bỏ hoàn toàn near-duplicates có biến thể hoa/thường, thẻ HTML và khoảng trắng thừa.
  - Loại bỏ triệt để các mẫu Test trùng lặp với Train trước khi chia tập.
  - Từ điển (`Vocabulary`) được fit **CHỈ** trên tập Train sau khi phân chia Stratified Split.

## 4. Hiệu Chuẩn Xác Suất & Chính Sách Quyết Định (Probability Calibration & Decision Policy)
- **Temperature Scaling:** Mạng nơ-ron được tối ưu hóa hệ số $T > 0$ trên tập Validation Logits để tránh overconfidence:
  $$\hat{p} = \sigma\left(\frac{z}{T}\right)$$
- **Đo lường hiệu chuẩn:** Brier Score và Expected Calibration Error (ECE qua 10 bins).
- **Chính sách vùng bất định:**
  - $\hat{p} > 0.60 \implies$ `Positive` (Decision: `accepted`)
  - $\hat{p} < 0.40 \implies$ `Negative` (Decision: `accepted`)
  - $0.40 \le \hat{p} \le 0.60 \implies$ `Uncertain` (Decision: `review_required` — gắn cờ cần con người rà soát).

## 5. Giao Thức Đánh Giá Độc Lập (Evaluation Protocol)
- **Loại bỏ Test Peeking:** Quá trình huấn luyện ứng viên chỉ lưu `validation_metrics.json`. Script `compare_models.py` tổng hợp Development Leaderboard từ Validation để chọn duy nhất một **Champion**.
- **Locked Final Test:** Chỉ có mô hình Champion được mở tập Test chính thức đúng **1 LẦN DUY NHẤT** thông qua `evaluate_final.py`.

## 6. Hạn Chế & Giảm Thiểu Rủi Ro (Limitations & Mitigations)
- **Cấu trúc ngôn ngữ phức tạp:** Có thể hiểu nhầm câu mỉa mai (sarcasm) hoặc câu có cảm xúc hỗn hợp/đảo chiều ở đoạn kết.
- **Ranh giới miền:** Gắn cờ cảnh báo `NON_ENGLISH_WARNING` khi văn bản đầu vào có tỷ lệ ký tự non-ASCII cao hoặc không chứa từ vựng tiếng Anh thông dụng.
- **Cắt ngắn chuỗi:** Tự động phát hiện và đính kèm cờ `TRUNCATED_INPUT` nếu câu vượt quá `max_length`.
- **Độ tin cậy từ vựng:** Cung cấp cảnh báo `HIGH_OOV_WARNING` khi tỷ lệ từ ngoài từ điển vượt quá 20%.

## 7. Khả Năng Tái Lập (Reproducibility)
- Checkpoint tuân thủ **Artifact Schema Version 2**, tích hợp: `model_state`, `vocabulary`, `config`, `temperature`, `decision_threshold`, `tokenizer_version`, `vocabulary_hash`, `training_data_hash`.
- Cố định seed ngẫu nhiên cho Python, NumPy và PyTorch cuDNN deterministic.
