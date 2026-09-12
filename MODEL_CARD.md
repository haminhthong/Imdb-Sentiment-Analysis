# Model Card — CineSentiment

## 1. Task (Nhiệm Vụ)
Phân loại cảm xúc nhị phân của các văn bản đánh giá phim (Movie Reviews) bằng tiếng Anh thành 2 lớp:
- `Positive` (Tích cực, điểm số >= 7/10)
- `Negative` (Tiêu cực, điểm số <= 4/10)

## 2. Dataset (Tập Dữ Liệu)
- **Bộ dữ liệu:** Large Movie Review Dataset (IMDB v1.0, Maas et al., ACL 2011).
- **Quy mô:** 50.000 đánh giá phân cực cân bằng (25.000 Train, 25.000 Test).
- **Phân chia phát triển:** Tập Train được chia stratified thành Train 80% (20.000), Validation 10% (2.500) và Calibration 10% (2.500). Tập Test hoàn toàn độc lập.
- **Từ điển:** Được xây dựng duy nhất từ tập Train để ngăn ngừa rò rỉ dữ liệu (Train-only vocabulary).

## 3. Models (Kiến Trúc Mô Hình)
Dự án đối chiếu hai mô hình tiêu biểu:
1. **Baseline (Sparse-Text):** TF-IDF (unigram + bigram, min_df=2, sublinear TF, 50.000 features) kết hợp hồi quy Logistic Regression. Cho phép trích xuất các n-gram có trọng số đóng góp mạnh nhất.
2. **Deep Learning:** Bidirectional LSTM (BiLSTM) với Embedding Layer (128d), 2 lớp LSTM 2 chiều (hidden 128d), áp dụng `pack_padded_sequence` để tối ưu tính toán chuỗi độ dài biến thiên, kết hợp Dropout (0.4) và Linear Layer.

## 4. Calibration (Hiệu Chuẩn Xác Suất)
Mô hình BiLSTM được hiệu chuẩn bằng **Temperature Scaling** ($T > 0$) trên Calibration logits thông qua tối ưu hóa NLL (Negative Log-Likelihood) bằng thuật toán L-BFGS:
$$\hat{p} = \sigma\left(\frac{z}{T}\right)$$
Giúp đưa xác suất dự đoán về gần với độ chính xác thực tế, giảm thiểu hiện tượng overconfidence của mạng nơ-ron.

## 5. Metrics (Chỉ Số Đánh Giá)
- **Phân loại:** Accuracy, Macro-F1, ROC-AUC, PR-AUC.
- **Độ tin cậy xác suất:** Brier Score, Expected Calibration Error (ECE qua 10 bins), và biểu đồ Reliability Diagram.
- **Hiệu năng hệ thống:** Thời gian suy luận trung bình trên CPU (ms/sample) và số lượng tham số.

## 6. Intended Use (Mục Đích Sử Dụng)
- Phân tích cảm xúc tổng quan của bài đánh giá phim tiếng Anh.
- Tham khảo kiến trúc chuẩn mực: Baseline so sánh, kiểm soát rò rỉ từ vựng, hiệu chuẩn xác suất và phân tích lỗi.

## 7. Limitations (Hạn Chế)
- **Ngôn ngữ:** Chỉ hỗ trợ tiếng Anh; không phù hợp với các ngôn ngữ khác hoặc văn bản ngoài miền đánh giá điện ảnh.
- **Hiện tượng ngôn ngữ phức tạp:** Có thể hiểu sai các câu mỉa mai (sarcasm), phủ định kép hoặc cảm xúc hỗn hợp/đảo chiều ở đoạn kết.
- **Độ dài và từ ngoài từ điển (OOV):** Các bài đánh giá dài (>256 tokens) phải cắt ngắn (head-tail truncation); từ vựng mới không có trong tập Train sẽ chuyển thành `<UNK>`.
