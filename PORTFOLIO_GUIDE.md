# Hướng Dẫn Đưa CineSentiment Platform Vào CV, LinkedIn & Phỏng Vấn

## 1. Định Vị Dự Án Chuẩn Mực Trong CV (Portfolio Positioning)

### Tên dự án gợi ý trong CV:
**CineSentiment — Leakage-Safe Sentiment Intelligence & NLP Benchmarking Platform**

### Mô tả ngắn 1 câu (GitHub Bio / Resume Bullet):
> *An end-to-end English movie-review sentiment platform comparing TF-IDF Logistic Regression with LSTM, GRU, and BiLSTM architectures under leakage-safe validation, calibrated decision policies, reproducible artifacts, multi-slice error analysis, and production-oriented FastAPI/Streamlit serving.*

### Pipeline ngắn gọn đưa vào mục Skills/Experience:
```text
IMDB Reviews → Data QA & Anti-Leakage → Train-Only Text Contract → TF-IDF / LSTM / GRU / BiLSTM → Validation Model Selection → Temperature Scaling Calibration → Locked Test Once → Error Analysis → Versioned Artifact → FastAPI / Streamlit
```

---

## 2. Checklist Kỹ Thuật Trước Khi Phỏng Vấn
- [x] Huấn luyện TF-IDF baseline (`python baseline.py`) để làm thước đo tham chiếu hiệu năng/độ trễ.
- [x] Huấn luyện các mô hình RNN (`python train.py --model bilstm`, `gru`, `lstm`).
- [x] Chạy `python compare_models.py` để tạo `artifacts/development_leaderboard.md` dựa hoàn toàn trên Validation.
- [x] Chạy `python evaluate_final.py` để mở Official Test duy nhất một lần cho Champion đã đóng băng.
- [x] Chạy `python analyze_errors.py` để phân tích các lát cắt lỗi theo độ dài, OOV và taxonomy ngữ nghĩa.
- [x] Chạy toàn bộ test suite (`python -m pytest`) xác nhận 100% tests vượt qua.
- [x] Khởi chạy Streamlit (`streamlit run app.py`) và chụp ảnh màn hình giao diện 3 tab.
- [x] Kiểm tra tài liệu Swagger UI của FastAPI tại `http://localhost:8000/docs`.

---

## 3. Bộ Câu Hỏi Phỏng Vấn Chuyên Sâu & Câu Trả Lời Chuẩn Mực

### Q1: Tại sao dự án của bạn lại tách biệt hoàn toàn giữa Validation Leaderboard và Locked Official Test?
- **Trả lời:** Trong quy trình machine learning chuẩn mực, nếu ta đánh giá nhiều kiến trúc (TF-IDF, LSTM, GRU, BiLSTM) trên tập Test rồi chọn mô hình có điểm Test cao nhất, tập Test thực chất đã bị biến thành tập chọn mô hình (Model Selection Set). Điều này gây ra hiện tượng **Test Peeking (Data Snooping Bias)**, khiến chỉ số công bố bị thổi phồng và không phản ánh đúng năng lực tổng quát hóa.
- Ở CineSentiment, tất cả các mô hình ứng viên chỉ được so sánh trên **Validation Set** thông qua `compare_models.py`. Chỉ duy nhất mô hình **Champion** được chọn mới được mở tập Official Test đúng **1 lần duy nhất** thông qua `evaluate_final.py` sau khi đã đóng băng toàn bộ tham số.

### Q2: Tại sao kiểm tra trùng lặp bằng string thông thường là chưa đủ để chống Data Leakage?
- **Trả lời:** Kiểm tra chuỗi chính xác (`df.drop_duplicates()`) sẽ bỏ lọt các biến thể như chữ hoa/thường (`"GREAT"` vs `"great"`), thẻ HTML dư thừa (`<br />`), hoặc khoảng trắng thừa. Những mẫu này bản chất mang ngữ nghĩa giống hệt nhau.
- CineSentiment triển khai cơ chế chống rò rỉ 2 tầng: **Exact Raw Hash** và **Normalized Text Hash** (HTML unescape + strip tags + lowercase + regex tokenize + SHA256). Mọi mẫu Test có normalized hash trùng với Train đều bị loại bỏ triệt để.

### Q3: Tại sao bạn không gọi output của Sigmoid là "Xác suất" (Probability) mà phải áp dụng Temperature Scaling?
- **Trả lời:** Điểm ra của hàm Sigmoid $\sigma(z)$ trên mạng nơ-ron sâu hiện đại thường bị **tự tin thái quá (Overconfident)** do mô hình được tối ưu bằng cross-entropy loss đẩy logits ra xa 0. Mô hình có thể dự đoán $0.99$ nhưng thực tế sai số lên đến $10\%$.
- CineSentiment giải quyết triệt để vấn đề này bằng cách học tham số nhiệt độ $T > 0$ (**Temperature Scaling**) trên tập Validation Logits, đồng thời đo lường chất lượng hiệu chuẩn qua **Brier Score** và **Expected Calibration Error (ECE)**.

### Q4: Sự khác biệt và bài toán đánh đổi giữa TF-IDF Logistic Regression và BiLSTM là gì?
- **Trả lời:**
  - **TF-IDF + Logistic Regression:** Rất nhẹ (~2MB), thời gian huấn luyện tính bằng giây, độ trễ CPU cực thấp (<1ms/sample), dễ giải thích qua trọng số n-gram, và trên IMDB đạt độ chính xác rất cao (~89.8%).
  - **BiLSTM:** Nắm bắt được thứ tự từ và sự phụ thuộc ngữ cảnh 2 chiều, nhưng số lượng tham số lớn hơn rất nhiều (~6.7 triệu tham số), latency cao hơn (~10-20ms) và cần nhiều tài nguyên tính toán hơn.
  - **Tư duy AI Engineer:** Nếu BiLSTM chỉ nhỉnh hơn baseline một biên độ rất nhỏ (<0.5% F1) trong khi latency và kích thước gấp hàng chục lần, thì việc giữ baseline hoặc chọn mô hình nhẹ hơn (Occam's Razor) là quyết định kỹ thuật đúng đắn cho môi trường production.

### Q5: Dự án xử lý như thế nào đối với các câu đánh giá có độ dài lớn vượt quá `max_length`?
- **Trả lời:** Đánh giá phim thường có cấu trúc dài (đoạn đầu kể nội dung, đoạn cuối mới chốt kết luận). Nếu cắt chuỗi đơn giản theo kiểu lấy 256 từ đầu (`first`), mô hình sẽ bỏ lỡ câu kết luận quan trọng nhất.
- CineSentiment đo lường phân phối độ dài ($p_{50}, p_{90}, p_{95}, \max$), cung cấp cảnh báo `TRUNCATED_INPUT` trong API, và hỗ trợ chiến lược cắt chuỗi `head_tail` (ghép nửa đầu và nửa cuối của văn bản) để bảo toàn tối đa thông điệp cảm xúc.
