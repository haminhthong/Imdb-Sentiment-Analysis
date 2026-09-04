# Model Card — CineSentiment AI

## Tổng quan

CineSentiment AI phân loại đánh giá phim tiếng Anh thành `Positive` hoặc `Negative`. Dự án hỗ trợ LSTM, GRU và Bidirectional LSTM; checkpoint triển khai mặc định là `artifacts/bilstm/model.pt`.

## Mục đích sử dụng

Phù hợp cho:

- minh họa quy trình NLP end-to-end;
- phân tích xu hướng cảm xúc tổng quan của đánh giá phim tiếng Anh;
- học tập, portfolio và thử nghiệm kiến trúc RNN.

Không phù hợp cho:

- quyết định có ảnh hưởng lớn đến con người;
- kiểm duyệt tự động không có con người giám sát;
- phân tích ngôn ngữ ngoài tiếng Anh hoặc văn bản không thuộc miền đánh giá phim;
- diễn giải xác suất đầu ra như xác suất đã được hiệu chỉnh.

## Dữ liệu

Dữ liệu đầu vào gồm 50.000 đánh giá phim IMDB, cân bằng giữa hai nhãn. Pipeline loại nội dung trùng trong từng tập và loại mẫu test đã xuất hiện trong train. Vocabulary chỉ được xây từ phần train sau khi chia validation.

Người công bố repository cần bổ sung liên kết nguồn dữ liệu và tuân thủ giấy phép/điều khoản của bộ dữ liệu được sử dụng.

## Tiền xử lý

- Loại thẻ HTML và giải mã HTML entities.
- Chuyển chữ thường.
- Tách token bằng biểu thức chính quy.
- Giữ dấu nháy trong các từ phủ định như `don't`.
- Cắt hoặc padding về độ dài cấu hình; mô hình dùng độ dài thật để bỏ qua padding.

## Đánh giá

Các chỉ số được lưu trong `metrics.json`:

- binary cross-entropy loss;
- accuracy;
- precision, recall và F1 cho từng lớp;
- macro/weighted average;
- confusion matrix.

Chưa có checkpoint/kết quả huấn luyện trong repository nên model card không công bố số liệu chưa được đo. Sau khi huấn luyện đủ ba mô hình, chạy `python compare_models.py` và cập nhật bảng kết quả thật tại đây.

## Hạn chế và rủi ro

- Có thể hiểu sai câu mỉa mai, phủ định phức tạp hoặc đánh giá chứa cả hai chiều cảm xúc.
- Từ ngoài vocabulary được ánh xạ thành `<UNK>`.
- Dữ liệu IMDB có thể không đại diện cho ngôn ngữ hiện đại hoặc miền sản phẩm khác.
- Độ tin cậy đầu ra chưa được calibration.
- Mô hình có thể học thiên lệch tồn tại trong dữ liệu huấn luyện.

## Giảm thiểu

- Hiển thị cảnh báo trong giao diện demo.
- Không tự động đưa ra quyết định quan trọng.
- Báo cáo chỉ số theo từng lớp thay vì chỉ accuracy.
- Giữ test độc lập và không dùng test để chọn siêu tham số.
- Đề xuất phân tích lỗi và calibration trước khi dùng ngoài mục đích demo.

## Tái lập

Checkpoint chứa trọng số, vocabulary và toàn bộ `ExperimentConfig`. Seed được đặt cho Python, NumPy và PyTorch; cuDNN deterministic được bật. Kết quả vẫn có thể lệch nhẹ giữa phiên bản thư viện và phần cứng.

