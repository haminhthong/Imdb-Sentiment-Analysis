# Hướng dẫn đưa CineSentiment AI vào CV và GitHub

## Checklist trước khi công khai

- [ ] Huấn luyện cả LSTM, GRU và BiLSTM với cùng cách chia dữ liệu.
- [ ] Chạy `python compare_models.py` và đưa bảng kết quả thật vào README.
- [ ] Chụp ảnh giao diện Streamlit sau khi có checkpoint.
- [ ] Ghi thời gian huấn luyện, phần cứng và phiên bản Python/PyTorch.
- [ ] Thêm liên kết nguồn dữ liệu và kiểm tra quyền phân phối hai file CSV.
- [ ] Đưa file dữ liệu lớn ra khỏi Git; cung cấp script hoặc đường dẫn tải dữ liệu.
- [ ] Deploy demo/API và thêm liên kết trực tiếp vào đầu README.
- [ ] Bật GitHub Actions và xác nhận pipeline màu xanh.
- [ ] Thêm thông tin tác giả, LinkedIn và email chuyên nghiệp.

## Cách trình bày kết quả

Không chọn mô hình chỉ theo accuracy. Bảng portfolio nên có:

| Model | Parameters | Validation F1 | Test F1 | Test Accuracy | Inference latency |
|---|---:|---:|---:|---:|---:|
| LSTM | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| GRU | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| BiLSTM | Chưa đo | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

Chỉ điền số liệu do chính pipeline tạo ra. Ghi rõ thiết bị và batch size khi đo latency.

## Câu hỏi phỏng vấn nên chuẩn bị

1. Vì sao vocabulary chỉ được xây từ train?
2. Vì sao packed sequence tốt hơn đưa toàn bộ padding qua RNN?
3. GRU và LSTM khác nhau thế nào về tham số và tốc độ?
4. BiLSTM có lợi gì và khi nào không phù hợp?
5. Tại sao không tối ưu hyperparameter trên test?
6. Accuracy có đủ cho bài toán phân loại không?
7. Làm thế nào phát hiện data leakage?
8. Nếu chuyển sang production, cần thêm monitoring và drift detection thế nào?

## Gợi ý mô tả ngắn trên GitHub

`End-to-end IMDB sentiment analysis with PyTorch LSTM/GRU/BiLSTM, leakage-safe evaluation, FastAPI, Streamlit, Docker and CI.`

