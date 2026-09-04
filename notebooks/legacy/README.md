# Legacy notebooks

Ba notebook trong thư mục này là lịch sử thử nghiệm ban đầu cho LSTM, GRU và BiLSTM.

Chúng không còn là entry point chính vì chứa pipeline lặp và một số lỗi đánh giá đã được sửa trong package `sentiment/`. Để huấn luyện hoặc so sánh mô hình, dùng:

```bash
python baseline.py
python train.py --model lstm
python train.py --model gru
python train.py --model bilstm
python compare_models.py
```

Notebook được giữ lại để thể hiện quá trình phát triển, không dùng làm nguồn logic production.

