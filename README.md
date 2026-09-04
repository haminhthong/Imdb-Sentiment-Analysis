# 🎬 CineSentiment AI — Deep Learning Sentiment Analysis Platform

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Clean%20Code-brightgreen.svg)](https://github.com/psf/black)

> **Dự án cá nhân chuẩn mực (End-to-End MLOps & Deep Learning Portfolio Project)** phân loại cảm xúc đánh giá phim tiếng Anh (IMDB Dataset) thành **Positive** hoặc **Negative** sử dụng các mô hình Recurrent Neural Network (LSTM, GRU, BiLSTM) với PyTorch.

---

## 🌟 Giá Trị Nổi Bật Dành Cho Portfolio & CV

Dự án này được thiết kế theo đúng chuẩn kỹ thuật của một sản phẩm **Production-Ready ML Engine**:

1. **Kiến Trúc Mô Hình & PyTorch Best Practices:**
   - Xây dựng lớp mô hình thống nhất (`SentimentRNN`) hỗ trợ cả **LSTM**, **GRU**, và **BiLSTM** (Bidirectional LSTM).
   - Tối ưu hiệu năng tính toán với `pack_padded_sequence` giúp RNN bỏ qua các bước thời gian thuộc vùng padding (`<PAD>`).
   - Tự động đếm tham số mô hình (`count_parameters`) và đo độ trễ suy luận (**Inference Latency** ms/sample).

2. **Chống Rò Rỉ Dữ Liệu Chặt Chẽ (Anti Data Leakage):**
   - Loại bỏ các câu trùng lặp nội dung hoàn toàn.
   - **Tự động xóa các mẫu Test có nội dung đã xuất hiện trong tập Train** để đảm bảo phép đo khả năng tổng quát hóa hoàn toàn khách quan.
   - Bộ từ vựng (`Vocabulary`) **CHỈ** được xây dựng từ phần dữ liệu Train sau khi đã chia tập Validation theo tỷ lệ nhãn (Stratified Split).

3. **Huấn Luyện Nâng Cao & Tái Lập (Reproducibility):**
   - Cố định hạt giống ngẫu nhiên (Random Seed) cho Python, NumPy, và PyTorch cuDNN.
   - Kỹ thuật **Gradient Clipping** phòng chống bùng nổ gradient.
   - Thuật toán **AdamW Optimizer** kết hợp L2 Weight Decay và **Early Stopping** tự động khôi phục checkpoint tốt nhất khi Validation Loss dừng giảm.

4. **Kiến Trúc Code Sạch & Modular (Clean Code & Software Engineering):**
   - Phân tách rõ ràng giữa Data Processing, Model Definition, Training Engine, Inference, Artifact Management và Utilities.
   - Đóng gói Type Hints (PEP 484/585) và **Chú thích docstrings Tiếng Việt chuẩn mực Google-style** cho 100% các lớp/hàm.

5. **MLOps & Triển Khai Đa Nền Tảng:**
   - **REST API (FastAPI):** Cung cấp các endpoints `/health`, `/ready`, `/info`, `/predict` và `/predict/batch` tích hợp OpenAPI Swagger Docs.
   - **Web Demo (Streamlit):** Giao diện tương tác 3 tab (Single Prediction, Batch Upload Analysis, Model Architecture Inspector).
   - **Đóng Gói Docker:** Container hóa nhẹ nhàng với `Dockerfile` chạy uvicorn server.
   - **CI/CD Pipeline:** Tự động hóa kiểm thử và linting với GitHub Actions (`quality.yml`).

---

## 📐 Kiến Trúc Hệ Thống (System Architecture)

```mermaid
flowchart TD
    subgraph Data Pipeline
        A[IMDB Dataset CSV] --> B[Data Cleaning & Deduplication]
        B --> C[Anti-Leakage Filter]
        C --> D[Stratified Train/Val Split]
        D --> E[Train Vocabulary Builder]
    end

    subgraph PyTorch Engine
        E --> F[PyTorch DataLoader]
        F --> G[SentimentRNN Model\n(LSTM / GRU / BiLSTM)]
        G --> H[Early Stopping & Validation Loop]
        H --> I[Save Checkpoint model.pt & Metrics]
    end

    subgraph Deployment Interfaces
        I --> J[CLI Tool\npredict.py]
        I --> K[FastAPI REST Server\napi.py]
        I --> L[Streamlit Web App\napp.py]
        K --> M[Docker Container]
    end
```

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
.
├── sentiment/                  # Package xử lý dữ liệu, mô hình và suy luận
│   ├── __init__.py             # Package initializer
│   ├── config.py               # Quản lý siêu tham số ExperimentConfig Dataclass
│   ├── text.py                 # Tokenizer, Vocabulary, Encoding và Padding
│   ├── data_validation.py      # Schema, duplicate và train-test overlap
│   ├── data.py                 # IMDBDataset, DataLoader và DataBundle
│   ├── model.py                # Mô hình SentimentRNN thống nhất (LSTM, GRU, BiLSTM)
│   ├── engine.py               # Vòng lặp Training, Early Stopping, Evaluation Metrics
│   ├── inference.py            # SentimentPredictor phục vụ suy luận đơn lẻ & Batch
│   ├── artifacts.py            # Quản lý lưu Checkpoint, tệp JSON và biểu đồ high-res
│   └── utils.py                # Seed ngẫu nhiên, Device selector & Latency Benchmark
├── tests/                      # Unit tests cho logic quan trọng
├── notebooks/legacy/           # Notebook ban đầu, chỉ giữ làm lịch sử
├── app.py                      # Giao diện Web Demo đa tab bằng Streamlit
├── api.py                      # REST API Server với FastAPI & Pydantic Schemas
├── train.py                    # Script CLI huấn luyện mô hình
├── baseline.py                 # TF-IDF + Logistic Regression baseline
├── predict.py                  # Script CLI suy luận dự đoán nhanh
├── compare_models.py           # Script tổng hợp bảng so sánh Benchmark mô hình
├── load_test.py                # Load test concurrency, p50/p95 và throughput
├── download_data.py            # Script tạo/nạp dữ liệu mẫu thử nghiệm
├── Dockerfile                  # Đóng gói sản phẩm thành Docker Image
├── MODEL_CARD.md               # Báo cáo phạm vi, giới hạn và đạo đức AI
├── DATASET_CARD.md              # Nguồn, schema, leakage và giới hạn dữ liệu
├── PORTFOLIO_GUIDE.md          # Checklist và bộ câu hỏi phỏng vấn CV
├── train.csv                   # Tập dữ liệu huấn luyện (IMDB Reviews)
├── test.csv                    # Tập dữ liệu kiểm thử độc lập
└── requirements.txt            # Danh sách các thư viện phụ thuộc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Khởi tạo Môi trường Ảo (Virtual Environment)

**Khuyến nghị Python 3.10 trở lên.**

Tạo môi trường ảo:
```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:
- **Windows PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

Cài đặt các thư viện cần thiết:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Lưu ý GPU CUDA:** Nếu muốn huấn luyện tăng tốc trên GPU NVIDIA, hãy cài bản `torch` tương thích với phiên bản CUDA của bạn từ trang chủ PyTorch trước khi cài `requirements.txt`.

---

### 2. Chuẩn bị Dữ liệu Huấn Luyện

Dự án đi kèm dữ liệu IMDB chuẩn (`train.csv` và `test.csv`). Mỗi tệp CSV yêu cầu 2 cột:
- `text` (string): Nội dung câu đánh giá phim.
- `label` (int/float): `0` cho Negative (Tiêu cực), `1` cho Positive (Tích cực).

Nếu chưa có dữ liệu chính thức, bạn có thể tạo nhanh dữ liệu mẫu bằng lệnh:
```bash
python download_data.py
```

---

### 3. Huấn Luyện Mô Hình (Training CLI)

Trước khi huấn luyện RNN, chạy baseline đơn giản để có mốc so sánh:

```bash
python baseline.py --output-dir artifacts/baseline
```

Baseline sử dụng TF-IDF bigram và Logistic Regression với cùng quy tắc làm sạch,
chia dữ liệu và chống train-test overlap như pipeline PyTorch.

Bạn có thể huấn luyện cả 3 kiến trúc mô hình (`bilstm`, `gru`, `lstm`) bằng tệp `train.py`:

**Huấn luyện BiLSTM (Cấu hình mặc định khuyến nghị):**
```bash
python train.py --model bilstm --output-dir artifacts/bilstm
```

**Huấn luyện GRU hoặc LSTM:**
```bash
python train.py --model gru --output-dir artifacts/gru
python train.py --model lstm --output-dir artifacts/lstm
```

**Chạy thử nghiệm nhanh (Smoke Test 1 Epoch):**
```bash
python train.py --model gru --epochs 1 --batch-size 256 --device cpu --output-dir artifacts/smoke-test
```

**Các tham số dòng lệnh CLI khả dụng:**

| Tham số | Mặc định | Ý nghĩa / Mô tả |
|---|---:|---|
| `--model` | `bilstm` | Lựa chọn kiến trúc: `lstm`, `gru`, hoặc `bilstm` |
| `--train-data` | `train.csv` | Đường dẫn tệp CSV dùng để chia Train/Validation |
| `--test-data` | `test.csv` | Đường dẫn tệp CSV dữ liệu Test độc lập |
| `--output-dir` | `artifacts/<model>` | Thư mục lưu xuất Checkpoint và Đồ thị |
| `--epochs` | `15` | Số lượng Epoch huấn luyện tối đa |
| `--batch-size` | `128` | Kích thước Lô (Batch size) |
| `--device` | `auto` | Thiết bị tính toán: `auto`, `cpu`, hoặc `cuda` |

---

### 4. Dự Đoán Nhanh Từ Dòng Lệnh (Prediction CLI)

Dự đoán cảm xúc của một câu đánh giá bất kỳ:
```bash
python predict.py "This movie is an absolute masterpiece with brilliant acting!" --checkpoint artifacts/bilstm/model.pt
```

**Kết quả đầu ra mẫu:**
```text
============================================================
🎬 KẾT QUẢ PHÂN TÍCH CẢM XÚC - CINESENTIMENT AI
============================================================
• Nội dung nhập vào : "This movie is an absolute masterpiece with brilliant acting!"
• Cảm xúc dự đoán   : [POSITIVE]
• Xác suất Positive : <kết quả thực tế từ checkpoint>
• Độ tin cậy (Conf) : <kết quả thực tế từ checkpoint>
============================================================
```

---

### 5. Khởi Chạy Web Demo (Streamlit App)

Sau khi đã huấn luyện mô hình và có tệp Checkpoint (`artifacts/bilstm/model.pt`):

```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở giao diện tương tác 3 Tab:
1. **📝 Phân Tích Câu Đơn:** Nhập câu văn bản, xem mẫu câu có sẵn, đo độ tin cậy và mức độ tích cực.
2. **📊 Phân Tích Batch:** Nhập danh sách nhiều câu (mỗi câu 1 dòng) để dự đoán theo lô hàng loạt.
3. **⚙️ Kiến Trúc & Benchmark:** Xem chi tiết tham số mô hình và các đồ thị Loss/Accuracy/Confusion Matrix.

---

### 6. Khởi Chạy REST API (FastAPI Server)

Chạy dịch vụ REST API với Uvicorn:
```bash
uvicorn api:app --reload --port 8000
```

Truy cập tài liệu tương tác **Swagger UI** tại: `http://localhost:8000/docs`

**Ví dụ gọi API bằng `curl`:**

**Endpoint Phân Tích Câu Đơn (`POST /predict`):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The cinematography was breathtaking and acting was superb!"}'
```

**Response JSON mẫu:**
```json
{
  "label": "Positive",
  "positive_probability": 0.9342,
  "confidence": 0.9342
}
```

---

### 7. Đóng Gói & Triển Khai Docker

Xây dựng Docker Image:
```bash
docker build -t cinesentiment-api:latest .
```

Chạy Docker Container:
```bash
docker run -d -p 8000:8000 --name cinesentiment-container cinesentiment-api:latest
```

Kiểm tra API trên container tại: `http://localhost:8000/health`

---

### 8. Kiểm Tra Tải API

Sau khi API và checkpoint đã sẵn sàng, mô phỏng 100 người dùng đồng thời:

```bash
python load_test.py --users 100 --requests 500
```

Script báo cáo số request thành công/lỗi, throughput, mean, p50 và p95 latency.
Kết quả phụ thuộc phần cứng, số worker và checkpoint nên phải đo trước khi đưa vào CV.

---

## 📊 Bảng So Sánh Benchmark Giữa Các Mô Hình

Sau khi chạy baseline và đủ ba kiến trúc (`lstm`, `gru`, `bilstm`), chạy:

```bash
python compare_models.py
```

Script sẽ tự động đo lường số lượng tham số, đo thời gian suy luận (Inference Latency trên CPU) và đọc các chỉ số thực tế từ `metrics.json` để tạo ra bảng so sánh bên dưới:

| Model | Parameters | Test Loss | Accuracy | Macro F1 | Latency (CPU) |
|---|---:|---:|---:|---:|---:|
| **TF-IDF + Logistic Regression** | N/A | 0.3205 | 89.91% | 89.91% | *Chưa đo* |
| **LSTM** | *Đang cập nhật* | *Đang cập nhật* | *Đang cập nhật* | *Đang cập nhật* | *Đang cập nhật* |
| **GRU** | *Đang cập nhật* | *Đang cập nhật* | *Đang cập nhật* | *Đang cập nhật* | *Đang cập nhật* |
| **BiLSTM** | *Đang cập nhật* | *Đang cập nhật* | *Đang cập nhật* | *Đang cập nhật* | *Đang cập nhật* |

*(Bảng trên sẽ tự động được điền các chỉ số đo đạc thực tế từ tệp `artifacts/model_comparison.md`)*

---

## 🧪 Bộ Kiểm Thử Tự Động (Unit Testing)

Dự án dùng `pytest` cho các logic quan trọng. Repository chưa công bố coverage 100%
vì chưa đo bằng công cụ coverage trong môi trường clean install.

Chạy bộ test:
```bash
python -m pytest --basetemp=./scratch/pytest_temp
```

Kiểm tra biên dịch cú pháp mã nguồn:
```bash
python -m compileall sentiment train.py baseline.py predict.py api.py app.py compare_models.py load_test.py
```

---

## 🎓 Bài Tập Chuẩn Bị Phỏng Vấn (Interview Prep Q&A)

Khi đưa dự án này vào CV, nhà tuyển dụng có thể hỏi các câu hỏi chuyên sâu sau:

1. **Tại sao Bộ Từ Vựng (Vocabulary) CHỈ được tạo từ tập Train?**
   - *Trả lời:* Để tránh rò rỉ dữ liệu (Data Leakage). Nếu xây từ điển từ cả tập Test/Val, mô hình sẽ "biết trước" sự tồn tại của các từ trong bài kiểm tra, dẫn đến đánh giá độ chính xác bị thổi phồng không thực tế.
2. **Kỹ thuật `pack_padded_sequence` mang lại lợi ích gì trong PyTorch?**
   - *Trả lời:* Giúp RNN bỏ qua các bước thời gian thuộc vùng padding `<PAD>`. Điều này tiết kiệm bộ nhớ, tăng tốc độ huấn luyện/suy luận và tránh làm nhiễu hidden state của mô hình.
3. **Sự khác biệt về kiến trúc giữa LSTM và GRU là gì?**
   - *Trả lời:* LSTM có 3 cổng (Input, Forget, Output) và giữ 2 trạng thái `(hidden state h, cell state c)`. GRU đơn giản hơn với 2 cổng (Reset, Update) và chỉ có 1 trạng thái `hidden state h`, giúp GRU huấn luyện nhanh hơn và ít tham số hơn (~25% ít tham số hơn LSTM).
4. **BiLSTM mang lại lợi thế gì so với LSTM một chiều?**
   - *Trả lời:* BiLSTM xử lý văn bản theo cả 2 chiều (từ trái sang phải và từ phải sang trái), giúp mô hình nắm bắt được ngữ cảnh cả trước và sau của một từ trong câu (đặc biệt quan trọng với các cấu trúc mỉa mai hoặc phủ định ở cuối câu).

---

## 📝 Tác Giả & Giấy Phép

- **Tác giả:** ML / MLOps Engineer Candidate
- **Giấy phép:** MIT License. Sử dụng tự do cho mục đích học tập và xây dựng Portfolio.
