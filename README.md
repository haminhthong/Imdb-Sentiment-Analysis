# 🎬 CineSentiment: IMDB Sentiment Analysis

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B.svg)](https://streamlit.io/)
[![CI](https://github.com/haminhthong/Imdb-Sentiment-Analysis/actions/workflows/quality.yml/badge.svg)](https://github.com/haminhthong/Imdb-Sentiment-Analysis/actions/workflows/quality.yml)

Dự án phân loại cảm xúc đánh giá phim (IMDB Sentiment Analysis) được thiết kế theo tư duy Machine Learning thực chất:
- **Baseline mạnh:** TF-IDF + Logistic Regression với tính giải thích qua các n-gram quan trọng.
- **Deep Learning:** Bidirectional LSTM (BiLSTM) tối ưu hóa chuỗi biến thiên bằng `pack_padded_sequence`.
- **Chống rò rỉ dữ liệu (Anti-Leakage):** Từ điển chỉ được xây dựng từ tập Train (Train-only vocabulary).
- **Hiệu chuẩn xác suất (Probability Calibration):** Áp dụng Temperature Scaling giải quyết hiện tượng overconfidence.
- **Phân tích lỗi (Error Analysis):** Khảo sát chuyên sâu theo các lát cắt ngôn ngữ học (phủ định, cảm xúc hỗn hợp, OOV, độ dài câu).
- **Phục vụ ứng dụng:** FastAPI REST API nhẹ nhàng và Web Demo trực quan với Streamlit.

---

## 🧭 Quy Trình Xử Lý (Pipeline Architecture)

```
        IMDB Reviews
             │
             ▼
      Clean / Tokenize (giữ nguyên "don't", "isn't")
             │
             ▼
   Train (80%) / Val (10%) / Calibration (10%)
             │
       ┌─────┴─────┐
       ▼           ▼
 TF-IDF + LR     BiLSTM (Embedding + pack_padded_sequence)
       │           │
       └─────┬─────┘
             ▼
     Validation Comparison (F1, Accuracy, Latency, Size)
             │
             ▼
   Temperature Calibration (học T trên Calibration set)
             │
             ▼
       Final Test (Brier Score, ECE, Reliability Diagram)
             │
             ▼
       Error Analysis (Negation, Mixed Sentiment, OOV, Length)
             │
             ▼
    FastAPI / Streamlit Demo
```

---

## 1. Dữ Liệu & Nguyên Tắc Chống Rò Rỉ (Data & Anti-Leakage)

Dự án sử dụng bộ dữ liệu kinh điển **Large Movie Review Dataset (IMDB v1.0)** gồm 50.000 đánh giá phân cực cân bằng (25.000 Train, 25.000 Test).

### Phân chia tập dữ liệu (Dataset Splitting)
- **Tập Train chính thức (25.000 mẫu)** được chia stratified thành:
  - **Train (80% - 20.000 mẫu):** Dùng để huấn luyện mô hình và cập nhật gradient.
  - **Validation (10% - 2.500 mẫu):** Dùng để giám sát dừng sớm (Early Stopping) và so sánh siêu tham số.
  - **Calibration (10% - 2.500 mẫu):** Dùng riêng biệt để học hệ số hiệu chuẩn Temperature Scaling $T$.
- **Tập Test chính thức (25.000 mẫu):** Hoàn toàn độc lập, tuyệt đối không dùng để chọn mô hình hoặc tune threshold.

### Xây dựng từ điển chỉ từ tập Train (Train-Only Vocabulary)
Một lỗi rò rỉ phổ biến trong các dự án NLP là fit từ điển hoặc TF-IDF trên toàn bộ dữ liệu (Train + Test). Khi đó, mô hình đã "nhìn trước" phân phối từ vựng của tập kiểm thử.
Trong dự án này:
- Bộ từ vựng `Vocabulary` **chỉ được trích xuất từ tập Train** sau khi đã phân chia split.
- Bất kỳ từ ngữ nào chỉ xuất hiện ở tập Validation hoặc Test sẽ được chuyển thành token `<UNK>`.

### Tiền xử lý & Tokenizer bảo tồn từ phủ định
- Giải mã thực thể HTML và loại bỏ các thẻ định dạng (`<br />`, `<p>`).
- Chuyển thành chữ thường.
- Biểu thức chính quy tách token được thiết kế để **bảo tồn nguyên vẹn các dạng phủ định viết tắt** như `don't`, `didn't`, `isn't`, `can't`, `won't` – yếu tố then chốt quyết định chiều hướng cảm xúc.

---

## 2. Baseline: TF-IDF + Logistic Regression

Thay vì trực tiếp kết luận *"cần dùng Deep Learning vì nó mạnh hơn"*, dự án thiết lập một baseline sparse-text rất mạnh:
- **Đặc trưng:** TF-IDF unigram + bigram (ngram_range=(1, 2)), `min_df=2`, `sublinear_tf=True`, tối đa 50.000 features.
- **Mô hình:** Hồi quy Logistic Regression với L2 regularization (`solver="liblinear"`).

### Khả năng giải thích (Lexical Explainability)
Thông qua hệ số (coefficients) của Logistic Regression, chúng ta có thể kiểm tra trực tiếp các tín hiệu từ ngữ mà mô hình dựa vào:
- **Top N-grams Tích cực:** `excellent`, `wonderfully acted`, `masterpiece`, `superb`, `brilliant`.
- **Top N-grams Tiêu cực:** `waste of time`, `poorly written`, `worst`, `terrible`, `awful`.

---

## 3. Deep Learning: BiLSTM với `pack_padded_sequence`

Kiến trúc mạng nơ-ron:
$$\text{Tokens} \longrightarrow \text{Embedding (128d)} \longrightarrow \text{BiLSTM (2 layers, hidden 128d)} \longrightarrow \text{Concat}[h_{\text{forward}}, h_{\text{backward}}] \longrightarrow \text{Dropout (0.4)} \longrightarrow \text{Linear} \longrightarrow \text{Logit}$$

### Tại sao bắt buộc dùng `pack_padded_sequence`?
Các bài đánh giá phim có độ dài rất khác nhau (từ 20 đến hơn 1.000 từ). Khi pad về độ dài cố định (`max_length=256`):
- Nếu RNN chạy qua toàn bộ vùng padding: vừa lãng phí tài nguyên tính toán, vừa khiến hidden state cuối cùng bị nhiễu bởi các token đệm.
- `torch.nn.utils.rnn.pack_padded_sequence` cho phép LSTM bỏ qua các bước thời gian thuộc vùng padding và chỉ xử lý độ dài thực tế của từng câu.

---

## 4. So Sánh Mô Hình (Model Comparison)

Mô hình Deep Learning có thực sự xứng đáng với chi phí tính toán không?

| Mô hình | Accuracy | Macro-F1 | Brier Score | ECE | Tham số | CPU Latency |
|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | ~89.2% | ~89.2% | 0.081 | 0.038 | ~50.000 | **~0.8 ms** |
| **BiLSTM (Calibrated)** | ~88.5% - 89.8% | ~88.5% - 89.8% | 0.078 | 0.024 | ~6.7M | **~12.5 ms** |

> **Nhận xét:** TF-IDF + Logistic Regression có tốc độ suy luận nhanh hơn khoảng 15 lần trên CPU với độ chính xác xấp xỉ BiLSTM. Tuy nhiên, BiLSTM có khả năng bắt được phụ thuộc ngữ cảnh chuỗi tốt hơn trong các câu có đảo chiều cảm xúc hoặc cấu trúc ngữ pháp phức tạp.

---

## 5. Hiệu Chuẩn Xác Suất (Temperature Scaling)

Các mạng nơ-ron hiện đại thường gặp hiện tượng **overconfidence** (xác suất đưa ra 99% nhưng độ chính xác thực tế chỉ 85%).

Dự án áp dụng **Temperature Scaling** trên tập Calibration:
$$\hat{p} = \sigma\left(\frac{z}{T}\right)$$
- Hệ số $T > 0$ được học bằng thuật toán L-BFGS để cực tiểu hóa Negative Log-Likelihood (NLL) trên Calibration split.
- Đo lường bằng **Brier Score** và **Expected Calibration Error (ECE)**, đồng thời kiểm tra bằng đồ thị **Reliability Diagram**.

---

## 6. Phân Tích Lỗi Chuyên Sâu (Error Analysis)

Dự án thực hiện phân tích lỗi thực chất theo 3 chiều:
1. **Lát cắt Ngôn ngữ học (Linguistic Slices):**
   - **Negation (`not`, `never`, `n't`):** Tỷ lệ lỗi tăng khi phủ định cách xa tính từ chính.
   - **Mixed Sentiment (`but`, `however`, `although`):** Model gặp khó khăn khi phần đầu khen nhưng đoạn kết lại chê.
   - **Sentiment Reversal (`at first`, `turned out`):** Đảo chiều cảm xúc ở cuối câu.
2. **Lát cắt Độ dài câu (Length Slices):**
   - Đánh giá ngắn (<=100 tokens), trung bình (101-256 tokens), và dài (>256 tokens).
   - Chiến lược cắt chuỗi `head_tail` (giữ nửa đầu và nửa kết luận) giảm lỗi so với việc chỉ giữ đoạn đầu (`first`).
3. **Lát cắt Tỷ lệ từ ngoài từ điển (OOV Slices):**
   - Khảo sát sự tương quan giữa tỷ lệ `<UNK>` và tỷ lệ lỗi dự đoán của mô hình word-level RNN.

---

## 7. Hướng Dẫn Cài Đặt & Chạy Dự Án

### Cài đặt môi trường
```bash
git clone https://github.com/haminhthong/Imdb-Sentiment-Analysis.git
cd Imdb-Sentiment-Analysis
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

### 1. Tải dữ liệu IMDB
```bash
python scripts/download_imdb.py
```

### 2. Huấn luyện Baseline (TF-IDF + Logistic Regression)
```bash
python baseline.py --max-features 50000
```

### 3. Huấn luyện BiLSTM & Hiệu chuẩn xác suất
```bash
python train.py --epochs 15 --batch-size 128 --truncation-strategy head_tail
```

### 4. Đánh giá trên tập Test
```bash
python evaluate.py --checkpoint artifacts/model.pt
```

### 5. So sánh đối chiếu hai mô hình
```bash
python compare_models.py
```

### 6. Phân tích lỗi (Error Analysis)
```bash
python -m scripts.analyze_errors --checkpoint artifacts/model.pt
```

### 7. Chạy dự đoán dòng lệnh (CLI)
```bash
python predict.py "This movie was beautifully directed and passionately acted!"
```

---

## 8. Phục Vụ Ứng Dụng (Serving & Demo)

### FastAPI REST API
Khởi động API server:
```bash
python api.py
# hoặc: uvicorn api:app --host 0.0.0.0 --port 8000
```
Swagger UI tài liệu tương tác sẵn sàng tại: `http://localhost:8000/docs`

**Ví dụ gọi API:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The cinematography was great, but the plot was completely boring."}'
```
**Response:**
```json
{
  "label": "Negative",
  "probability": 0.3214,
  "truncated": false,
  "oov_rate": 0.0,
  "token_count": 12,
  "warnings": []
}
```

### Streamlit Web Demo
Khởi động giao diện tương tác:
```bash
streamlit run app.py
```

---

## 9. Kiểm Thử Tự Động & CI

Chạy bộ unit test toàn diện:
```bash
pytest
```
Kiểm tra code style và format bằng Ruff:
```bash
ruff check .
ruff format --check .
```
GitHub Actions workflow (`.github/workflows/quality.yml`) tự động thực thi linter và test suite trên mỗi commit.
