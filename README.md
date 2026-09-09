# 🎬 CineSentiment — Leakage-Safe Calibrated BiLSTM Service

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI](https://github.com/haminhthong/Imdb-Sentiment-Analysis/actions/workflows/quality.yml/badge.svg)](https://github.com/haminhthong/Imdb-Sentiment-Analysis/actions/workflows/quality.yml)
[![Code Style](https://img.shields.io/badge/Code%20Style-Clean%20Code-brightgreen.svg)](https://github.com/psf/black)

> **CineSentiment** là dịch vụ phân loại sentiment tiếng Anh với một lifecycle
> duy nhất: TF-IDF + Logistic Regression làm baseline, BiLSTM làm candidate
> production, Calibration Set riêng và Official Test bị khóa.

---

## 📌 Pipeline Tóm Tắt Dành Cho CV & Portfolio

```text
IMDB Reviews
  → Data QA & Multi-Level Anti-Leakage (Exact + Normalized SHA256)
  → Train-Only Text Processing Contract
  → TF-IDF Baseline & BiLSTM Candidate
  → Validation Leaderboard & Release Gate
  → Final Fit on Train + Validation
  → Temperature Scaling & Abstention Policy on Calibration
  → Single-Pass Locked Official Test Evaluation + Prediction Facts
  → Multi-Slice Error Analysis từ Prediction Facts
  → Versioned Artifact Packaging Schema v3
  → Production-Oriented FastAPI & Streamlit Serving
```

---

## 1. Bài Toán & Phạm Vi Ứng Dụng (Problem & Scope)

- **Mục tiêu:** Phân loại cảm xúc đánh giá phim tiếng Anh thành `Positive` (Tích cực) hoặc `Negative` (Tiêu cực).
- **Trọng tâm kỹ thuật:** Không chạy theo việc gom nhặt số lượng lớn các mô hình phức tạp một cách tùy tiện, mà tập trung xây dựng một hệ thống kỹ thuật NLP chuẩn chỉnh:
  $$\text{Data QA} \longrightarrow \text{Leakage Control} \longrightarrow \text{Model Selection} \longrightarrow \text{Calibration} \longrightarrow \text{Final Test} \longrightarrow \text{Serving} \longrightarrow \text{Monitoring}$$
- **Định vị:** Chuyển đổi từ *"Deep Learning Sentiment Classification đơn thuần"* sang **"Sentiment Intelligence Platform — Leakage-Safe NLP Benchmarking, Calibrated Prediction & Production-Oriented Serving"**.

---

## 2. Kiến Trúc 9 Giai Đoạn Chuẩn Mực (Canonical 9-Stage Architecture)

Đây là quy trình kỹ thuật duy nhất chi phối toàn bộ mã nguồn, cấu hình và báo cáo của dự án:

```mermaid
flowchart TD
    subgraph S1["1. DATA INGESTION"]
        A[IMDB Reviews Dataset\nTrain Source + Official Test]
    end

    subgraph S2["2. DATA QUALITY & LEAKAGE CONTROL"]
        B1[Schema & Binary Label Validation] --> B2[Conflicting-Label Detection]
        B2 --> B3[Exact Raw Hash Dedup]
        B3 --> B4[Normalized Exact Hash Dedup & Overlap Audit]
    end

    subgraph S3["3. DEVELOPMENT SPLIT"]
        C1[Official Train Source] --> C2[Train 80%]
        C1 --> C3[Validation 10%]
        C1 --> C4[Calibration 10%]
        C5[Official Test Locked & Untouched]
    end

    subgraph S4["4. TRAIN-ONLY TEXT CONTRACT"]
        D1[HTML Cleanup & Lowercase] --> D2[word-regex-v2 Tokenizer]
        D2 --> D3[Vocabulary Fit on Train ONLY]
        D3 --> D4[Truncation & Padding Contract]
    end

    subgraph S5["5. MODEL DEVELOPMENT"]
        E1[TF-IDF + Logistic Regression Baseline]
        E2[BiLSTM Candidate]
        E1 --> E3[Validation Leaderboard]
        E2 --> E3
        E3 --> E4[Multi-Criteria Champion Policy]
    end

    subgraph S6["6. PROBABILITY / DECISION LAYER"]
        F1[Final Raw Model Logits on Calibration] --> F2[Temperature Scaling on Calibration]
        F2 --> F3[Confidence Threshold and Selective Policy]
    end

    subgraph S7["7. LOCKED FINAL TEST"]
        G1[Frozen Champion] --> G2[Official Test Evaluated ONCE]
        G2 --> G3[Metrics + Evaluation Records + Error Slices]
    end

    subgraph S8["8. MODEL PACKAGING"]
        H[Weights + Vocabulary + word-regex-v2 Contract + Schema v3 Metadata]
    end

    subgraph S9["9. ONLINE SERVING"]
        I1[Input Validation & English/OOV Audit] --> I2[Champion Model & Calibrated Inference]
        I2 --> I3[Positive / Negative / Review Required]
        I3 --> I4[FastAPI & Streamlit + Reliability Warnings]
    end

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9
```

---

## 3. Kiểm Soát Chất Lượng & Chống Rò Rỉ Đa Tầng (Data Quality & Anti-Leakage Policy)

Nhiều dự án NLP thông thường chỉ lọc trùng lặp bằng câu lệnh đơn giản `df.drop_duplicates(subset=['text'])`. Điều này bỏ lọt các trường hợp rò rỉ tinh vi:
- Biến thể viết hoa/thường: `"This movie is GREAT!"` và `"This movie is great"`
- Thẻ định dạng HTML dư thừa: `"This movie is great.<br /><br />"` và `"This movie is great."`
- Khoảng trắng và dấu chấm câu ngoại lai.

### Cơ chế chống rò rỉ và bảo toàn benchmark:
1. **Raw exact check:** phát hiện cùng chuỗi gốc nhưng khác nhãn trước khi deduplicate.
2. **Normalized exact hash (`compute_normalized_text_hash`):**
   $$\text{Raw Text} \xrightarrow{\text{HTML Unescape}} \xrightarrow{\text{Strip Tags}} \xrightarrow{\text{Lowercase}} \xrightarrow{\text{Canonical Tokenize}} \text{"this movie is great"} \xrightarrow{\text{SHA256}} \text{Hash}$$
3. **Phát hiện nhãn mâu thuẫn:** Nếu cùng một nội dung thô hoặc normalized exact có nhãn khác nhau, pipeline dừng bằng `ValueError`.
4. **Official Test immutable:** overlap giữa Train và Test chỉ được audit; nếu có overlap, pipeline fail-fast bằng `DATASET AUDIT FAILED`, không lọc hoặc sửa Test.

---

## 4. Giao Thức Train / Validation / Test: Loại Bỏ Triệt Để Test Peeking (P0)

> [!CAUTION]
> **Nguyên tắc cốt lõi về Đánh Giá Khách Quan:**
> Không bao giờ dùng tập Test làm tập chọn mô hình (Model Selection Set)!

- **Sai lầm phổ biến:** Huấn luyện LSTM, GRU, BiLSTM, sau đó đánh giá cả 3 mô hình trên tập Test rồi tuyên bố *"BiLSTM tốt nhất trên Test nên ta chọn BiLSTM"*. Thực chất tập Test đã bị biến thành tập chọn mô hình và con số báo cáo bị lạc quan thái quá (Overoptimistic / Data Snooping Bias).
- **Giao thức chuẩn của CineSentiment:**
  1. `train.py` và `baseline.py`: Chỉ đọc Official Train, tạo Train/Validation/Calibration và chỉ xuất validation artifacts. Không nhận `--test-data`.
  2. `compare_models.py`: Chỉ đọc validation artifacts để tạo leaderboard và chọn Champion; calibration chưa tham gia model selection.
  3. `python -m scripts.final_fit` và `python -m scripts.calibrate`: fit lại BiLSTM trên Train + Validation rồi khóa temperature/policy từ Calibration.
  4. `python -m scripts.evaluate_release`: entrypoint duy nhất mở Official Test, audit overlap, đánh giá một lượt và sinh `test_metrics.json`, `evaluation_records.csv`, `final_test_report.md`.
- **Quy ước tên gọi:** BiLSTM là cấu hình thực thi mặc định (**default implementation**), chỉ được tuyên bố là mô hình tốt nhất khi có kết quả benchmark thực nghiệm đầy đủ.

---

## 5. Hợp Đồng Xử Lý Văn Bản (Train-Only Text Contract)

- **Phiên bản Tokenizer:** `TOKENIZER_VERSION = "word-regex-v2"`.
- **Bảo toàn từ phủ định:** Regex `[a-z0-9]+(?:'[a-z0-9]+)?` giữ nguyên các từ co cụm phủ định như `don't`, `isn't`, `can't`, `won't`. Phủ định là tín hiệu quyết định trong phân tích cảm xúc, do đó hệ thống không áp dụng stopword removal tùy tiện.
- **Train-Only Vocabulary:** Bộ từ vựng (`Vocabulary`) được xây dựng **CHỈ** từ tập Train (sau khi đã chia Stratified Split). Mọi token chỉ xuất hiện ở Validation hoặc Test được ánh xạ chính xác về token đặc biệt `<UNK>`.
- **Toàn vẹn Train-Serving (Parity Invariant):** Cả quá trình huấn luyện và suy luận trực tuyến đều dùng chung hàm `encode_with_audit`, cam đoan cùng một văn bản sẽ tạo ra danh sách token và chỉ số index giống nhau 100%.

---

## 6. Mô Hình Cơ Sở: TF-IDF + Logistic Regression (First-Class Baseline)

Dự án không coi mô hình tuyến tính là "cho có", mà thiết lập nó thành một đối trọng học thuật mạnh mẽ:
- **Pipeline:** `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)` kết hợp `LogisticRegression(solver='liblinear')`.
- **Câu hỏi kỹ thuật cốt lõi:** *Mạng nơ-ron hồi quy sâu (RNN) có thực sự đem lại giá trị vượt trội so với baseline tuyến tính mạnh mẽ hay không khi xét trên bài toán đánh đổi giữa hiệu năng, kích thước mô hình và độ trễ CPU?*
- **Khả năng giải thích (Explainability):** Trích xuất top các n-gram có trọng số dương và âm lớn nhất từ hệ số hồi quy (kèm lưu ý: tương quan đặc trưng trong dữ liệu không đồng nghĩa với quan hệ nhân quả).

---

## 7. Kiến Trúc Mô Hình Production (BiLSTM)

Lifecycle production dùng `SentimentRNN` với cấu hình `model_type="bilstm"`:
- **Embedding Layer:** Nhúng từ ngữ không gian chiều thấp với `padding_idx` giúp bỏ qua cập nhật gradient tại các vị trí `<PAD>`.
- **Tối ưu hóa Sequence:** Tích hợp `torch.nn.utils.rnn.pack_padded_sequence` với độ dài thực tế của từng câu, loại bỏ hoàn toàn tính toán lãng phí trên vùng padding và tránh làm biến dạng hidden state.
- **BiLSTM Context:** Chiều xuôi và chiều ngược của lớp ẩn cuối cùng được ghép nối (`torch.cat((hidden[-2], hidden[-1]), dim=1)`), nắm bắt ngữ cảnh từ cả hai phía của câu đánh giá.
- **Regularization & Gradient Clipping:** Sử dụng `Dropout`, `AdamW` với Weight Decay và cắt norm gradient `clip_grad_norm_ <= 1.0` phòng chống bùng nổ gradient.

LSTM/GRU vẫn tồn tại trong module để đọc các artifact/test legacy cũ; chúng không
được train hoặc so sánh trong CLI production hiện tại và không phải release target.

---

## 8. Chính Sách Lựa Chọn Champion (Champion Selection Policy)

Thay vì chỉ chọn mô hình thuần túy theo Accuracy, hệ thống áp dụng bộ tiêu chí đa chiều:

| Nhóm Tiêu Chí | Chỉ Số Đo Lường | Mục Đích |
|---|---|---|
| **Primary Metric** | **Validation Macro-F1** | Đảm bảo cân bằng giữa cả 2 lớp Positive và Negative |
| **Generalization Guardrail**| **Validation Log-Loss / ROC-AUC** | Theo dõi chất lượng ngoài Macro-F1 |
| **Operational Guardrail** | **CPU Latency & Model Params** | Đảm bảo khả năng phục vụ với độ trễ thấp và chi phí tối ưu |
| **Occam's Razor Rule** | **Simplicity Margin ($\Delta \text{F1} < 0.005$)** | Nếu mô hình đơn giản hơn chỉ kém dưới 0.5% F1, ưu tiên mô hình đơn giản |

Nếu BiLSTM không cải thiện Validation Macro-F1 và còn chậm hơn baseline, code
giữ TF-IDF baseline như release gate thay vì ép deep model thành champion.

---

## 9. Tầng Xác Suất & Hiệu Chuẩn (Calibration & Decision Layer - P0)

> [!IMPORTANT]
> Điểm Sigmoid thô của mạng nơ-ron sâu ($\sigma(z)$) **chưa phải là xác suất chuẩn xác**. Mô hình học sâu thường mắc lỗi tự tin thái quá (dự đoán xác suất 0.99 nhưng thực tế sai 10%).

- **Temperature Scaling:** Học một tham số vô hướng $T > 0$ tối ưu trên Calibration Logits để làm mềm phân phối xác suất:
  $$\hat{p} = \sigma\left(\frac{z}{T}\right)$$
- **Các chỉ số đo lường hiệu chuẩn:**
  - **Brier Score:** Sai số bình phương trung bình giữa xác suất dự đoán và nhãn thực tế.
  - **Expected Calibration Error (ECE):** Độ lệch trung bình có trọng số giữa độ tin cậy và độ chính xác thực tế qua 10 bins xác suất.
- **Selective classification:** threshold phân loại là 0.5; `confidence = max(p, 1-p)`. Calibration chọn `confidence_threshold = τ`; confidence dưới τ được trả về `review_required`.

---

## 10. Giao Thức Đánh Giá Locked Final Test

Chạy `python -m scripts.evaluate_release` sau khi model, vocabulary,
temperature và confidence policy đã đóng băng. Evaluator đọc Official Train chỉ
để audit overlap, đọc Official Test nguyên vẹn đúng một lượt, sau đó ghi trong
`artifacts/releases/v1.0.0/`:

- `test_metrics.json`: Accuracy, Macro-F1, ROC-AUC, PR-AUC, Log Loss, Brier, ECE và selective metrics.
- `evaluation_records.csv`: prediction facts không chứa raw review text, dùng cho error analysis.
- `final_test_report.md`: bảng metrics, confusion matrix, hash và các slice độ dài/OOV/truncation.
- `benchmark.json`: latency CPU p50/p95 và throughput batch đại diện.

---

## 11. Phân Tích Lỗi & Taxonomy Ngôn Ngữ Học (Error Analysis)

Sau locked evaluation, dự án không dừng lại ở con số tổng thể mà mổ xẻ các nhóm lỗi từ `evaluation_records.csv` thông qua `analyze_errors.py`. Script này không đọc Official Test lần hai:
- **Phủ định (Negation):** Các mẫu chứa `not`, `n't`, `never`, `hardly` (ví dụ: *"not bad at all"* bị mô hình hiểu nhầm thành tiêu cực do có từ `bad`).
- **Mỉa mai (Sarcasm / Irony):** Sử dụng từ vựng tích cực với hàm ý châm biếm (ví dụ: *"Yeah, that was two hours well spent..."*).
- **Cảm xúc hỗn hợp (Mixed Sentiment):** Chứa liên từ đảo nghĩa `but`, `however`, `although` (ví dụ: *"Great acting, terrible story"*).
- **Đảo chiều cảm xúc (Sentiment Reversal):** Mở đầu khen ngợi nhưng kết luận tiêu cực ở cuối (ví dụ: *"At first I loved it... but the ending ruined everything"*).
- **Từ nhấn mạnh (Intensifiers):** `absolutely`, `totally`, `utterly`.

---

## 12. Độ Bền Vững, Truncation & Kiểm Toán OOV (Robustness Audit)

- **Audit Phân Phối Độ Dài Token:** Báo cáo các phân vị $p_{50}, p_{90}, p_{95}, \max$ và tỷ lệ cắt chuỗi ($\text{truncation\_rate} = \% \text{ review} > \text{max\_length}$).
- **Chiến Lược Cắt Chuỗi (Truncation Strategies):**
  - `first`: Lấy $N$ token đầu câu.
  - `head_tail`: Lấy $N/2$ token đầu kết hợp $N/2$ token cuối câu (nắm bắt kết luận quan trọng của người xem phim ở đoạn kết).
- **Kiểm toán OOV:** Đo lường tỷ lệ token ngoài từ điển (Train OOV, Val OOV, Test OOV).

---

## 13. Phục Vụ Trực Tuyến: REST API & Streamlit (Serving Layer)

### REST API (FastAPI)
- **Chuẩn hóa thông điệp:** Phân biệt rõ giữa `positive_probability` (đã hiệu chuẩn) và `positive_score` (sigmoid thô).
- **Unified artifact loader:** `sentiment.inference.load_predictor` tự nhận diện `model.pt` (BiLSTM) hoặc `model.joblib` (baseline) và trả cùng một response contract.
- **Cảnh báo độ tin cậy thời gian thực:** Tự động phát hiện và gắn cờ:
  - `TRUNCATED_INPUT`: Câu vượt độ dài `max_length`.
  - `HIGH_OOV_WARNING`: Câu có tỷ lệ từ lạ vượt 20%.
  - `OUT_OF_DOMAIN_LANGUAGE_HEURISTIC`: heuristic cảnh báo input có thể ngoài miền tiếng Anh; đây không phải language detector.
- **Endpoints:**
  - `GET /health` & `GET /ready`: Giám sát sẵn sàng phục vụ.
  - `GET /metrics`: Metric Prometheus (request count, latency, errors).
  - `GET /info`: Siêu tham số mô hình, temperature và kích thước từ điển.
  - `POST /predict`: Phân tích câu đơn kèm kiểm toán.
  - `POST /predict/batch`: Phân tích theo lô (tối đa 100 câu).

Khi validation chọn baseline làm Champion, chạy `python -m scripts.package_baseline_release`
rồi trỏ `CHECKPOINT_PATH` tới `artifacts/releases/v1.0.0/model.joblib`. Baseline
không cắt chuỗi và không có OOV theo vocabulary neural, nhưng vẫn trả cùng trường
`PredictionResult` để API/CLI/Streamlit không đổi giao diện.

### Streamlit Web App
- Giao diện trực quan 3 tab:
  1. Phân tích câu đơn với thanh đo xác suất hiệu chuẩn, badge quyết định và cảnh báo độ tin cậy.
  2. Phân tích theo lô với bảng thống kê số token và cờ bất định.
  3. Trình bày pipeline, Development Leaderboard và các chỉ số runtime của release.

---

## 14. Giám Sát & Phát Hiện Trôi Dạt Dữ Liệu (Monitoring Roadmap)

- **HTTP Metrics:** Theo dõi tổng số request, tỷ lệ lỗi và độ trễ $p_{50}, p_{95}$.
- **Model Layer Metrics (Privacy-Safe):**
  - Tỷ lệ Positive / Negative theo thời gian.
  - Phân phối độ dài chuỗi đầu vào và tỷ lệ cắt chuỗi (Truncation rate).
  - Tỷ lệ từ ngoài từ điển (OOV rate).
  - Phân phối điểm tin cậy và số lượng dự đoán rơi vào vùng bất định (`review_required`).
- **Data Drift Detection (Roadmap P2):** Sử dụng kiểm định PSI (Population Stability Index) hoặc Jensen-Shannon Divergence để so sánh phân phối độ dài và OOV giữa dữ liệu online và tập Train tham chiếu.

---

## 15. Đóng Gói Artifact Schema Version 3 & Tính Tái Lập

Checkpoint `model.pt` tuân thủ chuẩn siêu dữ liệu Version 3:

```json
{
  "artifact_schema_version": 3,
  "checkpoint_kind": "release",
  "model_version": "1.0.0",
  "model_type": "bilstm",
  "tokenizer_version": "word-regex-v2",
  "vocabulary_hash": "a4f8...c1e9",
  "source_dataset_hash": "e9b2...4f31",
  "train_split_hash": "...",
  "validation_split_hash": "...",
  "calibration_split_hash": "...",
  "official_test_hash": "...",
  "decision_threshold": 0.5,
  "confidence_threshold": 0.72,
  "temperature": 1.1452,
  "max_length": 256,
  "truncation_strategy": "head_tail",
  "python_version": "3.12.x",
  "torch_version": "2.x",
  "final_fit_epoch": 8,
  "config": { ... },
  "vocabulary": { ... },
  "model_state": { ... }
}
```

---

## 16. Giới Hạn, Ranh Giới Miền & Đạo Đức AI

- **Ranh giới miền dữ liệu (Domain Boundary):** Mô hình được huấn luyện chuyên biệt trên tập dữ liệu đánh giá phim tiếng Anh (IMDB). Không phù hợp cho văn bản đa ngôn ngữ hoặc văn bản ngoài miền đánh giá điện ảnh.
- **Rủi ro phân loại:** Mô hình có thể dự đoán sai trong các trường hợp cảm xúc đảo chiều phức tạp hoặc mỉa mai tinh vi.
- **Khuyến nghị sử dụng:** Phù hợp cho việc phân tích xu hướng đánh giá, portfolio học thuật; không dùng cho việc ra quyết định tự động ảnh hưởng trực tiếp đến quyền lợi con người mà không có sự giám sát.

---

## 17. Lộ Trình Mở Rộng Reference Transformer (Roadmap P2)

- **Modern Benchmark Reference:** Xây dựng nhánh tham chiếu nhẹ nhàng với `DistilBERT` hoặc `MiniLM` để xác định mức trần hiệu năng (performance ceiling) của Transformer hiện đại.
- **Ablation GloVe Embedding:** So sánh giữa Embedding khởi tạo ngẫu nhiên từ đầu với Pretrained GloVe 100d.
- **Thông điệp học thuật:** *Mô hình RNN được lưu giữ như một kiến trúc baseline tuần tự nhẹ nhàng, tiết kiệm tài nguyên và phục vụ mục đích giáo dục chuyên sâu, trong khi mô hình Transformer xác định mốc trần hiệu năng hiện đại.*

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
.
├── sentiment/                  # Core NLP Package
│   ├── __init__.py             # Package initializer
│   ├── config.py               # ExperimentConfig (dataclass, validation, temperature)
│   ├── text.py                 # word-regex-v2, Exact & Normalized SHA256, Vocabulary
│   ├── data_validation.py      # Schema check, Conflicting label, Anti-leakage dedup
│   ├── data.py                 # IMDBDataset, DataLoader, DataBundle & Audit stats
│   ├── model.py                # SentimentRNN production BiLSTM với packed sequence
│   ├── calibration.py          # TemperatureScaler, Brier, ECE, DecisionPolicy
│   ├── engine.py               # Training loop, Early stopping, Comprehensive evaluation
│   ├── inference.py            # SentimentPredictor with calibration & reliability audit
│   ├── artifacts.py            # Artifact Schema v3, plots & reliability diagrams
│   └── utils.py                # Reproducibility seed, device selector, latency benchmark
├── tests/                      # Pytest suite
│   ├── test_invariants.py      # 14 kiểm thử bất biến kiến trúc và anti-leakage
│   ├── test_api.py             # Kiểm thử FastAPI endpoints
│   ├── test_baseline.py        # Kiểm thử TF-IDF Logistic baseline
│   ├── test_compare_models.py  # Kiểm thử Leaderboard generation
│   ├── test_data.py            # Kiểm thử data loading & preprocessing
│   ├── test_inference.py       # Kiểm thử inference & predictions
│   └── ...
├── app.py                      # Streamlit Web Platform đa tab
├── api.py                      # FastAPI REST Server (Production-oriented)
├── train.py                    # CLI huấn luyện & validation (Validation-only)
├── baseline.py                 # CLI huấn luyện TF-IDF baseline & explainability
├── compare_models.py           # CLI tổng hợp Development Leaderboard & chọn Champion
├── evaluate_final.py           # Module evaluator được gọi bởi scripts/evaluate_release.py
├── analyze_errors.py           # Phân tích prediction facts, không đọc lại Test
├── predict.py                  # CLI suy luận nhanh câu đơn
├── load_test.py                # CLI kiểm tra tải đồng thời API
├── scripts/
│   ├── download_imdb.py        # Tải IMDB + checksum + manifest
│   ├── create_smoke_dataset.py # Fixture nhỏ, không phải official data
│   ├── final_fit.py            # Fit Train + Validation với best epoch đã khóa
│   ├── calibrate.py            # Temperature + confidence policy trên Calibration
│   ├── package_baseline_release.py # Đóng gói baseline nếu validation chọn baseline
│   └── evaluate_release.py     # Entry point duy nhất mở Official Test
├── configs/bilstm.yaml         # Cấu hình tham chiếu
├── data/raw/                   # Official CSV, gitignored
├── data/processed/             # Dữ liệu xử lý, gitignored
├── Dockerfile                  # Container hóa REST API
├── MODEL_CARD.md               # Model Card chi tiết
├── DATASET_CARD.md              # Dataset Card & Anti-leakage audit
└── requirements.txt            # Thư viện phụ thuộc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### 1. Cài đặt môi trường
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Kích hoạt môi trường (macOS / Linux)
source .venv/bin/activate

# Cài đặt thư viện runtime và dev
python -m pip install -e ".[dev]"
```

### 2. Chuẩn bị Official IMDB data

Không đặt `train.csv`/`test.csv` ở root và không dùng smoke data thay cho IMDB.
Tải archive chính thức, truyền checksum đã xác minh, rồi script sẽ ghi vào
`data/raw/` và tạo `data/manifest.json`:

```bash
python -m scripts.download_imdb --sha256 "<SHA256_OFFICIAL_ARCHIVE>"
```

Nếu chỉ cần kiểm tra pipeline bằng dữ liệu nhỏ:

```bash
python -m scripts.create_smoke_dataset
```

### 3. Chạy bộ kiểm thử và static checks
```bash
python -m pytest -q
python -c "from pathlib import Path; roots=[Path('sentiment'),Path('scripts'),Path('tests')]; files=list(Path('.').glob('*.py'))+[p for root in roots for p in root.rglob('*.py')]; [compile(p.read_text(encoding='utf-8'),str(p),'exec') for p in files]"
python -m ruff check sentiment train.py baseline.py compare_models.py evaluate_final.py analyze_errors.py scripts tests --select E4,E7,E9,F --ignore E402
```

### 4. Huấn luyện First-Class Baseline (TF-IDF + Logistic Regression)
```bash
python baseline.py --output-dir artifacts/baseline
```

### 5. Huấn luyện candidate BiLSTM (Validation-Only Protocol)
```bash
# Huấn luyện BiLSTM trên Train/Validation/Calibration; không có test flag
python train.py --epochs 15 --batch-size 128 --output-dir runs/dev_bilstm
```

### 6. So sánh trên Validation & Chọn Champion
```bash
python compare_models.py
```
*Kết quả xuất tại `artifacts/development_leaderboard.md`.*

### 7. Final fit, calibration và mở Locked Test
```bash
python -m scripts.final_fit
python -m scripts.calibrate
python -m scripts.evaluate_release
```
*Kết quả release nằm tại `artifacts/releases/v1.0.0/`; Official Test chỉ được đọc ở bước cuối.*

### 8. Phân tích lỗi sau release
```bash
python analyze_errors.py --evaluation-records artifacts/releases/v1.0.0/evaluation_records.csv
```

### 9. Khởi chạy Web App & REST API
```bash
# Khởi chạy giao diện Streamlit
streamlit run app.py

# Khởi chạy REST API Server
uvicorn api:app --reload --port 8000
```

---

## ✅ CI và Repo Cleanliness

GitHub Actions tại `.github/workflows/quality.yml` chạy trên Python 3.11:

1. cài project cùng dev dependencies từ `pyproject.toml`;
2. chạy `ruff check` trên source, scripts và tests;
3. chạy pytest với `addopts` rỗng để CI không phụ thuộc cache local;
4. chạy compileall để bắt lỗi syntax/import cơ bản.

Các thư mục sinh ra như `.pytest_tmp/`, `runs/`, `data/raw/`,
`data/processed/` và artifacts thử nghiệm không được commit. Official CSV chỉ
được lưu ở `data/raw/` sau khi checksum và manifest đã được ghi nhận.

## 📝 Giấy Phép & Tác Giả

- **Tác giả:** AI / MLOps Engineer Candidate
- **Giấy phép:** MIT License. Sử dụng tự do cho mục đích nghiên cứu, học tập và xây dựng Portfolio chuyên nghiệp.
