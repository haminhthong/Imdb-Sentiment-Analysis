"""Giao diện Web Demo đa năng cho CineSentiment AI sử dụng Streamlit.

Ứng dụng cung cấp 3 tab tương tác:
1. Tab 1: Phân tích đánh giá đơn (Single Review) với thanh đo độ tin cậy và xem mẫu có sẵn.
2. Tab 2: Phân tích theo lô (Batch Processing) cho danh sách nhiều câu đánh giá.
3. Tab 3: Thông tin kiến trúc mô hình & Bảng so sánh chỉ số Benchmark.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment.inference import SentimentPredictor

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "artifacts/bilstm/model.pt")
ARTIFACT_DIR = Path(CHECKPOINT_PATH).parent
LOGGER = logging.getLogger(__name__)

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="CineSentiment AI - Deep Learning Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor(path: str) -> SentimentPredictor:
    """Nạp và caching đối tượng SentimentPredictor để tối ưu hiệu năng."""
    return SentimentPredictor(path, device="cpu")


def show_unexpected_error(context: str) -> None:
    """Ghi chi tiết vào server log nhưng không làm lộ lỗi nội bộ trên giao diện."""
    LOGGER.exception("Lỗi Streamlit tại %s", context)
    st.error("Hệ thống gặp lỗi ngoài dự kiến. Vui lòng thử lại sau.")


# Sidebar thông tin dự án
with st.sidebar:
    st.markdown("# 🎬")
    st.title("CineSentiment AI")
    st.caption("PyTorch Deep Learning Platform")

    st.markdown("---")
    st.markdown("### 📌 Thông tin Checkpoint")
    if Path(CHECKPOINT_PATH).is_file():
        try:
            predictor = load_predictor(CHECKPOINT_PATH)
            st.success(f"**Mô hình:** `{predictor.config.model_type.upper()}`")
            st.info(f"**Từ điển:** `{len(predictor.vocabulary):,}` tokens")
            st.info(f"**Tham số:** `{predictor.model.count_parameters():,}` params")
        except Exception:
            show_unexpected_error("nạp checkpoint sidebar")
    else:
        st.warning("⚠️ Chưa tìm thấy checkpoint mặc định! Hãy chạy `train.py` để huấn luyện.")

    st.markdown("---")
    st.markdown("### 🛠 Công nghệ sử dụng")
    st.markdown("- **Framework:** PyTorch, FastAPI, Streamlit")
    st.markdown("- **RNN Models:** BiLSTM, GRU, LSTM")
    st.markdown("- **MLOps:** Docker, GitHub Actions CI")


# Tiêu đề ứng dụng
st.title("🎬 CineSentiment AI — Phân Tích Cảm Xúc Đánh Giá Phim")
st.markdown(
    "Hệ thống Deep Learning phân loại đánh giá phim tiếng Anh thành **Positive (Tích cực)** "
    "hoặc **Negative (Tiêu cực)** sử dụng PyTorch Recurrent Neural Networks."
)

# Khởi tạo 3 Tab ứng dụng
tab_single, tab_batch, tab_info = st.tabs(
    ["📝 Phân Tích Câu Đơn", "📊 Phân Tích Theo Lô (Batch)", "⚙️ Kiến Trúc & Benchmark"]
)

# ==============================================================================
# TAB 1: PHÂN TÍCH CÂU ĐƠN
# ==============================================================================
with tab_single:
    st.subheader("Nhập văn bản đánh giá phim (Tiếng Anh)")

    # Ví dụ mẫu để trải nghiệm nhanh
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    preset_text = ""
    if sample_col1.button("✨ Mẫu Tích Cực 1"):
        preset_text = (
            "This movie is an absolute masterpiece! Captivating performances, "
            "brilliant cinematography, and a thrilling soundtrack."
        )
    if sample_col2.button("⚠️ Mẫu Tiêu Cực 1"):
        preset_text = (
            "Terrible film. The storyline was completely boring, acting was dry, "
            "and it felt like a total waste of money."
        )
    if sample_col3.button("🔄 Reset"):
        preset_text = ""

    review_input = st.text_area(
        "Nội dung đánh giá:",
        value=preset_text,
        placeholder="Enter your movie review here...",
        height=160,
        max_chars=5_000,
    )

    if st.button("🚀 Phân Tích Cảm Xúc", type="primary", use_container_width=True):
        if not review_input.strip():
            st.warning("Vui lòng nhập nội dung câu đánh giá trước khi phân tích.")
        else:
            try:
                predictor = load_predictor(CHECKPOINT_PATH)
                result = predictor.predict(review_input)

                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    if result.label == "Positive":
                        st.success(f"### 😃 POSITIVE\n(Tích Cực)")
                    else:
                        st.error(f"### 🙁 NEGATIVE\n(Tiêu Cực)")

                with res_col2:
                    st.metric("Độ Tin Cậy (Confidence)", f"{result.confidence:.1%}")

                with res_col3:
                    st.metric("Xác Suất Positive", f"{result.positive_probability:.1%}")

                st.progress(
                    result.positive_probability,
                    text=f"Mức độ tích cực: {result.positive_probability:.1%}",
                )

            except FileNotFoundError as err:
                st.error(str(err))
            except Exception:
                show_unexpected_error("dự đoán câu đơn")

# ==============================================================================
# TAB 2: PHÂN TÍCH THEO LÔ (BATCH)
# ==============================================================================
with tab_batch:
    st.subheader("Phân Tích Danh Sách Nhiều Câu Đánh Giá")
    st.caption("Nhập nhiều câu (mỗi câu một dòng) để dự đoán hàng loạt.")

    batch_input = st.text_area(
        "Danh sách câu đánh giá:",
        placeholder="Awesome movie!\nBoring script and bad acting.\nI loved the climax scene!",
        height=200,
    )

    if st.button("⚡ Phân Tích Batch", type="primary", use_container_width=True):
        lines = [line.strip() for line in batch_input.splitlines() if line.strip()]
        if not lines:
            st.warning("Vui lòng nhập ít nhất một câu đánh giá.")
        elif len(lines) > 100:
            st.warning("Mỗi lần chỉ xử lý tối đa 100 câu đánh giá.")
        else:
            try:
                predictor = load_predictor(CHECKPOINT_PATH)
                results = predictor.predict_batch(lines)

                data_records = [
                    {
                        "STT": idx + 1,
                        "Nội dung câu": text,
                        "Cảm xúc (Label)": res.label,
                        "Xác suất Positive": f"{res.positive_probability:.2%}",
                        "Độ tin cậy": f"{res.confidence:.2%}",
                    }
                    for idx, (text, res) in enumerate(zip(lines, results))
                ]

                df_res = pd.DataFrame(data_records)
                st.dataframe(df_res, use_container_width=True)

                # Thống kê tổng quan batch
                pos_count = sum(1 for r in results if r.label == "Positive")
                neg_count = len(results) - pos_count
                st.info(
                    f"📊 **Tổng số:** {len(results)} câu | "
                    f"**Positive:** {pos_count} | **Negative:** {neg_count}"
                )

            except Exception:
                show_unexpected_error("dự đoán batch")

# ==============================================================================
# TAB 3: THÔNG TIN KIẾN TRÚC & BENCHMARK
# ==============================================================================
with tab_info:
    st.subheader("Tổng Quan Kiến Trúc & Benchmark Dự Án")

    st.markdown(
        """
        - **Pipeline Tái Lập (Reproducible Pipeline):** Khởi tạo seed ngẫu nhiên
          cho Python, NumPy và PyTorch cuDNN.
        - **Chống Rò Rỉ Dữ Liệu (Anti-Leakage):** Vocabulary chỉ xây dựng từ
          tập Train. Mẫu Test lặp trong Train bị loại bỏ hoàn toàn.
        - **Tối Ưu Sequence Padding:** Dùng `pack_padded_sequence` để RNN chỉ
          tính toán trên độ dài câu thực tế.
        - **Chống Overfitting:** Dùng AdamW, L2 Weight Decay, Dropout và Early
          Stopping tự động khôi phục checkpoint tốt nhất.
        """
    )

    st.markdown("---")
    st.markdown("### 📈 Biểu Đồ Huấn Luyện (Training Artifacts)")

    chart_col1, chart_col2 = st.columns(2)
    hist_img = ARTIFACT_DIR / "training_history.png"
    cm_img = ARTIFACT_DIR / "confusion_matrix.png"

    with chart_col1:
        if hist_img.is_file():
            st.image(str(hist_img), caption="Lịch sử Mất mát & Độ chính xác", use_column_width=True)
        else:
            st.info("Chưa có đồ thị training_history.png. Hãy chạy `train.py` để tạo đồ thị.")

    with chart_col2:
        if cm_img.is_file():
            st.image(
                str(cm_img),
                caption="Ma trận nhầm lẫn (Confusion Matrix)",
                use_column_width=True,
            )
        else:
            st.info("Chưa có đồ thị confusion_matrix.png.")
