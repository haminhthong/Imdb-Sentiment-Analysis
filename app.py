"""Giao diện Web Demo phân loại cảm xúc đánh giá phim bằng Streamlit."""

import json
import os
from pathlib import Path

import streamlit as st

from sentiment.inference import Predictor, load_predictor

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "artifacts/model.pt")
BASELINE_EXPLAINABILITY = Path("artifacts/baseline/explainability.json")

st.set_page_config(
    page_title="CineSentiment — IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_predictor(path: str) -> Predictor | None:
    """Nạp checkpoint mô hình vào cache."""
    try:
        if Path(path).is_file():
            return load_predictor(path, device="cpu")
    except Exception as exc:
        st.sidebar.error(f"Lỗi khi nạp mô hình: {exc}")
    return None


@st.cache_data
def get_explainability_data() -> dict | None:
    """Đọc dữ liệu top n-grams của baseline nếu có."""
    if BASELINE_EXPLAINABILITY.is_file():
        try:
            return json.loads(BASELINE_EXPLAINABILITY.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# Sidebar
with st.sidebar:
    st.markdown("# 🎬 CineSentiment")
    st.caption("IMDB Sentiment Analysis & Model Evaluation")
    st.markdown("---")

    predictor = get_predictor(CHECKPOINT_PATH)
    if predictor is not None:
        st.success("✅ Mô hình đã sẵn sàng")
        st.write(f"**Kiến trúc:** `{predictor.config.model_type.upper()}`")
        if hasattr(predictor, "vocabulary"):
            st.write(f"**Từ điển:** `{len(predictor.vocabulary):,}` từ")
            st.write(f"**Tham số:** `{predictor.model.count_parameters():,}` params")
            st.write(f"**Temperature:** `{predictor.temperature:.4f}`")
    else:
        st.warning(f"⚠️ Chưa tìm thấy checkpoint `{CHECKPOINT_PATH}`.")
        st.info("Chạy `python train.py` để huấn luyện mô hình.")

    st.markdown("---")
    st.markdown("### 💡 Về Dự Án")
    st.markdown(
        "- **Baseline:** TF-IDF + Logistic Regression\n"
        "- **Deep Model:** BiLSTM với `pack_padded_sequence`\n"
        "- **Hiệu chuẩn:** Temperature Scaling ($T > 0$)\n"
        "- **Chống rò rỉ:** Từ điển chỉ xây dựng từ tập Train"
    )

# Giao diện chính
st.title("🎬 Phân Tích Cảm Xúc Phim (IMDB)")
st.markdown("Nhập nội dung đánh giá phim (tiếng Anh) để dự đoán cảm xúc Tích cực hay Tiêu cực.")

# Nút mẫu nhanh
col1, col2, col3 = st.columns(3)
sample_text = ""
if col1.button("✨ Mẫu Tích Cực"):
    sample_text = (
        "This movie is an absolute masterpiece! Captivating performances, "
        "brilliant cinematography, and a thrilling soundtrack."
    )
if col2.button("⚠️ Mẫu Hỗn Hợp / Phủ Định"):
    sample_text = (
        "The performances were great and visuals were okay, but the ending was "
        "completely awful and ruined the entire movie."
    )
if col3.button("🗑️ Xóa"):
    sample_text = ""

review_input = st.text_area(
    "Nội dung đánh giá:",
    value=sample_text,
    placeholder="Enter your movie review here...",
    height=150,
)

if st.button("🚀 Phân Tích Cảm Xúc", type="primary", use_container_width=True):
    if not review_input.strip():
        st.warning("Vui lòng nhập văn bản đánh giá trước khi phân tích.")
    elif predictor is None:
        st.error(f"Không tìm thấy checkpoint mô hình tại `{CHECKPOINT_PATH}`.")
    else:
        try:
            result = predictor.predict(review_input)

            # Hiển thị kết quả trực quan
            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                if result.label == "Positive":
                    st.success("### 😃 POSITIVE\n(Tích Cực)")
                else:
                    st.error("### 🙁 NEGATIVE\n(Tiêu Cực)")

            with res_col2:
                st.metric("Xác Suất Hiệu Chuẩn", f"{result.probability:.1%}")
                st.progress(result.probability)

            with res_col3:
                st.metric("Số Lượng Token", result.token_count)
                st.metric("Tỷ Lệ OOV", f"{result.oov_rate:.1%}")

            # Cảnh báo rủi ro (nếu có)
            if result.truncated:
                st.warning("⚠️ Văn bản vượt quá độ dài tối đa (256 tokens) và đã được cắt ngắn.")
            if result.oov_rate > 0.20:
                st.warning("⚠️ Tỷ lệ từ ngoài từ điển cao (>20%), độ tin cậy có thể bị ảnh hưởng.")
            if "OUT_OF_DOMAIN_LANGUAGE_HEURISTIC" in result.warnings:
                st.info("ℹ️ Văn bản có thể chứa các từ ngoài miền đánh giá điện ảnh tiếng Anh.")

        except Exception as exc:
            st.error(f"Lỗi trong quá trình suy luận: {exc}")

# Phần Explainability: Top n-grams từ Baseline
explain_data = get_explainability_data()
if explain_data:
    st.markdown("---")
    with st.expander("🔍 Khám phá tính giải thích (Lexical Signals từ Baseline TF-IDF)"):
        st.caption(
            "Mô hình Logistic Regression phân tích các n-gram có trọng số đóng góp "
            "mạnh nhất vào chiều hướng cảm xúc:"
        )
        pos_col, neg_col = st.columns(2)

        with pos_col:
            st.markdown("**Top N-grams Tích Cực:**")
            for item in explain_data.get("top_positive_ngrams", [])[:8]:
                st.write(f"• `{item['ngram']}` (+{item['weight']:.2f})")

        with neg_col:
            st.markdown("**Top N-grams Tiêu Cực:**")
            for item in explain_data.get("top_negative_ngrams", [])[:8]:
                st.write(f"• `{item['ngram']}` ({item['weight']:.2f})")
