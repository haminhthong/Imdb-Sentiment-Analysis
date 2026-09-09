"""Giao diện Web Demo đa năng cho CineSentiment Platform sử dụng Streamlit.

Ứng dụng cung cấp 3 tab tương tác:
1. Tab 1: Phân tích đánh giá đơn (Single Review) với xác suất hiệu chuẩn, vùng bất định và kiểm toán token.
2. Tab 2: Phân tích theo lô (Batch Processing) kèm cờ cảnh báo rủi ro độ tin cậy.
3. Tab 3: Sơ đồ 9 giai đoạn chuẩn mực, Development Leaderboard và Báo cáo Locked Final Test.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment.inference import Predictor, load_predictor as load_model

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "artifacts/releases/v1.0.0/model.pt")
ARTIFACT_DIR = Path(CHECKPOINT_PATH).parent
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="CineSentiment — Sentiment Intelligence Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor(path: str) -> Predictor:
    """Nạp và caching model.pt hoặc model.joblib để tối ưu hiệu năng."""
    return load_model(path, device="cpu")


def show_unexpected_error(context: str) -> None:
    LOGGER.exception("Lỗi Streamlit tại %s", context)
    st.error("Hệ thống gặp lỗi ngoài dự kiến. Vui lòng thử lại sau.")


# Sidebar thông tin dự án
with st.sidebar:
    st.markdown("# 🎬")
    st.title("CineSentiment")
    st.caption("Leakage-Safe Sentiment Intelligence & NLP Benchmarking Platform")

    st.markdown("---")
    st.markdown("### 📌 Trạng Thái Checkpoint")
    if Path(CHECKPOINT_PATH).is_file():
        try:
            predictor = load_predictor(CHECKPOINT_PATH)
            st.success(f"**Mô hình:** `{predictor.config.model_type.upper()}`")
            st.info(f"**Từ điển:** `{len(predictor.vocabulary):,}` tokens")
            st.info(f"**Tham số:** `{predictor.model.count_parameters():,}` params")
            st.info(f"**Temperature:** `{predictor.temperature:.4f}`")
            st.info(f"**Ngưỡng quyết định:** `{predictor.decision_threshold:.2f}`")
        except Exception:
            show_unexpected_error("nạp checkpoint sidebar")
    else:
        st.warning("⚠️ Chưa tìm thấy checkpoint mặc định! Hãy chạy `train.py` để huấn luyện.")

    st.markdown("---")
    st.markdown("### 🏛 9 Giai Đoạn Canonical")
    st.markdown(
        "1. Data Ingestion\n2. Quality & Anti-Leakage\n3. Dev Split (Test locked)\n4. Train-Only Text Contract\n5. Model Development\n6. Calibration Layer\n7. Locked Final Test\n8. Model Packaging (v3)\n9. Online Serving"
    )


# Tiêu đề ứng dụng
st.title("🎬 CineSentiment — Sentiment Intelligence Platform")
st.markdown(
    "Nền tảng phân tích cảm xúc đánh giá phim chuẩn mực: **Leakage-Safe NLP Benchmarking, "
    "Calibrated Prediction & Production-Oriented Serving**."
)

tab_single, tab_batch, tab_info = st.tabs(
    ["📝 Phân Tích Câu Đơn", "📊 Phân Tích Theo Lô (Batch)", "⚙️ Kiến Trúc & Benchmark"]
)

# ==============================================================================
# TAB 1: PHÂN TÍCH CÂU ĐƠN
# ==============================================================================
with tab_single:
    st.subheader("Nhập văn bản đánh giá phim (Tiếng Anh)")

    sample_col1, sample_col2, sample_col3 = st.columns(3)
    preset_text = ""
    if sample_col1.button("✨ Mẫu Tích Cực"):
        preset_text = (
            "This movie is an absolute masterpiece! Captivating performances, "
            "brilliant cinematography, and a thrilling soundtrack."
        )
    if sample_col2.button("⚠️ Mẫu Phủ Định / Hỗn Hợp"):
        preset_text = (
            "The performances were great and visuals were okay, but the ending was "
            "completely awful and ruined the whole experience."
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

                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    if result.label == "Positive":
                        st.success("### 😃 POSITIVE\n(Tích Cực)")
                    else:
                        st.error("### 🙁 NEGATIVE\n(Tiêu Cực)")

                with res_col2:
                    st.metric("Xác Suất Hiệu Chuẩn", f"{result.positive_probability:.1%}")
                    st.caption(f"Raw score: {result.positive_score:.4f}")

                with res_col3:
                    st.metric("Độ Tin Cậy", f"{result.confidence:.1%}")

                with res_col4:
                    if result.uncertain:
                        st.warning("### ⚠️ REVIEW REQUIRED\n(Vùng Bất Định)")
                    else:
                        st.info("### ✅ ACCEPTED\n(Quyết Định Tin Cậy)")

                # Hiển thị kiểm toán token
                st.markdown("#### 🔍 Kiểm Toán Chuỗi Đầu Vào (Reliability Audit)")
                aud_col1, aud_col2, aud_col3, aud_col4 = st.columns(4)
                aud_col1.metric("Tổng Token Gốc", result.input_tokens)
                aud_col2.metric("Token Đã Dùng", result.used_tokens)
                aud_col3.metric("OOV Đã Dùng", f"{result.used_oov_rate:.1%}")
                aud_col4.metric("Cắt Chuỗi (Truncated)", "Có" if result.truncated else "Không")

                if result.warnings:
                    st.warning(f"⚠️ Cảnh báo độ tin cậy phát hiện: {', '.join(result.warnings)}")

            except FileNotFoundError as err:
                st.error(str(err))
            except Exception:
                show_unexpected_error("dự đoán câu đơn")

# ==============================================================================
# TAB 2: PHÂN TÍCH THEO LÔ (BATCH)
# ==============================================================================
with tab_batch:
    st.subheader("Phân Tích Danh Sách Đánh Giá Hàng Loạt")
    st.caption("Nhập danh sách các câu đánh giá (mỗi câu một dòng).")

    batch_input = st.text_area(
        "Danh sách câu đánh giá:",
        placeholder="Awesome movie with stunning visuals!\nAwful script and boring actors.\nNot bad at all, quite entertaining.",
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
                        "Nội dung": text[:80] + "..." if len(text) > 80 else text,
                        "Nhãn (Label)": res.label,
                        "Quyết Định": "⚠️ Review" if res.uncertain else "✅ Accepted",
                        "Xác Suất Hiệu Chuẩn": f"{res.positive_probability:.2%}",
                        "Raw Score": f"{res.positive_score:.2%}",
                        "Tokens": f"{res.used_tokens}/{res.input_tokens}",
                        "OOV Rate (used)": f"{res.used_oov_rate:.1%}",
                        "Cảnh Báo": ", ".join(res.warnings) if res.warnings else "None",
                    }
                    for idx, (text, res) in enumerate(zip(lines, results))
                ]

                df_res = pd.DataFrame(data_records)
                st.dataframe(df_res, use_container_width=True)

                pos_count = sum(1 for r in results if r.label == "Positive")
                uncertain_count = sum(1 for r in results if r.uncertain)
                st.info(
                    f"📊 **Tổng số:** {len(results)} mẫu | "
                    f"**Positive:** {pos_count} | **Negative:** {len(results) - pos_count} | "
                    f"**Vùng Bất Định (Review Required):** {uncertain_count}"
                )

            except Exception:
                show_unexpected_error("dự đoán batch")

# ==============================================================================
# TAB 3: THÔNG TIN KIẾN TRÚC & BENCHMARK
# ==============================================================================
with tab_info:
    st.subheader("Kiến Trúc Nền Tảng & Báo Cáo Benchmark Độc Lập")

    # Hiển thị Development Leaderboard nếu có
    leaderboard_file = Path("artifacts/development_leaderboard.md")
    if leaderboard_file.is_file():
        st.markdown("### 🏆 Development Validation Leaderboard")
        st.markdown(leaderboard_file.read_text(encoding="utf-8"))
        st.markdown("---")

    # Hiển thị Locked Final Test Report nếu có
    final_test_file = ARTIFACT_DIR / "final_test_report.md"
    if final_test_file.is_file():
        st.markdown("### 🔒 Locked Final Test Report (Champion Only)")
        st.markdown(final_test_file.read_text(encoding="utf-8"))
        st.markdown("---")

    # Biểu đồ artifacts
    st.markdown("### 📈 Biểu Đồ Huấn Luyện & Hiệu Chuẩn")
    chart_col1, chart_col2 = st.columns(2)
    hist_img = ARTIFACT_DIR / "training_history.png"
    cm_img = ARTIFACT_DIR / "confusion_matrix.png"

    with chart_col1:
        if hist_img.is_file():
            st.image(str(hist_img), caption="Lịch sử Mất mát & Độ chính xác", use_column_width=True)
        else:
            st.info("Chưa có đồ thị training_history.png.")

    with chart_col2:
        if cm_img.is_file():
            st.image(
                str(cm_img), caption="Ma trận nhầm lẫn (Confusion Matrix)", use_column_width=True
            )
        else:
            st.info("Chưa có đồ thị confusion_matrix.png.")
