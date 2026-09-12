"""Giao diện Web Demo phân loại cảm xúc đánh giá phim CineSentiment bằng Streamlit."""

import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment.inference import Predictor, load_predictor
from sentiment.text import tokenize

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "artifacts/model.pt")
BASELINE_MODEL_PATH = Path("artifacts/baseline/model.joblib")
BASELINE_EXPLAINABILITY = Path("artifacts/baseline/explainability.json")
VAL_METRICS_PATH = Path("artifacts/validation_metrics.json")
TEST_METRICS_PATH = Path("artifacts/test_metrics.json")
ERROR_ANALYSIS_PATH = Path("artifacts/error_analysis.json")
COMPARISON_MD_PATH = Path("artifacts/model_comparison.md")

st.set_page_config(
    page_title="CineSentiment AI — IMDB Sentiment & Reliability",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Cinematic Theme & Glassmorphism)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

.cine-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.cine-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.cine-subtitle {
    color: #94a3b8;
    font-size: 1.02rem;
    margin-bottom: 0;
}

.metric-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: center;
}

.metric-label {
    font-size: 0.82rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 4px;
}

.sentiment-badge-pos {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    color: #34d399;
}

.sentiment-badge-neg {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.25));
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    color: #f87171;
}

.token-chip {
    display: inline-block;
    padding: 3px 8px;
    margin: 3px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
}

.token-chip-normal {
    background: rgba(148, 163, 184, 0.12);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.2);
}

.token-chip-unk {
    background: rgba(239, 68, 68, 0.2);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.5);
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_model(path: str) -> Predictor | None:
    """Nạp predictor mô hình vào bộ nhớ đệm."""
    try:
        p = Path(path)
        if p.is_file():
            return load_predictor(p, device="cpu")
    except Exception:
        return None
    return None


@st.cache_data
def load_json_artifact(path: Path) -> dict | None:
    """Đọc tệp JSON từ thư mục artifacts."""
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


bilstm_predictor = get_model(CHECKPOINT_PATH)
baseline_predictor = get_model(str(BASELINE_MODEL_PATH))
explain_data = load_json_artifact(BASELINE_EXPLAINABILITY)
error_data = load_json_artifact(ERROR_ANALYSIS_PATH)
val_metrics = load_json_artifact(VAL_METRICS_PATH)
test_metrics = load_json_artifact(TEST_METRICS_PATH)

# Sidebar
with st.sidebar:
    st.markdown("### 🎬 CineSentiment AI")
    st.caption("Phân loại cảm xúc phim & Đánh giá độ tin cậy xác suất")
    st.markdown("---")

    st.markdown("#### ⚙️ Trạng Thái Checkpoints")
    if bilstm_predictor is not None:
        st.success("🟢 **BiLSTM (Calibrated):** Đã nạp")
        temp_val = getattr(bilstm_predictor, "temperature", 1.0)
        vocab_size = (
            len(bilstm_predictor.vocabulary) if hasattr(bilstm_predictor, "vocabulary") else 0
        )
        param_count = (
            bilstm_predictor.model.count_parameters()
            if hasattr(bilstm_predictor, "model")
            else 658_049
        )
        st.caption(
            f"• Từ điển: `{vocab_size:,}` từ\n"
            f"• Tham số: `{param_count:,}`\n"
            f"• Nhiệt độ T: `{temp_val:.4f}`"
        )
    else:
        st.error(f"🔴 **BiLSTM:** Thiếu `{CHECKPOINT_PATH}`")

    if baseline_predictor is not None:
        st.success("🟢 **TF-IDF + LR:** Đã nạp")
    else:
        st.warning(f"🟡 **Baseline:** Thiếu `{BASELINE_MODEL_PATH}`")

    st.markdown("---")
    st.markdown("#### 🎯 Điểm Sáng Kỹ Thuật")
    st.markdown(
        "- **Chống rò rỉ (Leakage Prevention):** Từ điển chỉ xây từ tập Train (80%).\n"
        "- **Hiệu chuẩn xác suất:** Temperature Scaling $T > 0$ tối ưu trên Calibration set.\n"
        "- **BiLSTM 2 chiều:** Đóng gói bằng `pack_padded_sequence` tăng tốc.\n"
        "- **Audit Token & OOV:** Cảnh báo sớm các mẫu suy luận ngoài miền."
    )

    st.markdown("---")
    st.caption("Phiên bản v1.0.0 • CineSentiment Production")

# Header Section
st.markdown(
    """
    <div class="cine-header">
        <div class="cine-title">CineSentiment AI Studio</div>
        <p class="cine-subtitle">
            Hệ thống phân loại cảm xúc đánh giá phim IMDB chuẩn công nghiệp:
            mô hình BiLSTM kết hợp hiệu chuẩn nhiệt độ và đối chiếu Baseline TF-IDF.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs navigation
tab_live, tab_compare, tab_errors, tab_batch, tab_arch = st.tabs(
    [
        "🧪 Live Sentiment Lab",
        "⚖️ So Sánh & Hiệu Chuẩn",
        "🔍 Phân Tích Lát Cắt Lỗi",
        "📦 Kiểm Thử Hàng Loạt",
        "📐 Kiến Trúc & REST API",
    ]
)

# ==============================================================================
# TAB 1: LIVE SENTIMENT LAB
# ==============================================================================
with tab_live:
    st.markdown("### 🧪 Thử Nghiệm Dự Đoán Trực Tiếp")
    st.write(
        "Nhập đánh giá phim tiếng Anh hoặc chọn một trong các mẫu kiểm thử bên dưới để "
        "quan sát so sánh thời gian thực giữa **BiLSTM Calibrated** và **Baseline TF-IDF**."
    )

    sample_library = {
        "masterpiece": (
            "This movie is an absolute masterpiece! Captivating performances, brilliant "
            "cinematography, and a thrilling soundtrack that stays with you forever."
        ),
        "mixed_ending": (
            "The performances were great and visuals were okay, but the ending was completely "
            "awful and ruined the entire movie experience for me."
        ),
        "negation_trick": (
            "I cannot say that I didn't enjoy the film, because the acting wasn't bad at all."
        ),
        "harsh_critic": (
            "Terrible script, zero character development, "
            "and incredibly boring dialogue throughout."
        ),
        "oov_heavy": (
            "The protagonist exhibited excessive schadenfreude and solipsistic megalomania in the "
            "cinematographic avant-garde portrayal."
        ),
    }

    if "input_review" not in st.session_state:
        st.session_state["input_review"] = sample_library["masterpiece"]

    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5, col_btn_clr = st.columns(6)
    if col_btn1.button("✨ Kiệt Tác", use_container_width=True):
        st.session_state["input_review"] = sample_library["masterpiece"]
    if col_btn2.button("⚠️ Hỗn Hợp", use_container_width=True):
        st.session_state["input_review"] = sample_library["mixed_ending"]
    if col_btn3.button("🔄 Phủ Định", use_container_width=True):
        st.session_state["input_review"] = sample_library["negation_trick"]
    if col_btn4.button("💥 Phê Bình", use_container_width=True):
        st.session_state["input_review"] = sample_library["harsh_critic"]
    if col_btn5.button("🧬 Từ Lạ (OOV)", use_container_width=True):
        st.session_state["input_review"] = sample_library["oov_heavy"]
    if col_btn_clr.button("🗑️ Xóa Trắng", use_container_width=True):
        st.session_state["input_review"] = ""

    review_text = st.text_area(
        "Nội dung bài đánh giá:",
        value=st.session_state["input_review"],
        height=120,
        placeholder="Type or paste your English movie review here...",
        key="review_area",
    )

    if st.button("🚀 Chạy Phân Tích Song Song", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("⚠️ Vui lòng nhập nội dung đánh giá trước khi phân tích.")
        elif bilstm_predictor is None:
            st.error(f"⚠️ Không tìm thấy checkpoint mô hình tại `{CHECKPOINT_PATH}`.")
        else:
            try:
                bilstm_res = bilstm_predictor.predict(review_text)
                baseline_res = (
                    baseline_predictor.predict(review_text)
                    if baseline_predictor is not None
                    else None
                )

                st.markdown("---")
                col_m1, col_m2 = st.columns(2)

                with col_m1:
                    st.markdown("#### 🧠 BiLSTM (Đã Hiệu Chuẩn Xác Suất)")
                    prob_text = f"{bilstm_res.probability:.2%}"
                    if bilstm_res.label == "Positive":
                        st.markdown(
                            f"""
                            <div class="sentiment-badge-pos">
                                <h2 style="margin:0; color:#34d399;">😃 POSITIVE (TÍCH CỰC)</h2>
                                <p style="margin:4px 0 0 0; font-size:1.1rem; font-weight:600;">
                                    Xác suất: <strong>{prob_text}</strong>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="sentiment-badge-neg">
                                <h2 style="margin:0; color:#f87171;">🙁 NEGATIVE (TIÊU CỰC)</h2>
                                <p style="margin:4px 0 0 0; font-size:1.1rem; font-weight:600;">
                                    Xác suất: <strong>{prob_text}</strong>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(bilstm_res.probability)
                    temp_val = getattr(bilstm_predictor, "temperature", 1.0)
                    st.caption(f"Nhiệt độ $T = {temp_val:.4f}$ • Ngưỡng quyết định = 0.50")

                with col_m2:
                    st.markdown("#### 📊 Baseline (TF-IDF + Logistic Regression)")
                    if baseline_res is not None:
                        if baseline_res.label == "Positive":
                            st.markdown(
                                f"""
                                <div class="sentiment-badge-pos">
                                    <h2 style="margin:0; color:#34d399;">😃 POSITIVE (TÍCH CỰC)</h2>
                                    <p style="margin:4px 0 0 0; font-size:1.1rem; font-weight:600;">
                                        Xác suất: <strong>{baseline_res.probability:.2%}</strong>
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="sentiment-badge-neg">
                                    <h2 style="margin:0; color:#f87171;">🙁 NEGATIVE (TIÊU CỰC)</h2>
                                    <p style="margin:4px 0 0 0; font-size:1.1rem; font-weight:600;">
                                        Xác suất: <strong>{baseline_res.probability:.2%}</strong>
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.progress(baseline_res.probability)
                        st.caption("Mô hình tuyến tính dựa trên tần suất n-grams (1-2)")
                    else:
                        st.info("Chưa nạp được checkpoint Baseline.")

                # Metrics summary bar
                st.markdown("<br>", unsafe_allow_html=True)
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Tổng Tokens</div>
                            <div class="metric-value">{bilstm_res.token_count}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with stat_col2:
                    oov_color = "#ef4444" if bilstm_res.oov_rate > 0.20 else "#10b981"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Tỷ Lệ OOV</div>
                            <div class="metric-value" style="color:{oov_color};">
                                {bilstm_res.oov_rate:.1%}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with stat_col3:
                    trunc_val = "CÓ (256)" if bilstm_res.truncated else "KHÔNG"
                    trunc_color = "#f59e0b" if bilstm_res.truncated else "#10b981"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Bị Cắt Ngắn</div>
                            <div class="metric-value" style="color:{trunc_color};">{trunc_val}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with stat_col4:
                    warn_count = len(bilstm_res.warnings)
                    warn_color = "#10b981" if warn_count == 0 else "#f59e0b"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Cảnh Báo Độ Tin Cậy</div>
                            <div class="metric-value" style="color:{warn_color};">{warn_count}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Diagnostic Warnings Box
                if bilstm_res.warnings or bilstm_res.truncated:
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.container():
                        st.markdown("##### ⚠️ Cảnh Báo Chẩn Đoán Dữ Liệu Đầu Vào:")
                        if bilstm_res.truncated:
                            st.warning(
                                "• **Văn bản bị cắt ngắn (Truncated):** Câu chứa hơn 256 tokens. "
                                "Mô hình giữ nửa đầu và nửa cuối (Head-Tail truncation)."
                            )
                        if bilstm_res.oov_rate > 0.20:
                            st.warning(
                                f"• **Tỷ lệ từ ngoài từ điển cao ({bilstm_res.oov_rate:.1%}):** "
                                "Nhiều từ chưa xuất hiện trong tập Train, độ tin cậy có thể giảm."
                            )
                        if "OUT_OF_DOMAIN_LANGUAGE_HEURISTIC" in bilstm_res.warnings:
                            st.info(
                                "• **Ngôn ngữ ngoài miền:** Văn bản có dấu hiệu "
                                "không phải tiếng Anh."
                            )

                # Token Audit Inspector
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔍 Kiểm Toán Token Hóa (Token Inspection & OOV Highlighter)"):
                    st.write(
                        "Các token sau tiền xử lý (loại HTML, chuẩn hóa chữ thường). "
                        "Các từ viền đỏ là `<UNK>` (ngoài từ điển):"
                    )
                    tokens = tokenize(review_text)
                    vocab = getattr(bilstm_predictor, "vocabulary", None)
                    token_html = []
                    for t in tokens:
                        if vocab and t not in vocab.token_to_index:
                            chip = f'<span class="token-chip token-chip-unk">{t}*</span>'
                            token_html.append(chip)
                        else:
                            token_html.append(
                                f'<span class="token-chip token-chip-normal">{t}</span>'
                            )
                    st.markdown("".join(token_html), unsafe_allow_html=True)
                    st.caption("*(Dấu sao màu đỏ là các từ ngoài từ điển `token_to_index`)*")

            except Exception as exc:
                st.error(f"Lỗi khi suy luận: {exc}")

# ==============================================================================
# TAB 2: MODEL COMPARISON & CALIBRATION
# ==============================================================================
with tab_compare:
    st.markdown("### ⚖️ Đối Chiếu Mô Hình & Phân Tích Hiệu Chuẩn (Calibration)")
    st.write(
        "Đối chiếu khoa học giữa Baseline truyền thống và Mô hình Deep Learning BiLSTM "
        "được kiểm toán trên tập Validation và tập Test độc lập."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">BiLSTM Val Accuracy</div>
                <div class="metric-value">88.38%</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">Test: 87.89%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Baseline Val Accuracy</div>
                <div class="metric-value">89.58%</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">TF-IDF + LR</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">BiLSTM Brier Score</div>
                <div class="metric-value" style="color:#38bdf8;">0.0883</div>
                <div style="font-size:0.82rem; color:#94a3b8; margin-top:4px;">
                    (Càng thấp càng tốt)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">BiLSTM ECE (Calibrated)</div>
                <div class="metric-value" style="color:#10b981;">0.0264</div>
                <div style="font-size:0.82rem; color:#94a3b8; margin-top:4px;">
                    (Lệch xác suất ~2.6%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Head-to-head comparison table
    st.markdown("#### 📋 Bảng So Sánh Toàn Diện (Head-to-Head Benchmark)")
    comparison_data = [
        {
            "Mô Hình": "TF-IDF + Logistic Regression",
            "Accuracy": "89.58%",
            "Macro-F1": "89.58%",
            "Brier Score": "0.0768",
            "ECE": "0.0276",
            "Tham Số": "100,001",
            "CPU Latency": "0.33 ms",
            "Ưu Điểm Cốt Lõi": "Cực nhanh, giải thích rõ qua trọng số n-grams",
        },
        {
            "Mô Hình": "BiLSTM + Temperature Scaling",
            "Accuracy": "88.38%",
            "Macro-F1": "88.38%",
            "Brier Score": "0.0883",
            "ECE": "0.0264",
            "Tham Số": "658,049",
            "CPU Latency": "3.55 ms",
            "Ưu Điểm Cốt Lõi": "Hiểu ngữ cảnh tuần tự, xác suất hiệu chuẩn đáng tin cậy",
        },
    ]
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📈 Đồ Thị Đánh Giá & Đường Cong Tin Cậy (Reliability Diagrams)")

    img_col1, img_col2 = st.columns(2)
    history_img = Path("artifacts/training_history.png")
    cm_img = Path("artifacts/confusion_matrix.png")
    rel_img = Path("artifacts/reliability_diagram.png")
    test_rel_img = Path("artifacts/test_reliability_diagram.png")

    with img_col1:
        if history_img.is_file():
            st.image(
                str(history_img),
                caption="Lịch sử Huấn luyện (Loss & Accuracy qua các Epochs)",
                use_container_width=True,
            )
        if rel_img.is_file():
            st.image(
                str(rel_img),
                caption="Reliability Diagram trên tập Validation (Calibration Curve)",
                use_container_width=True,
            )

    with img_col2:
        if cm_img.is_file():
            st.image(
                str(cm_img),
                caption="Ma trận nhầm lẫn (Confusion Matrix trên Validation)",
                use_container_width=True,
            )
        if test_rel_img.is_file():
            st.image(
                str(test_rel_img),
                caption="Reliability Diagram trên tập Test độc lập",
                use_container_width=True,
            )

# ==============================================================================
# TAB 3: ERROR ANALYSIS
# ==============================================================================
with tab_errors:
    st.markdown("### 🔍 Phân Tích Lát Cắt Lỗi Ngôn Ngữ Học (Error Analysis)")
    st.write(
        "Mô hình học máy trong môi trường thực tế cần phân tích kỹ lưỡng các lát cắt "
        "ngôn ngữ học dễ gây sai lệch (Linguistic Failure Slices)."
    )

    if error_data:
        err_c1, err_c2, err_c3 = st.columns(3)
        with err_c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Tổng Mẫu Phân Tích</div>
                    <div class="metric-value">{error_data.get("total_samples", 0):,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with err_c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Tổng Số Lỗi</div>
                    <div class="metric-value" style="color:#ef4444;">
                        {error_data.get("total_errors", 0):,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with err_c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Tỷ Lệ Lỗi Chung</div>
                    <div class="metric-value">{error_data.get("overall_error_rate", 0.0):.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        slice_col1, slice_col2 = st.columns(2)

        with slice_col1:
            st.markdown("#### 🗣️ Lát Cắt Ngôn Ngữ Học (Linguistic Slices)")
            ling_rows = []
            for _k, v in error_data.get("linguistic_slices", {}).items():
                ling_rows.append(
                    {
                        "Lát Cắt": v["name"],
                        "Số Lượng": f"{v['total_count']:,}",
                        "Số Lỗi": v["error_count"],
                        "Tỷ Lệ Lỗi": f"{v['error_rate']:.2%}",
                    }
                )
            st.dataframe(pd.DataFrame(ling_rows), use_container_width=True, hide_index=True)

            st.markdown("#### 📏 Lát Cắt Theo Độ Dài Câu (Length Slices)")
            len_rows = []
            for k, v in error_data.get("length_slices", {}).items():
                len_rows.append(
                    {
                        "Độ Dài Câu": k,
                        "Tổng Mẫu": f"{v['total']:,}",
                        "Số Lỗi": v["errors"],
                        "Tỷ Lệ Lỗi": f"{v['error_rate']:.2%}",
                    }
                )
            st.dataframe(pd.DataFrame(len_rows), use_container_width=True, hide_index=True)

        with slice_col2:
            st.markdown("#### 🧩 Lát Cắt Từ Ngoài Từ Điển (OOV Slices)")
            oov_rows = []
            for k, v in error_data.get("oov_slices", {}).items():
                oov_rows.append(
                    {
                        "Mức Độ OOV": k,
                        "Tổng Mẫu": f"{v['total']:,}",
                        "Số Lỗi": v["errors"],
                        "Tỷ Lệ Lỗi": f"{v['error_rate']:.2%}",
                    }
                )
            st.dataframe(pd.DataFrame(oov_rows), use_container_width=True, hide_index=True)

            st.markdown("#### 🎯 Top Mẫu Tự Tin Sai (High-Confidence Errors)")
            high_conf = error_data.get("high_confidence_examples", [])
            for idx, ex in enumerate(high_conf[:3], 1):
                lbl_str = "Positive" if ex["label"] == 1 else "Negative"
                pred_str = "Positive" if ex["predicted"] == 1 else "Negative"
                st.markdown(
                    f"""
                    <div style="background:rgba(239,68,68,0.08); border-left:3px solid #ef4444;
                                padding:10px 14px; border-radius:6px; margin-bottom:10px;">
                        <div style="font-size:0.85rem; color:#f87171; font-weight:600;">
                            #{idx} • Nhãn Thực: <strong>{lbl_str}</strong>
                            | Dự Đoán: <strong>{pred_str}</strong>
                            | Tin cậy: <strong>{ex["confidence"]:.1%}</strong>
                        </div>
                        <div style="font-size:0.9rem; color:#e2e8f0; margin-top:4px;">
                            "{ex["text_snippet"]}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Chưa tìm thấy báo cáo phân tích lỗi `artifacts/error_analysis.json`.")

    # Lexical Explainability from Baseline
    if explain_data:
        st.markdown("---")
        st.markdown("#### 💡 Tín Hiệu Từ Vựng Quan Trọng (Lexical Signals từ TF-IDF)")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("**🟢 Top N-grams Tích Cực:**")
            pos_df = pd.DataFrame(explain_data.get("top_positive_ngrams", [])[:10])
            if not pos_df.empty:
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
        with exp_col2:
            st.markdown("**🔴 Top N-grams Tiêu Cực:**")
            neg_df = pd.DataFrame(explain_data.get("top_negative_ngrams", [])[:10])
            if not neg_df.empty:
                st.dataframe(neg_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 4: BATCH REVIEW TESTER
# ==============================================================================
with tab_batch:
    st.markdown("### 📦 Kiểm Thử Đánh Giá Theo Lô (Batch Inference)")
    st.write("Xử lý hàng loạt danh sách câu đánh giá hoặc tải lên tệp CSV chứa cột `text`:")

    batch_mode = st.radio(
        "Chọn phương thức nhập dữ liệu:",
        ["Nhập văn bản từng dòng", "Tải lên tệp CSV/TXT"],
        horizontal=True,
    )
    batch_texts: list[str] = []

    if batch_mode == "Nhập văn bản từng dòng":
        default_batch = (
            "An incredible masterpiece with magnificent soundtrack and acting.\n"
            "Utterly boring and a complete waste of two hours.\n"
            "Decent effects but the script makes very little sense.\n"
            "I genuinely enjoyed the witty humor and unexpected twist ending.\n"
            "Worst film of the decade, totally unwatchable."
        )
        batch_raw = st.text_area(
            "Danh sách các câu đánh giá (mỗi dòng một câu):",
            value=default_batch,
            height=130,
        )
        batch_texts = [line.strip() for line in batch_raw.splitlines() if line.strip()]
    else:
        uploaded_file = st.file_uploader(
            "Tải lên file CSV hoặc TXT (cần có cột 'text' nếu là CSV):",
            type=["csv", "txt"],
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded_file)
                    if "text" in df_up.columns:
                        batch_texts = df_up["text"].dropna().astype(str).tolist()
                        st.success(f"Đã nạp {len(batch_texts):,} câu từ tệp CSV.")
                    else:
                        st.error("Tệp CSV cần có cột tên là 'text'.")
                else:
                    content = uploaded_file.read().decode("utf-8", errors="replace")
                    batch_texts = [line.strip() for line in content.splitlines() if line.strip()]
                    st.success(f"Đã nạp {len(batch_texts):,} dòng từ tệp TXT.")
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {e}")

    if st.button("⚡ Chạy Dự Đoán Hàng Loạt", type="primary", use_container_width=True):
        if not batch_texts:
            st.warning("Vui lòng nhập hoặc tải dữ liệu trước khi chạy.")
        elif bilstm_predictor is None:
            st.error("Chưa nạp được checkpoint mô hình BiLSTM.")
        else:
            with st.spinner(f"Đang phân tích {len(batch_texts):,} đánh giá..."):
                results = bilstm_predictor.predict_batch(batch_texts)

                pos_count = sum(1 for r in results if r.label == "Positive")
                neg_count = len(results) - pos_count
                pos_pct = pos_count / len(results) if results else 0.0

                st.markdown("<br>", unsafe_allow_html=True)
                b_c1, b_c2, b_c3 = st.columns(3)
                with b_c1:
                    st.metric("Tổng Số Mẫu", len(results))
                with b_c2:
                    st.metric(
                        "Tỷ Lệ Tích Cực (Positive)",
                        f"{pos_pct:.1%}",
                        delta=f"{pos_count} câu",
                    )
                with b_c3:
                    st.metric(
                        "Tỷ Lệ Tiêu Cực (Negative)",
                        f"{1 - pos_pct:.1%}",
                        delta=f"-{neg_count} câu",
                        delta_color="inverse",
                    )

                result_records = []
                for t, r in zip(batch_texts, results, strict=False):
                    result_records.append(
                        {
                            "Review Text": t[:120] + ("..." if len(t) > 120 else ""),
                            "Prediction": r.label,
                            "Calibrated Prob": f"{r.probability:.2%}",
                            "Tokens": r.token_count,
                            "OOV Rate": f"{r.oov_rate:.1%}",
                            "Truncated": "Yes" if r.truncated else "No",
                        }
                    )
                res_df = pd.DataFrame(result_records)
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                csv_buffer = io.StringIO()
                res_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Tải Kết Quả (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="cinesentiment_batch_results.csv",
                    mime="text/csv",
                )

# ==============================================================================
# TAB 5: ARCHITECTURE & REST API
# ==============================================================================
with tab_arch:
    st.markdown("### 📐 Kiến Trúc Hệ Thống & Tích Hợp REST API")
    st.write(
        "CineSentiment AI được thiết kế theo các tiêu chuẩn Machine Learning hiện đại, "
        "tách biệt rõ ràng giữa huấn luyện, hiệu chuẩn và phục vụ suy luận."
    )

    st.markdown(
        """
        ```
        +-----------------------------------------------------------------------------------+
        |                         KIẾN TRÚC MÔ HÌNH BiLSTM CINESENTIMENT                    |
        +-----------------------------------------------------------------------------------+
        |  Input Text: "This movie is wonderful..."                                         |
        |        |                                                                          |
        |        v                                                                          |
        |  Tokenizer (Regex bảo tồn từ phủ định "don't", "isn't", loại bỏ HTML)             |
        |        |                                                                          |
        |        v                                                                          |
        |  Truncation Strategy: Head-Tail (max_length=256, giữ 128 đầu + 128 cuối)         |
        |        |                                                                          |
        |        v                                                                          |
        |  Embedding Layer (dim=128, padding_idx=<PAD>, bỏ qua gradient padding)            |
        |        |                                                                          |
        |        v                                                                          |
        |  PyTorch pack_padded_sequence (loại bỏ lãng phí tính toán RNN trên vùng padding)  |
        |        |                                                                          |
        |        v                                                                          |
        |  2-Layer Bidirectional LSTM (hidden_dim=128, dropout=0.4)                         |
        |        |                                                                          |
        |        v                                                                          |
        |  Concat(h_forward[-2], h_backward[-1]) -> Feature Vector 256 chiều                |
        |        |                                                                          |
        |        v                                                                          |
        |  Dropout(0.4) -> Linear(256, 1) -> Raw Logit z                                    |
        |        |                                                                          |
        |        v                                                                          |
        |  Temperature Scaling (z / T) với T tối ưu qua L-BFGS trên Calibration set        |
        |        |                                                                          |
        |        v                                                                          |
        |  Calibrated Probability: p = Sigmoid(z / T) -> Nhãn (p >= 0.5: Positive)          |
        +-----------------------------------------------------------------------------------+
        ```
        """
    )

    st.markdown("#### 🔌 Hướng Dẫn Tích Hợp FastAPI REST API")
    st.write("Khởi chạy REST API: `uvicorn api:app --host 0.0.0.0 --port 8000`")

    api_code_python = """import requests

url = "http://localhost:8000/predict"
payload = {
    "text": "This movie was absolutely captivating with stunning visuals!"
}
response = requests.post(url, json=payload)
data = response.json()
print("Kết quả:", data["label"], "| Xác suất:", data["probability"])
"""
    api_code_curl = """curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "This movie had a fantastic storyline and outstanding performances!"}'
"""
    code_tab1, code_tab2 = st.tabs(["Python Code", "cURL Command"])
    with code_tab1:
        st.code(api_code_python, language="python")
    with code_tab2:
        st.code(api_code_curl, language="bash")
