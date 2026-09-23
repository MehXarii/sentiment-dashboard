import streamlit as st
import pandas as pd
from pathlib import Path
from utils.analyzer import load_model, analyze_sentiment
from utils.charts import sentiment_bar_chart, sentiment_pie_chart, confidence_chart

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load External CSS ───────────────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.warning("`style.css` not found. Place it in the same folder as app.py.")

load_css()

# ── Cache Model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()

# ── Session State Initialization ────────────────────────────────────────────────
if "df_results" not in st.session_state:
    st.session_state.df_results = None

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This dashboard uses **DistilBERT** (fine-tuned on SST-2) "
        "to classify product reviews into sentiment categories with confidence scoring."
    )
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Enter text lines or upload a CSV file.")
    st.markdown("2. Click **Analyze** to run inference.")
    st.markdown("3. Filter and export analysis results.")
    st.markdown("---")
    st.caption("**Model:** `distilbert-base-uncased-finetuned-sst-2-english`")
    st.caption("**Author:** Mehak Ansari")

# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Sentiment Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze product reviews instantly with AI — paste text or upload a CSV file.</div>', unsafe_allow_html=True)

# ── Data Input Section ───────────────────────────────────────────────────────────
input_tab, upload_tab = st.tabs(["Paste Reviews", "Upload CSV"])

reviews_to_analyze = []

with input_tab:
    text_input = st.text_area(
        "Enter reviews (one per line):",
        height=180,
        placeholder="Great product, love it!\nTerrible quality, very disappointed.\nIt's okay, nothing special.",
        key="text_input_area",
    )
    if st.button("Analyze Text", key="btn_text"):
        lines = [line.strip() for line in text_input.strip().split("\n") if line.strip()]
        if lines:
            reviews_to_analyze = lines
        else:
            st.warning("Please enter at least one valid review sentence.")

with upload_tab:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="file_uploader")
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if not df_upload.empty:
                col_choice = st.selectbox("Select column containing reviews:", df_upload.columns.tolist())
                if st.button("Analyze CSV", key="btn_csv"):
                    reviews_to_analyze = df_upload[col_choice].dropna().astype(str).tolist()
            else:
                st.error("The uploaded CSV file is empty.")
        except Exception as err:
            st.error(f"Failed to process CSV file: {err}")

# ── Model Execution ─────────────────────────────────────────────────────────────
if reviews_to_analyze:
    with st.spinner("Analyzing sentiment with DistilBERT..."):
        try:
            pipeline = get_model()
            results = analyze_sentiment(pipeline, reviews_to_analyze)
            st.session_state.df_results = pd.DataFrame(results)
        except Exception as e:
            st.error(f"Error during analysis: {e}")

# ── Render Dashboard Output ─────────────────────────────────────────────────────
if st.session_state.df_results is not None and not st.session_state.df_results.empty:
    df = st.session_state.df_results

    # Summary Metrics
    total     = len(df)
    pos_count = (df["label"] == "POSITIVE").sum()
    neg_count = (df["label"] == "NEGATIVE").sum()
    neu_count = (df["label"] == "NEUTRAL").sum()
    avg_conf  = df["confidence"].mean() if "confidence" in df else 0.0

    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.markdown(f'<div class="metric-card"><div class="metric-label">Total</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-label">Positive</div><div class="metric-value positive">{pos_count}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-label">Negative</div><div class="metric-value negative">{neg_count}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-label">Neutral</div><div class="metric-value neutral">{neu_count}</div></div>', unsafe_allow_html=True)
    m5.markdown(f'<div class="metric-card"><div class="metric-label">Avg Conf.</div><div class="metric-value">{avg_conf:.0%}</div></div>', unsafe_allow_html=True)

    # Visualization Charts
    st.markdown("---")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(sentiment_bar_chart(df), use_container_width=True)
    with ch2:
        st.plotly_chart(sentiment_pie_chart(df), use_container_width=True)

    st.plotly_chart(confidence_chart(df), use_container_width=True)

    # Automated Summary Insight
    pos_pct = (pos_count / total) * 100
    neg_pct = (neg_count / total) * 100

    if pos_pct >= 70:
        tone   = "overwhelmingly positive"
        action = "Customers are highly satisfied — leverage positive feedback in marketing campaigns."
    elif neg_pct >= 70:
        tone   = "predominantly negative"
        action = "Urgent action required — investigate frequent issues in customer complaints."
    elif pos_pct > neg_pct:
        tone   = "mostly positive with minor concerns"
        action = "Overall sentiment is positive; address specific recurring negative feedback."
    elif neg_pct > pos_pct:
        tone   = "leaning negative"
        action = "Negative feedback outweighs positive — evaluate product or support quality."
    else:
        tone   = "mixed"
        action = "Customer sentiment is split evenly — conduct further qualitative review."

    st.markdown(f"""
    <div class="insight-box">
        <strong>Summary Insight</strong><br>
        Across <strong>{total}</strong> reviews, overall sentiment is <strong>{tone}</strong>
        ({pos_count} positive, {neg_count} negative, {neu_count} neutral) with an average model confidence of <strong>{avg_conf:.0%}</strong>.<br><br>
        <strong>Recommendation:</strong> {action}
    </div>
    """, unsafe_allow_html=True)

    # Per-review Breakdown with Filter Controls
    st.markdown("---")
    st.markdown("### Detailed Review Breakdown")

    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        selected_label = st.selectbox("Filter by Sentiment:", ["All", "POSITIVE", "NEGATIVE", "NEUTRAL"])
    with search_col:
        search_query = st.text_input("Search reviews:", placeholder="Type keywords...")

    filtered_df = df.copy()
    if selected_label != "All":
        filtered_df = filtered_df[filtered_df["label"] == selected_label]
    if search_query.strip():
        filtered_df = filtered_df[filtered_df["review"].str.contains(search_query, case=False, na=False)]

    st.caption(f"Showing {len(filtered_df)} of {total} reviews")

    for _, row in filtered_df.iterrows():
        badge_class   = f"badge-{row['label'].lower()}"
        label_display = row["label"].capitalize()
        conf_display  = f"{row['confidence']:.0%}"
        st.markdown(f"""
        <div class="review-row">
            <span class="badge {badge_class}">{label_display}</span>
            &nbsp; <span style="font-size:0.85rem; color:#64748b;"><strong>{conf_display}</strong> confidence</span>
            <div style="color:#334155; margin-top:0.4rem; font-size:0.92rem;">{row['review']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Export Results
    st.markdown("---")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Results CSV",
        data=csv_data,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv",
        key="btn_download",
    )