"""
dashboard.py — RNIA Visualization Dashboard
============================================

Streamlit dashboard for the Event-Driven Retail News Impact Analyst.

Tabs:
    1. 📊 Overview       — KPI cards, event-type pie chart, stance bar chart
    2. 📈 Impact Analytics — histogram, scatter plot, top-10 table
    3. 🔍 Single Event    — drill-down into individual articles
    4. 📋 Data Explorer   — full filterable table with CSV download

Launch:
    streamlit run dashboard/dashboard.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "results", "news_with_impact_scores.csv")

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RNIA — Retail News Impact Analyst",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark modern card style for metrics */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    /* Header banner */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        padding: 24px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        border: 1px solid #334155;
    }
    .main-header h1 { color: #e2e8f0; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.95rem; }

    /* Impact badge */
    .impact-high   { color: #ef4444; font-weight: 700; }
    .impact-medium { color: #f59e0b; font-weight: 700; }
    .impact-low    { color: #22c55e; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
EVENT_COLORS = {
    "earnings":              "#6366f1",
    "leadership_change":     "#f59e0b",
    "regulatory_action":     "#ef4444",
    "mergers_acquisitions":  "#10b981",
    "legal_action":          "#f97316",
    "product_announcement":  "#3b82f6",
    "market_movement":       "#8b5cf6",
}

STANCE_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#64748b",
}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the scored dataset, with caching."""
    if not os.path.isfile(DATA_FILE):
        st.error(
            f"Results file not found at `{DATA_FILE}`.\n\n"
            "Please run `python pipeline/run_pipeline.py` first."
        )
        st.stop()
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    return df


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🔧 Filters")

# Event type filter
all_events = sorted(df["event_type"].dropna().unique())
selected_events = st.sidebar.multiselect(
    "Event Type",
    options=all_events,
    default=all_events,
)

# Stance filter
all_stances = sorted(df["stance"].dropna().unique())
selected_stances = st.sidebar.multiselect(
    "Stance",
    options=all_stances,
    default=all_stances,
)

# Impact score range
min_score = float(df["impact_score"].min())
max_score = float(df["impact_score"].max())
score_range = st.sidebar.slider(
    "Impact Score Range",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score),
    step=0.1,
)

# Apply filters
mask = (
    df["event_type"].isin(selected_events)
    & df["stance"].isin(selected_stances)
    & df["impact_score"].between(score_range[0], score_range[1])
)
filtered = df[mask].reset_index(drop=True)

st.sidebar.markdown("---")
st.sidebar.metric("Showing", f"{len(filtered)} / {len(df)} articles")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>📰 Event-Driven Retail News Impact Analyst</h1>
        <p>Financial news classification, stance detection & impact scoring dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Impact Analytics",
    "🔍 Single Event Analysis",
    "📋 Data Explorer",
])

# ===== TAB 1 — Overview ====================================================
with tab1:
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Articles", len(filtered))
    col2.metric("Avg Impact Score", f"{filtered['impact_score'].mean():.2f}")
    col3.metric(
        "Top Event Type",
        filtered["event_type"].value_counts().idxmax() if len(filtered) else "—",
    )
    col4.metric("Sources", filtered["source"].nunique() if len(filtered) else 0)

    st.markdown("")

    # Charts row
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Event Type Distribution")
        event_counts = filtered["event_type"].value_counts().reset_index()
        event_counts.columns = ["event_type", "count"]
        fig_pie = px.pie(
            event_counts,
            values="count",
            names="event_type",
            color="event_type",
            color_discrete_map=EVENT_COLORS,
            hole=0.4,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend=dict(font=dict(size=11)),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_right:
        st.subheader("Stance Distribution")
        stance_counts = filtered["stance"].value_counts().reset_index()
        stance_counts.columns = ["stance", "count"]
        fig_bar = px.bar(
            stance_counts,
            x="stance",
            y="count",
            color="stance",
            color_discrete_map=STANCE_COLORS,
            text="count",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis_title="",
            yaxis_title="Count",
            showlegend=False,
            margin=dict(t=20, b=20),
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Source breakdown
    st.subheader("Articles by Source")
    source_counts = filtered["source"].value_counts().reset_index()
    source_counts.columns = ["source", "count"]
    fig_source = px.bar(
        source_counts,
        x="source",
        y="count",
        color="count",
        color_continuous_scale="Blues",
        text="count",
    )
    fig_source.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis_title="",
        yaxis_title="Articles",
        margin=dict(t=20, b=20),
    )
    fig_source.update_traces(textposition="outside")
    st.plotly_chart(fig_source, use_container_width=True)


# ===== TAB 2 — Impact Analytics =============================================
with tab2:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Impact Score Distribution")
        fig_hist = px.histogram(
            filtered,
            x="impact_score",
            nbins=20,
            color="stance",
            color_discrete_map=STANCE_COLORS,
            barmode="overlay",
            opacity=0.75,
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis_title="Impact Score",
            yaxis_title="Count",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("Confidence vs Impact")
        fig_scatter = px.scatter(
            filtered,
            x="event_confidence",
            y="impact_score",
            color="event_type",
            color_discrete_map=EVENT_COLORS,
            size="impact_score",
            hover_data=["headline", "stance"],
            opacity=0.8,
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis_title="Event Classification Confidence",
            yaxis_title="Impact Score",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Average impact by event type
    st.subheader("Average Impact Score by Event Type")
    avg_impact = (
        filtered.groupby("event_type")["impact_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    avg_impact.columns = ["event_type", "avg_impact"]
    fig_avg = px.bar(
        avg_impact,
        x="event_type",
        y="avg_impact",
        color="event_type",
        color_discrete_map=EVENT_COLORS,
        text=avg_impact["avg_impact"].round(2),
    )
    fig_avg.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis_title="",
        yaxis_title="Avg Impact Score",
        showlegend=False,
        margin=dict(t=20, b=20),
    )
    fig_avg.update_traces(textposition="outside")
    st.plotly_chart(fig_avg, use_container_width=True)

    # Top-10 highest-impact articles
    st.subheader("🔥 Top 10 Highest-Impact Articles")
    top10 = filtered.nlargest(10, "impact_score")[
        ["headline", "event_type", "stance", "impact_score", "source"]
    ].reset_index(drop=True)
    top10.index += 1
    st.dataframe(top10, use_container_width=True)


# ===== TAB 3 — Single Event Analysis =======================================
with tab3:
    if filtered.empty:
        st.info("No articles match the current filters.")
    else:
        options = filtered["headline"].tolist()
        selected_headline = st.selectbox("Select an article", options)

        article = filtered[filtered["headline"] == selected_headline].iloc[0]

        # Impact badge colour
        score = article["impact_score"]
        if score >= 6:
            badge_class = "impact-high"
            badge_label = "HIGH IMPACT"
        elif score >= 3:
            badge_class = "impact-medium"
            badge_label = "MEDIUM IMPACT"
        else:
            badge_class = "impact-low"
            badge_label = "LOW IMPACT"

        st.markdown(f"### {article['headline']}")
        st.markdown(
            f'<span class="{badge_class}">● {badge_label} — {score:.2f}/10</span>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Detail columns
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Event Type", article["event_type"].replace("_", " ").title())
        mc2.metric("Event Confidence", f"{article['event_confidence']:.2%}")
        mc3.metric("Stance", article["stance"].title())
        mc4.metric("Stance Confidence", f"{article['stance_confidence']:.2%}")

        st.markdown("---")
        col_info, col_text = st.columns([1, 2])

        with col_info:
            st.markdown("**Source:**  " + str(article["source"]))
            st.markdown("**Timestamp:**  " + str(article["timestamp"]))
            if pd.notna(article["url"]):
                st.markdown(f"**URL:**  [Open article]({article['url']})")

        with col_text:
            st.markdown("**Article Text:**")
            st.text_area(
                label="Full text",
                value=str(article["clean_text"]),
                height=250,
                disabled=True,
                label_visibility="collapsed",
            )


# ===== TAB 4 — Data Explorer ================================================
with tab4:
    st.subheader("Full Dataset")

    # Display
    display_cols = [
        "headline", "event_type", "event_confidence",
        "stance", "stance_confidence", "impact_score", "source", "timestamp",
    ]
    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        height=500,
    )

    # Download button
    csv_bytes = filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="⬇️  Download filtered dataset as CSV",
        data=csv_bytes,
        file_name="rnia_filtered_results.csv",
        mime="text/csv",
    )
