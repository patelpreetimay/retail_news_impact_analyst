"""
app.py — RNIA Final Streamlit Dashboard
========================================

Interactive dashboard for the Event-Driven Retail News Impact Analyst.

Features:
    - Sidebar filters  : Event Category, Stance, News Source, Min Impact Score
    - Summary metrics  : Total Articles, Average Impact Score, Top Event Type
    - Data table       : Filterable article table
    - Article details  : Expandable panels with explanation text
    - Visualizations   : Event distribution bar chart, Stance pie chart

Launch:
    streamlit run dashboard/app.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(
    PROJECT_ROOT, "data", "final_outputs", "news_analysis_report.csv"
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RNIA — Retail News Impact Analyst",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Modern dark-themed styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }

    /* ---- Header banner ---- */
    .main-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        padding: 28px 36px;
        border-radius: 14px;
        margin-bottom: 28px;
        border: 1px solid #334155;
    }
    .main-banner h1 { color: #e2e8f0; margin: 0; font-size: 1.9rem; }
    .main-banner p  { color: #94a3b8; margin: 6px 0 0 0; font-size: 0.95rem; }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    /* ---- Expander styling ---- */
    .stExpander {
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        margin-bottom: 8px;
    }

    /* Impact badge colours */
    .badge-high   { color: #ef4444; font-weight: 700; font-size: 1.1rem; }
    .badge-medium { color: #f59e0b; font-weight: 700; font-size: 1.1rem; }
    .badge-low    { color: #22c55e; font-weight: 700; font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Colour palettes for charts
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
# STEP 3 — Load Data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the final analysis report dataset with caching."""
    if not os.path.isfile(DATA_FILE):
        st.error(
            f"**Report file not found:**\n\n`{DATA_FILE}`\n\n"
            "Please run `python reporting/explanation_generator.py` first."
        )
        st.stop()
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    return df


df = load_data()

# ---------------------------------------------------------------------------
# STEP 4 — Sidebar Filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🔧 Filters")

# Event Category filter
all_events = sorted(df["event_type"].dropna().unique())
selected_events = st.sidebar.multiselect(
    "Event Category",
    options=all_events,
    default=all_events,
    help="Filter articles by financial event type",
)

# Stance filter
all_stances = sorted(df["stance"].dropna().unique())
selected_stances = st.sidebar.multiselect(
    "Stance",
    options=all_stances,
    default=all_stances,
    help="Filter by sentiment stance",
)

# News Source filter
all_sources = sorted(df["source"].dropna().unique())
selected_sources = st.sidebar.multiselect(
    "News Source",
    options=all_sources,
    default=all_sources,
    help="Filter by news source",
)

# Minimum Impact Score slider
min_impact = float(df["impact_score"].min())
max_impact = float(df["impact_score"].max())
min_score = st.sidebar.slider(
    "Minimum Impact Score",
    min_value=min_impact,
    max_value=max_impact,
    value=min_impact,
    step=0.01,
    help="Show only articles with impact score above this threshold",
)

# Apply all filters
mask = (
    df["event_type"].isin(selected_events)
    & df["stance"].isin(selected_stances)
    & df["source"].isin(selected_sources)
    & (df["impact_score"] >= min_score)
)
filtered = df[mask].reset_index(drop=True)

# Sidebar count
st.sidebar.markdown("---")
st.sidebar.metric("Showing", f"{len(filtered)} / {len(df)} articles")

# ---------------------------------------------------------------------------
# Header Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-banner">
        <h1>📰 Event-Driven Retail News Impact Analyst</h1>
        <p>Explore financial news events, sentiment stance, impact scores,
        and AI-generated explanations</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# STEP 7 — Summary Metrics
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Articles", len(filtered))

col2.metric(
    "Avg Impact Score",
    f"{filtered['impact_score'].mean():.2f}" if len(filtered) else "—",
)

col3.metric(
    "Top Event Type",
    filtered["event_type"].value_counts().idxmax().replace("_", " ").title()
    if len(filtered) else "—",
)

col4.metric(
    "News Sources",
    filtered["source"].nunique() if len(filtered) else 0,
)

st.markdown("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📋 Articles & Details",
    "📊 Visualizations",
    "📥 Export",
])

# ===== TAB 1 — Main Table + Article Details =================================
with tab1:

    # ---- STEP 5 — Main Table Display ----------------------------------------
    st.subheader("Filtered Articles")

    display_cols = [
        "headline", "event_type", "stance",
        "impact_score", "source", "timestamp",
    ]
    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        height=350,
        column_config={
            "headline":     st.column_config.TextColumn("Headline", width="large"),
            "event_type":   st.column_config.TextColumn("Event Type"),
            "stance":       st.column_config.TextColumn("Stance"),
            "impact_score": st.column_config.NumberColumn("Impact Score", format="%.2f"),
            "source":       st.column_config.TextColumn("Source"),
            "timestamp":    st.column_config.TextColumn("Timestamp"),
        },
    )

    # ---- STEP 6 — Article Details Panel -------------------------------------
    st.subheader("Article Details")

    if filtered.empty:
        st.info("No articles match the current filters.")
    else:
        for idx, row in filtered.iterrows():
            # Determine impact badge
            score = row["impact_score"]
            if score >= 0.80:
                badge = '<span class="badge-high">● HIGH IMPACT</span>'
            elif score >= 0.60:
                badge = '<span class="badge-medium">● MEDIUM IMPACT</span>'
            else:
                badge = '<span class="badge-low">● LOW IMPACT</span>'

            # Expandable panel for each article
            with st.expander(f"📄 {row['headline'][:90]}"):
                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Event Type", str(row["event_type"]).replace("_", " ").title())
                m2.metric("Stance", str(row["stance"]).title())
                m3.metric("Impact Score", f"{score:.2f}")
                m4.markdown(f"<br>{badge}", unsafe_allow_html=True)

                st.markdown("---")

                # Explanation text
                explanation = row.get("explanation", "No explanation available.")
                st.markdown("**🔍 Explanation:**")
                st.info(explanation)

                # Source & timestamp
                col_l, col_r = st.columns(2)
                col_l.markdown(f"**Source:** {row.get('source', 'N/A')}")
                col_r.markdown(f"**Timestamp:** {row.get('timestamp', 'N/A')}")


# ===== TAB 2 — Visualizations ===============================================
with tab2:

    if filtered.empty:
        st.info("No articles match the current filters.")
    else:
        chart_left, chart_right = st.columns(2)

        # ---- STEP 8a — Event Category Distribution (Bar Chart) ----
        with chart_left:
            st.subheader("Event Category Distribution")
            event_counts = filtered["event_type"].value_counts().reset_index()
            event_counts.columns = ["event_type", "count"]

            fig_bar = px.bar(
                event_counts,
                x="event_type",
                y="count",
                color="event_type",
                color_discrete_map=EVENT_COLORS,
                text="count",
                labels={"event_type": "Event Type", "count": "Articles"},
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis_title="",
                yaxis_title="Number of Articles",
                showlegend=False,
                margin=dict(t=20, b=20),
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        # ---- STEP 8b — Stance Distribution (Pie Chart) ----
        with chart_right:
            st.subheader("Stance Distribution")
            stance_counts = filtered["stance"].value_counts().reset_index()
            stance_counts.columns = ["stance", "count"]

            fig_pie = px.pie(
                stance_counts,
                values="count",
                names="stance",
                color="stance",
                color_discrete_map=STANCE_COLORS,
                hole=0.4,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                legend=dict(font=dict(size=12)),
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ---- Extra: Impact Score by Source ----
        st.subheader("Average Impact Score by Source")
        source_impact = (
            filtered.groupby("source")["impact_score"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        source_impact.columns = ["source", "avg_impact"]

        fig_source = px.bar(
            source_impact,
            x="source",
            y="avg_impact",
            color="avg_impact",
            color_continuous_scale="Blues",
            text=source_impact["avg_impact"].round(2),
            labels={"source": "Source", "avg_impact": "Avg Impact Score"},
        )
        fig_source.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis_title="",
            yaxis_title="Average Impact Score",
            margin=dict(t=20, b=20),
        )
        fig_source.update_traces(textposition="outside")
        st.plotly_chart(fig_source, use_container_width=True)


# ===== TAB 3 — Export ========================================================
with tab3:
    st.subheader("Download Filtered Results")
    st.write(f"**{len(filtered)}** articles match your current filters.")

    csv_bytes = filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="⬇️  Download as CSV",
        data=csv_bytes,
        file_name="rnia_filtered_report.csv",
        mime="text/csv",
    )
