# --------------------------------------------------
# app.py — Innovation Signal Analytics
# --------------------------------------------------

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Innovation Analytics · World Bank PADs",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.app-header {
    background: #0a1628;
    color: white;
    padding: 1.75rem 2rem;
    margin: -4rem -4rem 2rem -4rem;
}

.app-header h1 {
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0 0 0.2rem 0;
    color: white;
}

.app-header p {
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 0;
}

.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    font-weight: 600;
    margin: 1.5rem 0 0.75rem 0;
}

.footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
    font-size: 0.75rem;
    color: #94a3b8;
    text-align: center;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA
# --------------------------------------------------

ROOT = Path(__file__).parent

@st.cache_data
def load_data():
    d = {}
    paths = {
        "results":    ROOT / "outputs" / "innovation_classification_results.csv",
        "recurrence": ROOT / "outputs" / "diffusion" / "innovation_type_recurrence.csv",
        "temporal":   ROOT / "outputs" / "diffusion" / "innovation_type_temporal_summary.csv",
        "confidence": ROOT / "outputs" / "diffusion" / "innovation_type_confidence_weighted.csv",
        "fingerprint":ROOT / "outputs" / "diffusion" / "innovation_type_tfidf_fingerprint.csv",
        "evaluation": ROOT / "evaluation" / "validation_evaluated.csv",
        "eval_summary":ROOT / "evaluation" / "evaluation_summary.csv",
    }
    for key, path in paths.items():
        if path.exists():
            d[key] = pd.read_csv(path)

    by_year_path = ROOT / "outputs" / "diffusion" / "innovation_type_by_year.csv"
    if by_year_path.exists():
        d["by_year"] = pd.read_csv(by_year_path, index_col=0)

    by_country_path = ROOT / "outputs" / "diffusion" / "innovation_type_by_country.csv"
    if by_country_path.exists():
        d["by_country"] = pd.read_csv(by_country_path, index_col=0)

    return d

data = load_data()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

NAVY   = "#0a1628"
COLORS = px.colors.qualitative.Set2

def get_eval_metric(key, fallback):
    es = data.get("eval_summary")
    if es is not None:
        row = es.loc[es["metric"] == key, "value"]
        if len(row):
            return float(row.values[0])
    return fallback

HARDCODED = {
    "multiclass_accuracy": 0.743,
    "macro_f1":            0.785,
    "binary_precision":    1.000,
    "binary_recall":       0.885,
    "binary_f1":           0.939,
}

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="app-header">
    <h1>Innovation Signal Analytics</h1>
    <p>World Bank Project Appraisal Documents &nbsp;·&nbsp; Sub-Saharan Africa &nbsp;·&nbsp; 2015 – 2025</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("### About")
    st.markdown("""
This prototype detects and analyzes innovation signals in World Bank
Project Appraisal Documents (PADs) using semantic retrieval, ontology-guided
LLM classification, and expert validation.

**Corpus:** 60-PAD prototype  
**Region:** Sub-Saharan Africa  
**Period:** 2015 – 2025
    """)
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("""
API ingestion → component extraction  
→ semantic embeddings → candidate retrieval  
→ LLM classification → expert validation  
→ diffusion analysis
    """)
    st.divider()
    st.markdown(
        "[GitHub repo](https://github.com/willibroad/innovation-analytics-poc)"
        " · [wbuma.com](https://wbuma.com)"
    )

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Geography", "Browse Records"])


# ==================================================
# TAB 1 — OVERVIEW
# ==================================================

with tab1:

    results = data.get("results")

    if results is not None:
        valid = results[results["innovation_presence"] != "error"].copy()
        signal_rate = valid["innovation_presence"].isin(
            ["innovation", "possible_innovation"]
        ).mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Classified units",        f"{len(valid)}")
        c2.metric("Innovation signal rate",  f"{signal_rate:.0%}")
        c3.metric("Macro F1",                f"{get_eval_metric('macro_f1', HARDCODED['macro_f1']):.3f}")
        c4.metric("Binary innovation F1",    f"{get_eval_metric('binary_f1', HARDCODED['binary_f1']):.3f}")

    st.divider()

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<p class="section-label">Innovation types — distinct projects</p>', unsafe_allow_html=True)
        rec = data.get("recurrence")
        if rec is not None:
            fig = px.bar(
                rec.sort_values("distinct_projects"),
                x="distinct_projects",
                y="innovation_type",
                orientation="h",
                color_discrete_sequence=[NAVY],
                labels={"distinct_projects": "Distinct projects", "innovation_type": ""},
            )
            fig.update_layout(
                margin=dict(l=0, r=20, t=10, b=0), height=280,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter", size=12),
                xaxis=dict(gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-label">Classification distribution</p>', unsafe_allow_html=True)
        if results is not None:
            dist = (
                valid["innovation_presence"]
                .value_counts()
                .reset_index()
            )
            dist.columns = ["Classification", "Count"]
            dist["Pct"] = (dist["Count"] / len(valid) * 100).round(1).astype(str) + "%"
            st.dataframe(dist, hide_index=True, use_container_width=True)

    st.markdown('<p class="section-label">Validation results — 35 expert-labelled units</p>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("4-class accuracy",  f"{get_eval_metric('multiclass_accuracy', HARDCODED['multiclass_accuracy']):.1%}")
    m2.metric("Macro F1",          f"{get_eval_metric('macro_f1',            HARDCODED['macro_f1']):.3f}")
    m3.metric("Binary precision",  f"{get_eval_metric('binary_precision',    HARDCODED['binary_precision']):.3f}")
    m4.metric("Binary recall",     f"{get_eval_metric('binary_recall',       HARDCODED['binary_recall']):.3f}")
    m5.metric("Binary F1",         f"{get_eval_metric('binary_f1',           HARDCODED['binary_f1']):.3f}")

    if Path(ROOT / "assets" / "confusion_matrix.png").exists():
        st.divider()
        st.markdown('<p class="section-label">Confusion matrix</p>', unsafe_allow_html=True)
        st.image(str(ROOT / "assets" / "confusion_matrix.png"), width=600)


# ==================================================
# TAB 2 — TRENDS
# ==================================================

with tab2:

    by_year = data.get("by_year")

    if by_year is not None:
        st.markdown('<p class="section-label">Innovation types across PAD years</p>', unsafe_allow_html=True)
        long = by_year.reset_index().melt(
            id_vars="document_year",
            var_name="Innovation type",
            value_name="Distinct projects",
        )
        fig = px.line(
            long, x="document_year", y="Distinct projects",
            color="Innovation type", markers=True,
            color_discrete_sequence=COLORS,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0), height=360,
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter", size=12),
            xaxis=dict(gridcolor="#f1f5f9", title="PAD year"),
            yaxis=dict(gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2)

    with cl:
        temporal = data.get("temporal")
        if temporal is not None:
            st.markdown('<p class="section-label">Observation span</p>', unsafe_allow_html=True)
            st.dataframe(
                temporal.rename(columns={
                    "innovation_type":      "Type",
                    "first_observed_year":  "First",
                    "last_observed_year":   "Last",
                    "distinct_years":       "Years",
                    "distinct_projects":    "Projects",
                }),
                hide_index=True, use_container_width=True,
            )

    with cr:
        confidence = data.get("confidence")
        if confidence is not None:
            st.markdown('<p class="section-label">Confidence-weighted signal strength</p>', unsafe_allow_html=True)
            st.dataframe(
                confidence[["innovation_type", "mean_confidence", "distinct_projects"]].rename(columns={
                    "innovation_type":   "Type",
                    "mean_confidence":   "Mean confidence",
                    "distinct_projects": "Projects",
                }),
                hide_index=True, use_container_width=True,
            )

    fingerprint = data.get("fingerprint")
    if fingerprint is not None:
        st.markdown('<p class="section-label">TF-IDF keyword fingerprint by innovation type</p>', unsafe_allow_html=True)
        st.dataframe(
            fingerprint.rename(columns={"innovation_type": "Type", "top_terms": "Top terms"}),
            hide_index=True, use_container_width=True,
        )


# ==================================================
# TAB 3 — GEOGRAPHY
# ==================================================

with tab3:

    by_country = data.get("by_country")

    if by_country is not None:
        selected = st.selectbox(
            "Filter by innovation type",
            ["All types"] + list(by_country.columns),
        )

        if selected == "All types":
            display = by_country.copy()
            display["Total"] = display.sum(axis=1)
            display = display.sort_values("Total", ascending=False)
        else:
            display = by_country[[selected]].copy()
            display = display[display[selected] > 0].sort_values(selected, ascending=False)

        st.markdown('<p class="section-label">Signals by country</p>', unsafe_allow_html=True)
        st.dataframe(display, use_container_width=True)

        if selected != "All types":
            chart_df = display.reset_index()
            chart_df.columns = ["Country", "Projects"]
            fig = px.bar(
                chart_df.head(15).sort_values("Projects"),
                x="Projects", y="Country", orientation="h",
                color_discrete_sequence=[NAVY],
            )
            fig.update_layout(
                margin=dict(l=0, r=20, t=10, b=0), height=400,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter", size=12),
                xaxis=dict(gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ==================================================
# TAB 4 — BROWSE RECORDS
# ==================================================

with tab4:

    results = data.get("results")

    if results is not None:
        valid = results[results["innovation_presence"] != "error"].copy()
        valid["confidence"] = pd.to_numeric(valid["confidence"], errors="coerce")

        f1, f2, f3 = st.columns(3)

        with f1:
            presence_opts = ["All"] + sorted(valid["innovation_presence"].dropna().unique())
            sel_presence = st.selectbox("Innovation presence", presence_opts)

        with f2:
            conf_min = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

        with f3:
            search = st.text_input("Search title", placeholder="digital, irrigation...")

        filtered = valid.copy()

        if sel_presence != "All":
            filtered = filtered[filtered["innovation_presence"] == sel_presence]

        filtered = filtered[filtered["confidence"] >= conf_min]

        if search:
            filtered = filtered[
                filtered["title"].str.contains(search, case=False, na=False)
            ]

        st.markdown(
            f'<p class="section-label">{len(filtered)} records'
            f'{"  (showing first 50)" if len(filtered) > 50 else ""}</p>',
            unsafe_allow_html=True,
        )

        for _, row in filtered.head(50).iterrows():
            label = (
                f"**{row['projectid']}** · "
                f"{row['type']} {row['number']} · "
                f"{str(row.get('title', ''))[:90]}"
            )
            with st.expander(label):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Presence:** `{row['innovation_presence']}`")
                    st.markdown(f"**Types:** {row.get('innovation_types', '')}")
                    st.markdown(f"**Novelty:** {row.get('novelty_basis', '')}")
                with c2:
                    st.markdown(f"**Maturity:** {row.get('maturity_stage', '')}")
                    st.markdown(f"**Confidence:** {row.get('confidence', '')}")
                    st.markdown(f"**Evidence strength:** {row.get('evidence_strength', '')}")
                if row.get("rationale"):
                    st.markdown("**Rationale**")
                    st.info(row["rationale"])
                if row.get("evidence_text"):
                    st.caption(f"Evidence: {row['evidence_text']}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Prototype &nbsp;·&nbsp; Innovation Signal Analytics &nbsp;·&nbsp;
    <a href="https://wbuma.com" target="_blank">wbuma.com</a>
</div>
""", unsafe_allow_html=True)