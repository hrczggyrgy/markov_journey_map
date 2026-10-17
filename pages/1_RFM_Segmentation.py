"""RFM Segmentation & Transition Drift — RFM → Markov segment comparison.

Compares two RFM segment transition probability matrices (ΔP = P_A − P_B),
displaying drop-off points in a heatmap where new customers differ from VIPs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.markov_engine import MarkovJourneyEngine
from core.data_loader import load_rfm_segments
from core.bridge_connectors import compare_rfm_transition_matrices

st.set_page_config(page_title="RFM Segmentation", layout="wide")

st.title("RFM Segmentation & Transition Drift")
st.markdown(
    "**RFM → Markov integration.** Difference of two RFM segment transition "
    "probability matrices (Δ**P** = **P**ₐ − **P**ᵦ). Positive value: the step is "
    "more common in the first segment. Red/green heatmap immediately shows where "
    "new prospects drop off compared to VIPs."
)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data
def _load_data():
    return load_rfm_segments("data_lake/unified_customers_rfm.parquet")

rfm_df = _load_data()

# Journey data loading (simplified: all journeys in one table)
import duckdb
journeys = duckdb.execute(
    "SELECT path, rfm_segment, SUM(conversions) AS conversions, SUM(nulls) AS nulls, "
    "SUM(revenue) AS revenue, SUM(net_margin) AS net_margin "
    "FROM read_parquet('data_lake/customer_journeys.parquet') "
    "GROUP BY path, rfm_segment"
).df()

if journeys.empty:
    st.warning("No journey data.")
    st.stop()

st.subheader("1. RFM segment overview")
seg_stats = (
    rfm_df.groupby("rfm_segment")
    .agg(users=("user_id", "count"), average_monetary=("monetary", "mean"))
    .reset_index()
)
st.dataframe(seg_stats, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Segment pair selection
# -----------------------------------------------------------------------------
segments = sorted(journeys["rfm_segment"].unique().tolist())
col_sel1, col_sel2 = st.columns(2)
vip_idx = segments.index("VIP (Champions)") if "VIP (Champions)" in segments else 0
with col_sel1:
    seg_a = st.selectbox("Segment A", segments, index=vip_idx)
with col_sel2:
    other = [s for s in segments if s != seg_a]
    default_b = other[0] if other else seg_a
    seg_b = st.selectbox("Segment B", segments, index=segments.index(default_b))

if seg_a == seg_b:
    st.warning("Select two different segments.")
    st.stop()

# -----------------------------------------------------------------------------
# Delta matrix computation
# -----------------------------------------------------------------------------
with st.spinner(f"Computing transition matrices: {seg_a} vs {seg_b}..."):
    delta_P = compare_rfm_transition_matrices(journeys, seg_a, seg_b, path_sep=">", k=2)

st.subheader(f"2. Transition matrix difference: {seg_a} − {seg_b} (percentage points)")
st.markdown(
    f"Positive (green): {seg_a} more characteristic → compared to {seg_b}. "
    f"Negative (red): {seg_b} more characteristic."
)

# Filter: only larger absolute differences
threshold = st.slider("Minimum absolute difference (percentage points):", 0.0, 20.0, 2.0, 0.5)

# Heatmap: highlight only different cells
plot_df = delta_P.copy()
masked = plot_df.where(plot_df.abs() >= threshold)

fig = px.imshow(
    masked,
    text_auto=".1f",
    aspect="auto",
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    labels=dict(x="Target state", y="Source state", color="ΔP (pp)"),
    title=f"Transition Drift: {seg_a} minus {seg_b}",
)
fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# Top differences table
st.subheader("3. Largest differences (Top 15)")
long_df = (
    delta_P.stack()
    .reset_index()
    .rename(columns={"level_0": "Source", "level_1": "Target", 0: "ΔP_pp"})
)
long_df["abs"] = long_df["ΔP_pp"].abs()
top_diff = long_df.sort_values("abs", ascending=False).head(15)
st.dataframe(
    top_diff[["Source", "Target", "ΔP_pp"]].style.format({"ΔP_pp": "{:+.2f} pp"}),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Method: A separate k=2 order sparse Markov matrix is built for each segment, "
    "then mapped onto a common state space and subtracted from each other "
    "(ΔP · 100 = percentage point difference)."
)
