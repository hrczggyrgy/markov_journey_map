"""MMM Prior Calibration — Markov → MMM Bayesian prior calibration.

Based on channel roles calculated from Markov Removal Effect and daily spends,
we generate a LogNormal prior for channel coefficients, which can be passed
to PyMC-Marketing or a custom MMM model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json
import duckdb

from core.markov_engine import MarkovJourneyEngine
from core.bridge_connectors import export_markov_priors_for_mmm

st.set_page_config(page_title="MMM Prior Calibration", layout="wide")

st.title("MMM Prior Calibration")
st.markdown(
    "**Markov → MMM integration.** Based on the Journey Mapper Removal Effect and "
    "channel spends, we generate the Bayesian prior (LogNormal) for channel "
    "coefficients, which can help with MMM multicollinearity (e.g. Social Ads "
    "and Search moving together)."
)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    confidence_weight = st.slider(
        "Prior confidence weight (confidence_weight)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="The higher, the narrower (stronger) the Markov-based prior for MMM."
    )
    st.markdown("---")
    st.markdown("**Data source:** `data_lake/channel_daily_spend_mmm.parquet`")

# Daily spend aggregated by channel (parquet is already at daily level)
# Custom spend dict built from the channel column:
spend_raw = duckdb.execute(
    "SELECT channel, SUM(spend) AS total_spend FROM read_parquet('data_lake/channel_daily_spend_mmm.parquet') GROUP BY channel"
).df()
spend_dict = dict(zip(spend_raw["channel"], spend_raw["total_spend"]))

st.subheader("1. Channel spends")
st.dataframe(
    spend_raw.sort_values("total_spend", ascending=False).style.format(
        {"total_spend": "{:,.0f} Ft"}
    ),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# Markov fitting on journeys (all paths)
# -----------------------------------------------------------------------------
journeys = duckdb.execute(
    "SELECT path, SUM(conversions) AS conversions, SUM(nulls) AS nulls, SUM(revenue) AS revenue, SUM(net_margin) AS net_margin "
    "FROM read_parquet('data_lake/customer_journeys.parquet') GROUP BY path"
).df()

if journeys.empty:
    st.warning("No journey data.")
    st.stop()

engine = MarkovJourneyEngine(journeys, k=2, collapse_loops=True)
attr_df = engine.compute_attribution()

# Fill Markov Revenue (Ft) into attr_df (total revenue prorated by Volume_RE)
total_rev = float(journeys["revenue"].sum())
if not attr_df.empty and attr_df["Volume_RE"].sum() > 0:
    attr_df["Markov Revenue (Ft)"] = (
        attr_df["Volume_RE"] / attr_df["Volume_RE"].sum() * total_rev
    )
else:
    attr_df["Markov Revenue (Ft)"] = 0.0

st.subheader("2. Markov Attribution (Removal Effect)")
st.dataframe(
    attr_df[["Channel", "Volume_RE_pct", "Margin_RE_pct", "Markov Revenue (Ft)"]].style.format(
        {"Volume_RE_pct": "{:.2f}%", "Margin_RE_pct": "{:.2f}%", "Markov Revenue (Ft)": "{:,.0f} Ft"}
    ),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# Bayesian prior generation
# -----------------------------------------------------------------------------
st.subheader("3. Bayesian priors for MMM")
priors = export_markov_priors_for_mmm(attr_df, spend_dict, confidence_weight=confidence_weight)

if not priors:
    st.info("No valid channel for spend.")
else:
    prior_df = pd.DataFrame([
        {
            "Channel": ch,
            "Markov ROAS": v["markov_roas_estimate"],
            "Distribution": v["prior_distribution"],
            "μ (mu)": v["mu"],
            "σ (sigma)": v["sigma"],
        }
        for ch, v in priors.items()
    ]).sort_values("Markov ROAS", ascending=False)

    st.dataframe(
        prior_df.style.format({
            "Markov ROAS": "{:.3f}",
            "μ (mu)": "{:.4f}",
            "σ (sigma)": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        prior_df,
        x="Channel",
        y="Markov ROAS",
        color="σ (sigma)",
        color_continuous_scale="RdYlGn_r",
        title="Markov-based ROAS estimate by channel (prior basis)",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "μ and σ are LogNormal prior parameters: mu = ln(roas) − 0.5·σ², "
        "σ = max(0.15, 1 − confidence_weight). These can be passed directly "
        "to the PyMC-Marketing MMM model channel prior."
    )

    st.download_button(
        label="💾 Download priors as JSON",
        data=__import__("json").dumps(priors, indent=2, ensure_ascii=False),
        file_name="markov_mmm_priors.json",
        mime="application/json",
    )
