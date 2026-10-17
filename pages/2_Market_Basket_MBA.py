"""Market Basket (MBA) → Markov: basket affinity paths.

Connects journeys to MBA output so that which campaign path leads to
high-Lift cross-sell baskets can be analyzed separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.markov_engine import MarkovJourneyEngine
from core.data_loader import load_mba_rules, load_aggregated_edges
from core.bridge_connectors import enrich_journeys_with_mba_rules

st.set_page_config(page_title="Market Basket MBA", layout="wide")

st.title("Market Basket (MBA) → Markov")
st.markdown(
    "**MBA → Markov integration.** Journeys are labeled with MBA basket analysis "
    "(Cross-sell bundle / Single-product), and separately examines which marketing "
    "path leads to multi-product, high-margin purchases."
)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
import duckdb

mba_df = load_mba_rules("data_lake/transaction_baskets_mba.parquet")
journeys_raw = duckdb.execute(
    "SELECT path_id, path, rfm_segment, SUM(conversions) AS conversions, SUM(nulls) AS nulls, "
    "SUM(revenue) AS revenue, SUM(net_margin) AS net_margin "
    "FROM read_parquet('data_lake/customer_journeys.parquet') GROUP BY path_id, path, rfm_segment"
).df()

if journeys_raw.empty or mba_df.empty:
    st.warning("Not enough data.")
    st.stop()

# -----------------------------------------------------------------------------
# 1. MBA overview
# -----------------------------------------------------------------------------
st.subheader("1. MBA basket analysis overview")
cs = mba_df["is_cross_sell_bundle"].value_counts()
col1, col2, col3 = st.columns(3)
col1.metric("Total orders", f"{len(mba_df):,}")
col2.metric("Cross-sell baskets", f"{int(cs.get(True, 0)):,}")
col3.metric("Average net margin", f"{mba_df['net_margin'].mean():,.0f} Ft")

fig_bundle = px.pie(
    mba_df,
    names="bundle_type",
    title="Basket type distribution",
    hole=0.4,
)
fig_bundle.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=20))
st.plotly_chart(fig_bundle, use_container_width=True)

# -----------------------------------------------------------------------------
# 2. Journey enrichment with MBA
# -----------------------------------------------------------------------------
st.subheader("2. Journey enrichment with MBA basket")
merged = enrich_journeys_with_mba_rules(journeys_raw, mba_df)

st.dataframe(
    merged[["path_id", "path", "is_cross_sell_bundle", "bundle_type", "conversions", "net_margin", "cross_sell_conversions"]].head(10),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# 3. Basket structure among converted orders
# -----------------------------------------------------------------------------
st.subheader("3. Cross-sell share among converted orders")

# In the generated data, baskets are only attached to converted journeys, so the
# cross-sell cohort conversion rate would necessarily be 100% — not informative.
# Instead, we show the cross-sell share among converted orders and distinguish
# confirmed single-product from unmatched journeys.
conv_mask = merged["conversions"] == 1
n_converted = int(conv_mask.sum())
cs_mask = conv_mask & (merged["is_cross_sell_bundle"] == True)
single_mask = conv_mask & (merged["is_cross_sell_bundle"] == False)
unmatched_mask = conv_mask & merged["is_cross_sell_bundle"].isna()

n_cs = int(cs_mask.sum())
n_single = int(single_mask.sum())
n_unmatched = int(unmatched_mask.sum())
cs_share = (n_cs / n_converted * 100) if n_converted > 0 else 0.0
single_share = (n_single / n_converted * 100) if n_converted > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Converted orders", f"{n_converted:,}")
c2.metric("Cross-sell orders", f"{n_cs:,}")
c2.caption(f"{cs_share:.1f}% of converted orders")
c3.metric("Single-product orders", f"{n_single:,}")
c3.caption(f"{single_share:.1f}% of converted orders")
c4.metric("Unmatched converted paths", f"{n_unmatched:,}")

# -----------------------------------------------------------------------------
# 4. Sparse Markov engine cross-sell weighted Removal Effect
# -----------------------------------------------------------------------------
st.subheader("4. Value-weighted Removal Effect (cross-sell conversions)")
st.markdown(
    "Only the **cross_sell_conversions** column is used as weight, so the Removal "
    "Effect shows which channel removal loses the most cross-sell conversions."
)

# Removal Effect is computed only on cross-sell purchases. Among converted orders,
# these are the records selected by the cross-sell mask.
cs_weighted = merged.loc[cs_mask].copy()
cs_weighted["conversions"] = cs_weighted["cross_sell_conversions"]

if cs_weighted.empty or cs_weighted["conversions"].sum() == 0:
    st.info("No cross-sell conversions, so Removal Effect is not available.")
else:
    engine = MarkovJourneyEngine(cs_weighted, k=2, collapse_loops=True)
    base_cvr = engine.solve_prob(weight_col="conversions")

    rows = []
    for ch in engine.channels:
        rem = engine.solve_prob(weight_col="conversions", drop=[ch])
        re_ = 1.0 - (rem / base_cvr) if base_cvr > 0 else 0.0
        rows.append({"Channel": ch, "Cross-sell Removal Effect (%)": max(0.0, re_) * 100})

    re_df = pd.DataFrame(rows).sort_values("Cross-sell Removal Effect (%)", ascending=False)

    fig_re = px.bar(
        re_df,
        x="Channel",
        y="Cross-sell Removal Effect (%)",
        text_auto=".1f",
        color="Cross-sell Removal Effect (%)",
        color_continuous_scale="Blues",
        title="Which channel removal loses the most cross-sell conversions",
    )
    fig_re.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig_re, use_container_width=True)

    st.dataframe(
        re_df.style.format({"Cross-sell Removal Effect (%)": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Method: based on the MBA table, only cross-sell basket conversions are used to "
    "weight the Markov transition matrix. This indicates channel dependence within "
    "the observed cross-sell buyer group — not causally proven cross-sell growth."
)
