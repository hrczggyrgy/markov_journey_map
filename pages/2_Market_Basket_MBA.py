"""Market Basket (MBA) → Markov: kosár-affinitási útvonalak.

Összeköti az útvonalakat az MBA kimenettel, így külön vizsgálható, melyik kampány-
útvonal készíti elő a magas Lift-értékű keresztértékesítési (cross-sell) kosarakat.
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
    "**MBA → Markov integráció.** Az útvonalakat felcímkézzük az MBA-kosár elemzéssel "
    "(Cross-sell bundle / Egytermékes), és külön vizsgáljuk, mely marketing-útvonal "
    "vezeti a többtermékes, magas árrésű vásárlásokhoz."
)

# -----------------------------------------------------------------------------
# Adat betöltés
# -----------------------------------------------------------------------------
import duckdb

mba_df = load_mba_rules("data_lake/transaction_baskets_mba.parquet")
journeys_raw = duckdb.execute(
    "SELECT path_id, path, rfm_segment, SUM(conversions) AS conversions, SUM(nulls) AS nulls, "
    "SUM(revenue) AS revenue, SUM(net_margin) AS net_margin "
    "FROM read_parquet('data_lake/customer_journeys.parquet') GROUP BY path_id, path, rfm_segment"
).df()

if journeys_raw.empty or mba_df.empty:
    st.warning("Nincs elég adat.")
    st.stop()

# -----------------------------------------------------------------------------
# 1. MBA áttekintés
# -----------------------------------------------------------------------------
st.subheader("1. MBA kosár-elemzés áttekintés")
cs = mba_df["is_cross_sell_bundle"].value_counts()
col1, col2, col3 = st.columns(3)
col1.metric("Összes rendelés", f"{len(mba_df):,}")
col2.metric("Cross-sell kosárak", f"{int(cs.get(True, 0)):,}")
col3.metric("Átlagos nettó fedezet", f"{mba_df['net_margin'].mean():,.0f} Ft")

fig_bundle = px.pie(
    mba_df,
    names="bundle_type",
    title="Kosártípus-eloszlás",
    hole=0.4,
)
fig_bundle.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=20))
st.plotly_chart(fig_bundle, use_container_width=True)

# -----------------------------------------------------------------------------
# 2. Útvonalak enrichment-je MBA-val
# -----------------------------------------------------------------------------
st.subheader("2. Útvonalak enrichment-je MBA kosár-sorral")
merged = enrich_journeys_with_mba_rules(journeys_raw, mba_df)

st.dataframe(
    merged[["path_id", "path", "is_cross_sell_bundle", "bundle_type", "conversions", "net_margin", "cross_sell_conversions"]].head(10),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# 3. Markov illesztés külön cross-sell vs egytermékes kosarakra
# -----------------------------------------------------------------------------
st.subheader("3. Markov konverziós ráta: Cross-sell vs. Egytermékes kosár")

def _cvr(df_part):
    if df_part.empty:
        return 0.0, 0
    conv = float(df_part["conversions"].sum())
    nulls = float(df_part["nulls"].sum())
    return conv / (conv + nulls) if (conv + nulls) > 0 else 0.0, int(conv + nulls)

cs_df = merged[merged["is_cross_sell_bundle"] == True]
non_cs_df = merged[merged["is_cross_sell_bundle"] != True]

cvr_cs, n_cs = _cvr(cs_df)
cvr_non, n_non = _cvr(non_cs_df)

c1, c2, c3 = st.columns(3)
c1.metric("Cross-sell konverziós ráta", f"{cvr_cs*100:.2f}%", f"{n_cs:,} útvonal")
c2.metric("Egytermékes konverziós ráta", f"{cvr_non*100:.2f}%", f"{n_non:,} útvonal")
c3.metric("Különbözet (pp)", f"{(cvr_cs-cvr_non)*100:+.2f} pp")

# -----------------------------------------------------------------------------
# 4. Sparse Markov motor cross-sell súlyozott Removal Effect
# -----------------------------------------------------------------------------
st.subheader("4. Érték-súlyozott Removal Effect (cross-sell konverziókra)")
st.markdown(
    "Csak a **cross_sell_conversions** oszlopot használjuk súlyként, így a Removal Effect "
    "megjeleníti, melyik csatorna kiesése veszíti le a legtöbb keresztértékesítést."
)

cs_weighted = cs_df.copy()
cs_weighted["conversions"] = cs_weighted["cross_sell_conversions"]

engine = MarkovJourneyEngine(cs_weighted, k=2, collapse_loops=True)
base_cvr = engine.solve_prob(weight_col="conversions")

rows = []
for ch in engine.channels:
    rem = engine.solve_prob(weight_col="conversions", drop=[ch])
    re_ = 1.0 - (rem / base_cvr) if base_cvr > 0 else 0.0
    rows.append({"Csatorna": ch, "Cross-sell Removal Effect (%)": max(0.0, re_) * 100})

if rows:
    re_df = pd.DataFrame(rows).sort_values("Cross-sell Removal Effect (%)", ascending=False)
    fig_re = px.bar(
        re_df,
        x="Csatorna",
        y="Cross-sell Removal Effect (%)",
        text_auto=".1f",
        color="Cross-sell Removal Effect (%)",
        color_continuous_scale="Blues",
        title="Melyik csatorna kiesése veszítheti le a legtöbb cross-sell konverziót",
    )
    fig_re.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig_re, use_container_width=True)

    st.dataframe(
        re_df.style.format({"Cross-sell Removal Effect (%)": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Nincs cross-sell konverzió, ezért nem áll rendelkezésre Removal Effect.")

st.caption(
    "Módszer: az MBA tábla alapján csak a cross-sell kosárak konverzióit használjuk a Markov "
    "átmeneti mátrix súlyozásához, így az attribution a többtermékes vásárlásokhoz igazodik."
)
