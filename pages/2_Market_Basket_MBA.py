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
# 3. Kosár-szerkezet a konvertáló rendelések között
# -----------------------------------------------------------------------------
st.subheader("3. Cross-sell megosztás a konvertáló rendelések között")

# A generált adatban a kosár csak konvertáló útvonalakhoz csatolódik, ezért a
# cross-sell cohort konverziós ráta feltétlenül 100% lenne — az nem tájékoztató.
# Ehelyett a cross-sell megosztását a konvertáló rendelések között mutatjuk, és
# különbszét tesszük a megerősített egytermékes és a nem csatolt útvonalakat.
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
c1.metric("Konvertáló rendelések", f"{n_converted:,}")
c2.metric("Cross-sell rendelések", f"{n_cs:,}")
c2.caption(f"{cs_share:.1f}% a konvertáló rendelések közül")
c3.metric("Egytermékes rendelések", f"{n_single:,}")
c3.caption(f"{single_share:.1f}% a konvertáló rendelések közül")
c4.metric("Nem csatolt konvertáló útvonalak", f"{n_unmatched:,}")

# -----------------------------------------------------------------------------
# 4. Sparse Markov motor cross-sell súlyozott Removal Effect
# -----------------------------------------------------------------------------
st.subheader("4. Érték-súlyozott Removal Effect (cross-sell konverziókra)")
st.markdown(
    "Csak a **cross_sell_conversions** oszlopot használjuk súlyként, így a Removal Effect "
    "megjeleníti, melyik csatorna kiesése veszíti le a legtöbb keresztértékesítést."
)

# A Removal Effect csak a cross-sell vásárlásokra számítódik. A konvertáló rendelések
# között ezek a cross-sell maszk által kiválasztott rekordok.
cs_weighted = merged.loc[cs_mask].copy()
cs_weighted["conversions"] = cs_weighted["cross_sell_conversions"]

if cs_weighted.empty or cs_weighted["conversions"].sum() == 0:
    st.info("Nincs cross-sell konverzió, ezért nem áll rendelkezésre Removal Effect.")
else:
    engine = MarkovJourneyEngine(cs_weighted, k=2, collapse_loops=True)
    base_cvr = engine.solve_prob(weight_col="conversions")

    rows = []
    for ch in engine.channels:
        rem = engine.solve_prob(weight_col="conversions", drop=[ch])
        re_ = 1.0 - (rem / base_cvr) if base_cvr > 0 else 0.0
        rows.append({"Csatorna": ch, "Cross-sell Removal Effect (%)": max(0.0, re_) * 100})

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

st.caption(
    "Módszer: az MBA tábla alapján csak a cross-sell kosárak konverzióit használjuk a Markov "
    "átmeneti mátrix súlyozásához. Ez a cross-sell vásárlói csoporton belüli csatornafüggőséget "
    "jelzi, nem pedig ok-okozatilag bizonyított cross-sell növekedést."
)
