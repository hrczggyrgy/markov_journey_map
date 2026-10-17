"""RFM Segmentation & Transition Drift — RFM → Markov szegmens-hasonlítás.

Két RFM szegmens átmeneti valószínűségi mátrixát comparesolja (ΔP = P_A − P_B),
hőtérképen jeleníti meg a lelépési pontokat, ahol az új vásárlók eltérnek a VIP-ktől.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.markov_engine import MarkovJourneyEngine
from core.data_loader import load_rfm_segments
from core.bridge_connectors import compare_rfm_transition_matrices

st.set_page_config(page_title="RFM Segmentation", page_icon="🎯", layout="wide")

st.title("🎯 RFM Segmentation & Transition Drift")
st.markdown(
    "**RFM → Markov integráció.** Két RFM szegmens átmeneti valószínűségi mátrixának "
    "különbsége (Δ**P** = **P**ₐ − **P**ᵦ). Pozitív érték: a primera szegmensben "
    "gyakrabban fordul elő az adott lépés. Piros/zöld hőtérképen azonnal látszik, "
    "melyik útvonal-nál szakadnak le az új érdeklődők a VIP-hez képest."
)

# -----------------------------------------------------------------------------
# Adat betöltés
# -----------------------------------------------------------------------------
@st.cache_data
def _load_data():
    return load_rfm_segments("data_lake/unified_customers_rfm.parquet")

rfm_df = _load_data()

# Journey adatok betöltése (egyszerűsített: az összes journey egy tábla)
import duckdb
journeys = duckdb.execute(
    "SELECT path, rfm_segment, SUM(conversions) AS conversions, SUM(nulls) AS nulls, "
    "SUM(revenue) AS revenue, SUM(net_margin) AS net_margin "
    "FROM read_parquet('data_lake/customer_journeys.parquet') "
    "GROUP BY path, rfm_segment"
).df()

if journeys.empty:
    st.warning("Nincs journey adat.")
    st.stop()

st.subheader("1️⃣ RFM szegmens áttekintés")
seg_stats = (
    rfm_df.groupby("rfm_segment")
    .agg(users=("user_id", "count"), avg_clv=("monetary", "mean"))
    .reset_index()
)
st.dataframe(seg_stats, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Szegmens pár kiválasztása
# -----------------------------------------------------------------------------
segments = sorted(journeys["rfm_segment"].unique().tolist())
col_sel1, col_sel2 = st.columns(2)
vip_idx = segments.index("VIP (Champions)") if "VIP (Champions)" in segments else 0
with col_sel1:
    seg_a = st.selectbox("Szegmens A", segments, index=vip_idx)
with col_sel2:
    other = [s for s in segments if s != seg_a]
    default_b = other[0] if other else seg_a
    seg_b = st.selectbox("Szegmens B", segments, index=segments.index(default_b))

if seg_a == seg_b:
    st.warning("Válassz két különböző szegmenst.")
    st.stop()

# -----------------------------------------------------------------------------
# Delta mátrix számítása
# -----------------------------------------------------------------------------
with st.spinner(f"Átmeneti mátrixok számítása: {seg_a} vs {seg_b}..."):
    delta_P = compare_rfm_transition_matrices(journeys, seg_a, seg_b, path_sep=">", k=2)

st.subheader(f"2️⃣ Átmeneti mátrix különbség: {seg_a} − {seg_b} (százalékpont)")
st.markdown(
    f"Pozitív (zöld): {seg_a} jellemzőbb → {seg_b}hez képest. "
    f"Negatív (piros): {seg_b} jellemzőbb."
)

# Szűrő: csak nagyobb abszolút különbségek
threshold = st.slider("Minimum abszolút eltérés (százalékpont):", 0.0, 20.0, 2.0, 0.5)

# Hőtérkép: csak a különböző cellák kiemelése
plot_df = delta_P.copy()
masked = plot_df.where(plot_df.abs() >= threshold)

fig = px.imshow(
    masked,
    text_auto=".1f",
    aspect="auto",
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    labels=dict(x="Cél állapot", y="Forrás állapot", color="ΔP (pp)"),
    title=f"Transition Drift: {seg_a} minus {seg_b}",
)
fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# Top eltérék táblázat
st.subheader("3️⃣ Legnagyobb eltérések (Top 15)")
long_df = (
    delta_P.stack()
    .reset_index()
    .rename(columns={"level_0": "Forrás", "level_1": "Cél", 0: "ΔP_pp"})
)
long_df["abs"] = long_df["ΔP_pp"].abs()
top_diff = long_df.sort_values("abs", ascending=False).head(15)
st.dataframe(
    top_diff[["Forrás", "Cél", "ΔP_pp"]].style.format({"ΔP_pp": "{:+.2f} pp"}),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Módszer: Mindkét szegmenshez külön k=2 rendű sparse Markov-matix épül, majd közös "
    "állapottérre rushkítjuk és kivonjuk egymásból (ΔP · 100 = százalékpontos különbség)."
)
