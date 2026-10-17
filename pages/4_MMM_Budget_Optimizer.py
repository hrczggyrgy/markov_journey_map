"""MMM Budget Optimizer — Markov → MMM Bayes-i prior kalibráció.

A Markov Removal Effect alapján kiszámolt csatorna-szerepeket és a napi költéseket
használva előállítunk LogNormal priort a csatornaegyütthatókre, amelyeket átadhatunk
a PyMC-Marketing vagy saját MMM modellednek.
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

st.set_page_config(page_title="MMM Budget Optimizer", page_icon="📈", layout="wide")

st.title("📈 MMM Budget Optimizer")
st.markdown(
    "**Markov → MMM integráció.** A Journey Mapper Removal Effect-ja és a csatorna-költségek "
    "alapján előállítjuk a Bayes-i priort (LogNormal) a csatornaegyütthatókra, amellyel "
    "feloldható a MMM multikollinearitási問題a (pl. Social Ads és Search egyidézik mozog)."
)

# -----------------------------------------------------------------------------
# Adat betöltés
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Beállítások")
    confidence_weight = st.slider(
        "Prior hitelességi súlya (confidence_weight)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="Minél magasabb, annál szűkebb (erősebb) a Markov-alapú prior az MMM-hez."
    )
    st.markdown("---")
    st.markdown("**Adatforrás:** `data_lake/channel_daily_spend_mmm.parquet`")

# Napi költés aggregálása csatornánként (a parquet már napi szintre van)
# Egyéni spend dict összeállítása a channel oszlopból:
spend_raw = duckdb.execute(
    "SELECT channel, SUM(spend) AS total_spend FROM read_parquet('data_lake/channel_daily_spend_mmm.parquet') GROUP BY channel"
).df()
spend_dict = dict(zip(spend_raw["channel"], spend_raw["total_spend"]))

st.subheader("1️⃣ Csatorna-költségek")
st.dataframe(
    spend_raw.sort_values("total_spend", ascending=False).style.format(
        {"total_spend": "{:,.0f} Ft"}
    ),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# Markov illesztés a journeyson (összes útvonal)
# -----------------------------------------------------------------------------
journeys = duckdb.execute(
    "SELECT path, SUM(conversions) AS conversions, SUM(nulls) AS nulls, SUM(revenue) AS revenue, SUM(net_margin) AS net_margin "
    "FROM read_parquet('data_lake/customer_journeys.parquet') GROUP BY path"
).df()

if journeys.empty:
    st.warning("Nincs journey adat.")
    st.stop()

engine = MarkovJourneyEngine(journeys, k=2, collapse_loops=True)
attr_df = engine.compute_attribution()

# Markov Bevétel (Ft) feltöltése az attr_df-be (a teljes bevétel arányosítása Volume_RE szerint)
total_rev = float(journeys["revenue"].sum())
if not attr_df.empty and attr_df["Volume_RE"].sum() > 0:
    attr_df["Markov Bevétel (Ft)"] = (
        attr_df["Volume_RE"] / attr_df["Volume_RE"].sum() * total_rev
    )
else:
    attr_df["Markov Bevétel (Ft)"] = 0.0

st.subheader("2️⃣ Markov Attribúció (Removal Effect)")
st.dataframe(
    attr_df[["Csatorna", "Volume_RE_pct", "Margin_RE_pct", "Markov Bevétel (Ft)"]].style.format(
        {"Volume_RE_pct": "{:.2f}%", "Margin_RE_pct": "{:.2f}%", "Markov Bevétel (Ft)": "{:,.0f} Ft"}
    ),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# Bayes-i prior generálása
# -----------------------------------------------------------------------------
st.subheader("3️⃣ Bayes-i Priorok az MMM-hez")
priors = export_markov_priors_for_mmm(attr_df, spend_dict, confidence_weight=confidence_weight)

if not priors:
    st.info("Nincs egyedüli érvényes csatorna spend델 광áshoz.")
else:
    prior_df = pd.DataFrame([
        {
            "Csatorna": ch,
            "Markov ROAS": v["markov_roas_estimate"],
            "Elosztás": v["prior_distribution"],
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
        x="Csatorna",
        y="Markov ROAS",
        color="σ (sigma)",
        color_continuous_scale="RdYlGn_r",
        title="Markov-alapú ROAS-becslés csatornánként (prior alap)",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "A μ és σ egy LogNormal prior paraméterei: mu = ln(roas) − 0.5·σ², σ = max(0.15, 1 − confidence_weight). "
        "Ezeket közvetlenül átadhatod a PyMC-Marketing MMM modell channel-priorjaként."
    )

    st.download_button(
        label="💾 Priors letöltése JSON-ként",
        data=__import__("json").dumps(priors, indent=2, ensure_ascii=False),
        file_name="markov_mmm_priors.json",
        mime="application/json",
    )
