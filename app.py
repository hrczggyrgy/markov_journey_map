import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

from core.markov_engine import (
    MarkovJourneyEngine,
    clean_journey,
    make_k_order_states,
    CONVERSION,
    NULL_STATE,
)
from core.data_loader import (
    load_aggregated_edges,
    load_rfm_segments,
    load_mba_rules,
    load_csv_journeys,
)
from core.bridge_connectors import (
    export_markov_priors_for_mmm,
    compare_rfm_transition_matrices,
    enrich_journeys_with_mba_rules,
)

# -----------------------------------------------------------------------------
# OLDAL KONFIGURÁCIÓ
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Markov Journey Mapper — Sparse Higher-Order Engine",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. SZINTETIKUS DEMÓ ADATGENERÁTOR (RFM + MARGIN)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_demo_data() -> pd.DataFrame:
    data = [
        ("Social_Ads > Blog > Search_NonBrand > Email", "Új vásárlók", 140, 420, 24000),
        ("Social_Ads > Search_NonBrand > Search_Brand", "Új vásárlók", 190, 510, 26500),
        ("Social_Ads > Blog", "Új vásárlók", 15, 890, 18000),
        ("Social_Ads", "Új vásárlók", 20, 1450, 15000),
        ("Display_Ads > Social_Ads > Search_NonBrand > Retargeting", "Új vásárlók", 95, 380, 29000),
        ("Display_Ads", "Új vásárlók", 5, 1100, 14000),
        ("Search_NonBrand > Blog > Email > Search_Brand", "Potenciális hűségesek", 210, 310, 34000),
        ("Search_NonBrand > Retargeting > Search_Brand", "Potenciális hűségesek", 175, 290, 31000),
        ("Search_NonBrand", "Potenciális hűségesek", 45, 720, 22000),
        ("Email > Search_Brand", "VIP (Champions)", 420, 180, 52000),
        ("Search_Brand > Email", "VIP (Champions)", 310, 120, 48000),
        ("Email", "VIP (Champions)", 260, 240, 45000),
        ("Search_Brand", "VIP (Champions)", 290, 190, 49000),
        ("Retargeting > Email > Search_Brand", "Lemorzsolódó (At Risk)", 130, 410, 36000),
        ("Social_Ads > Retargeting > Email", "Lemorzsolódó (At Risk)", 85, 520, 27500),
        ("Retargeting", "Lemorzsolódó (At Risk)", 30, 640, 25000),
        ("Blog > Search_NonBrand > Email", "Potenciális hűségesek", 110, 280, 33000),
        ("Display_Ads > Blog > Retargeting", "Lemorzsolódó (At Risk)", 65, 490, 28000),
    ]
    df = pd.DataFrame(
        data, columns=["path", "rfm_segment", "conversions", "nulls", "avg_order_value"]
    )
    df["revenue"] = df["conversions"] * df["avg_order_value"]
    df["net_margin"] = df["revenue"] * 0.30
    return df


# -----------------------------------------------------------------------------
# SIDEBAR: ADATFORRÁS, SZŰRŐK, MOTOR BEÁLLÍTÁSOK
# -----------------------------------------------------------------------------
st.title("Markov Journey Mapper & Attribúciós Szimulátor")
st.markdown(
    "**Ritkamátrixos (Sparse), k-adik rendű és fedezetsúlyozott** elnyelő Markov-lánc, "
    "Removal Effect kalkulátor, konverziós sebesség-diagnosztika és integrációs hídak az MMM, "
    "RFM és MBA alkalmazásokhoz."
)

with st.sidebar:
    st.header("Adatforrás és Szűrők")
    data_source = st.radio(
        "Válassz adatforrást:",
        ["Beépített RFM + Journey demó adat", "Saját CSV feltöltése"],
    )

    effective_source = "demó"

    if data_source == "Saját CSV feltöltése":
        st.info(
            "Kötelező oszlopok: `path`, `conversions`, `nulls`. "
            "Opcionális: `rfm_segment`, `revenue`, `net_margin`."
        )
        uploaded_file = st.file_uploader("CSV fájl kiválasztása", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_df = load_csv_journeys(uploaded_file)
                effective_source = "feltöltött CSV"
                st.success(
                    f"CSV betöltve: {len(raw_df)} sor, "
                    f"{raw_df['path'].nunique()} egyedi útvonal."
                )
            except Exception as exc:
                st.error(f"CSV betöltési hiba: {exc}")
                st.info("A rendszer a beépített demó adatokra váltott vissza.")
                raw_df = generate_demo_data()
        else:
            st.warning("Nincs CSV feltöltve; a beépített demó adatok láthatók.")
            raw_df = generate_demo_data()
    else:
        raw_df = generate_demo_data()

    # RFM szegmens szűrő
    segments = ["Összes szegmens"] + sorted(raw_df["rfm_segment"].unique().tolist())
    selected_segment = st.selectbox("RFM Szegmens szűrő:", segments)

    if selected_segment != "Összes szegmens":
        filtered_df = raw_df[raw_df["rfm_segment"] == selected_segment].copy()
    else:
        filtered_df = raw_df.copy()

    st.divider()
    st.header("Markov Motor Beállítások")
    k_order = st.selectbox("Lánc-rend (k):", [1, 2, 3], index=1)
    collapse_loops = st.checkbox("Ötvöző hurkok eltávolítása (RLE)", value=True)
    st.caption(
        "Visszamenőleges ablak (lookback): nem állítható, mert az aggregált bemeneti adatok "
        "nem tartalmaznak időbélyeget, így az útvonalak dátuma szerinti szűrése nem lehetséges."
    )

    min_edge_pct = st.slider(
        "Minimum él-gyakoriság a Sankey-gráfon (%):",
        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
    )

    st.divider()
    st.info(
        f"Adatforrás: {effective_source}. A bevétel- és nettófedezet-értékek a betöltő "
        "25 000 Ft / 30 % alapértékekkel kiegészítik, ha a forrás adat nem tartalmazza őket."
    )

# -----------------------------------------------------------------------------
# MOTOR INICIALIZÁLÁSA
# -----------------------------------------------------------------------------
@st.cache_data
def compute_attribution_cached(filtered_df, k_order, collapse_loops):
    """Gyorsítótárazott attribúciós számítás: az engine illesztése és a Removal Effect."""
    engine = MarkovJourneyEngine(
        filtered_df,
        path_sep=">",
        k=k_order,
        collapse_loops=collapse_loops,
    )
    base_cvr = engine.solve_prob(weight_col="conversions")
    base_margin = engine.solve_prob(weight_col="net_margin")
    attr_df = engine.compute_attribution()
    transition_counts = engine.transition_counts
    return base_cvr, base_margin, attr_df, transition_counts, engine.all_states, engine

base_cvr, base_margin, attr_df, edge_counts, all_states, engine = compute_attribution_cached(
    filtered_df, k_order, collapse_loops
)

# Összesített metrikák KPI-khoz
total_conv = int(filtered_df["conversions"].sum())
total_null = int(filtered_df["nulls"].sum())
total_journeys = total_conv + total_null
total_rev = float(filtered_df["revenue"].sum())

# A teljes attr_df-hez hozzáadjuk a bevételt a Markov részesedés alapján (alap CVR arányban)
if not attr_df.empty:
    attr_df["Markov Konverziók"] = attr_df["Volume_RE"] / attr_df["Volume_RE"].sum() * total_conv if attr_df["Volume_RE"].sum() > 0 else 0
    attr_df["Markov Bevétel (Ft)"] = attr_df["Volume_RE"] / attr_df["Volume_RE"].sum() * total_rev if attr_df["Volume_RE"].sum() > 0 else 0

top_channel = attr_df.iloc[0]["Csatorna"] if not attr_df.empty else "-"
top_re = attr_df.iloc[0]["Volume_RE_pct"] if not attr_df.empty else 0.0

# -----------------------------------------------------------------------------
# FELSŐ KPI KÁRTYÁK
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Összes vizsgált útvonal", f"{total_journeys:,}".replace(",", " "))
col2.metric(
    "Markov Alap Konverziós Ráta",
    f"{base_cvr * 100:.2f}%",
)
col2.caption(f"{total_conv:,} konverzió".replace(",", " "))
col3.metric(
    "Összes bevétel", f"{int(total_rev):,} Ft".replace(",", " ")
)
col4.metric(
    "Legnagyobb eltávolítási hatású csatorna",
    top_channel,
    f"-{top_re:.1f}% konverziós hatás ha eltávolítva",
    delta_color="normal",
)

st.divider()

# -----------------------------------------------------------------------------
# FÜLEK
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Attribúció",
    "Útvonalak",
    "What-if",
])

# =============================================================================
# TAB 1: ATTRIBÚCIÓS IGAZSÁGMÉRLEG
# =============================================================================
with tab1:
    st.subheader("Markov Removal Effect — Volume és Nettó Fedezet")
    st.markdown(
        "A Removal Effect két dimenzióban jelenik meg: **Volume_RE** (konverziós darabszám kiesése) "
        "és **Margin_RE** (nettó fedezet kiesése). A különbség lekéri azokat a csatornákat, "
        "ami sok kisebb konverziós darabot hoz vs. ami nagy árrésű kosarakat."
    )

    if attr_df.empty:
        st.warning("Nincs elég adat az attribúcióhoz.")
    else:
        melt_df = attr_df.melt(
            id_vars="Csatorna",
            value_vars=["Volume_RE_pct", "Margin_RE_pct"],
            var_name="Mérés",
            value_name="Removal Effect (%)",
        )
        melt_df["Mérés"] = melt_df["Mérés"].map({
            "Volume_RE_pct": "Konverziós darabszám",
            "Margin_RE_pct": "Nettó fedezet",
        })

        fig_bar = px.bar(
            melt_df,
            x="Csatorna",
            y="Removal Effect (%)",
            color="Mérés",
            barmode="group",
            text_auto=".1f",
            color_discrete_sequence=["#2563EB", "#10B981"],
            height=460,
        )
        fig_bar.update_layout(
            legend_title_text="Mérési mód", xaxis_title="", yaxis_title="Removal Effect (%)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### Részletes Attribúciós Táblázat")
        show_cols = [
            "Csatorna",
            "Volume_RE_pct",
            "Margin_RE_pct",
            "Markov Konverziók",
            "Markov Bevétel (Ft)",
        ]
        st.dataframe(
            attr_df[show_cols].style.format({
                "Volume_RE_pct": "{:.2f}%",
                "Margin_RE_pct": "{:.2f}%",
                "Markov Konverziók": "{:.1f}",
                "Markov Bevétel (Ft)": "{:,.0f} Ft",
            }),
            use_container_width=True,
        )

# =============================================================================
# TAB 2: SANKEY ÉS ÁTMENETI MÁTRIX
# =============================================================================
with tab2:
    col_sankey, col_heat = st.columns([1.15, 1])

    with col_sankey:
        st.subheader("Vásárlói Forgalom (Sankey-gráf)")
        # Valós átmenet-darabszámok a becsolt láncból (nem P * journey_sorösszeg becslés).
        state_idx = {s: i for i, s in enumerate(all_states)}

        sources, targets, values, link_colors = [], [], [], []
        total_transitions = 0.0
        if edge_counts:
            total_transitions = sum(w for _, _, w in edge_counts)
            # A szűrés határa: az összes átmenet X százaléka (megtartja a jelentős folyókat).
            threshold = (min_edge_pct / 100.0) * total_transitions
            for frm, to, w in edge_counts:
                if w < threshold:
                    continue
                sources.append(state_idx[frm])
                targets.append(state_idx[to])
                values.append(w)
                if to == CONVERSION:
                    link_colors.append("rgba(16, 185, 129, 0.45)")
                elif to == NULL_STATE:
                    link_colors.append("rgba(239, 68, 68, 0.25)")
                else:
                    link_colors.append("rgba(59, 130, 246, 0.35)")
        else:
            threshold = 0.0

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=18, thickness=20, line=dict(color="black", width=0.5),
                      label=all_states),
            link=dict(source=sources, target=targets, value=values, color=link_colors)
        )])
        fig_sankey.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig_sankey, use_container_width=True)
        st.caption(
            f"Élesszűrő: {min_edge_pct}% az összes átmenetből ({int(total_transitions):,} darab). "
            "A szűrés elrejt néhány élt, ezért a látható gráf nem mutatja meg a teljes forgalmat."
        )

    with col_heat:
        st.subheader("Átmeneti Valószínűségi Mátrix ($P_{ij}$)")
        n_trans = engine.n_trans
        n_all = len(engine.all_states)
        P_dense = np.zeros((n_all, n_all))
        if engine.Q_conv is not None:
            Q_arr = engine.Q_conv.toarray()
            R_arr = engine.R_conv.toarray()
            P_dense[:n_trans, :n_trans] = Q_arr
            P_dense[:n_trans, n_trans:] = R_arr
        for abs_state in engine.absorbing_states:
            idx = state_idx[abs_state]
            P_dense[idx, idx] = 1.0
        p_df = pd.DataFrame(
            P_dense[:n_trans, :] * 100,
            index=engine.transient_states,
            columns=engine.all_states,
        )
        fig_heat = px.imshow(
            p_df,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Cél állapot (j)", y="Forrás állapot (i)", color="Átmenet (%)"),
        )
        fig_heat.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

# =============================================================================
# TAB 3: WHAT-IF & SEBESSÉG DIAGNOSTIKA
# =============================================================================
with tab3:
    st.subheader("What-If Csatorna-kiesés Szimulátor")
    st.markdown(
        "Válassz ki egy vagy több csatornát, és nézd meg, hogyan változik a konverziós ráta, "
        "ha az érintkezési pontokat eltávolítjuk (Removal Effect szimuláció)."
    )

    disabled_channels = st.multiselect(
        "Kapcsold ki a tesztelni kívánt csatornákat:",
        options=engine.channels,
        default=[top_channel] if top_channel != "-" else [],
    )

    sim_cvr = engine.solve_prob(weight_col="conversions", drop=disabled_channels)
    cvr_drop_pct = (1.0 - (sim_cvr / base_cvr)) * 100 if base_cvr > 0 else 0.0
    lost_conversions = total_conv * (cvr_drop_pct / 100.0)
    lost_revenue = total_rev * (cvr_drop_pct / 100.0)

    s_col1, s_col2, s_col3 = st.columns(3)
    delta_label = f"{-cvr_drop_pct:.1f}%" if cvr_drop_pct > 0 else "0.0%"
    s_col1.metric(
        "Szimulált Konverziós Ráta",
        f"{sim_cvr * 100:.2f}%",
        delta_label,
        delta_color="normal",
    )
    s_col1.caption(f"{int(round(lost_conversions)):,} konverzió veszteség".replace(",", " "))
    s_col2.metric(
        "Becsült Konverzió-kiesés",
        f"{int(round(lost_conversions)):,} db".replace(",", " "),
    )
    s_col3.metric(
        "Kockáztatott Bevétel (Removal Loss)",
        f"{int(round(lost_revenue)):,} Ft".replace(",", " "),
    )

    st.divider()

    st.subheader("Várható Lépésszám az Elnyelésig")
    st.markdown(
        r"A fundamentális mátrix ($\mathbf{N} \cdot \mathbf{1}$) alapján: hány további lépésre van "
        r"átlagosan szükség az adott állapotból a konverzióig vagy a kilépésig. Ez a lépésszám, "
        r"nem idő mértéke, és nem jelzi egyedi csatorna 'fricción'át."
    )
    steps_dict = engine.expected_steps_to_absorption()
    if steps_dict:
        steps_df = pd.DataFrame(
            list(steps_dict.items()), columns=["Állapot", "Várható lépésszám"]
        ).sort_values("Várható lépésszám", ascending=False)
        fig_steps = px.bar(
            steps_df.head(15),
            x="Várható lépésszám",
            y="Állapot",
            orientation="h",
            title="Top 15 legtöbben lépést igénylő állapot",
        )
        fig_steps.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(fig_steps, use_container_width=True)
        st.dataframe(steps_df, use_container_width=True, hide_index=True)
