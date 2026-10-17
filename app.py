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
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Markov Journey Mapper — Sparse Higher-Order Engine",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. SYNTHETIC DEMO DATA GENERATOR (RFM + MARGIN)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_demo_data() -> pd.DataFrame:
    data = [
        ("Social_Ads > Blog > Search_NonBrand > Email", "New Customers", 140, 420, 24000, 1.0),
        ("Social_Ads > Search_NonBrand > Search_Brand", "New Customers", 190, 510, 26500, 1.0),
        ("Social_Ads > Blog", "New Customers", 15, 890, 18000, 1.0),
        ("Social_Ads", "New Customers", 20, 1450, 15000, 1.0),
        ("Display_Ads > Social_Ads > Search_NonBrand > Retargeting", "New Customers", 95, 380, 29000, 1.0),
        ("Display_Ads", "New Customers", 5, 1100, 14000, 1.0),
        ("Search_NonBrand > Blog > Email > Search_Brand", "Potential Loyalists", 210, 310, 34000, 1.0),
        ("Search_NonBrand > Retargeting > Search_Brand", "Potential Loyalists", 175, 290, 31000, 1.0),
        ("Search_NonBrand", "Potential Loyalists", 45, 720, 22000, 1.0),
        ("Email > Search_Brand", "VIP (Champions)", 420, 180, 52000, 1.0),
        ("Search_Brand > Email", "VIP (Champions)", 310, 120, 48000, 1.0),
        ("Email", "VIP (Champions)", 260, 240, 45000, 1.0),
        ("Search_Brand", "VIP (Champions)", 290, 190, 49000, 1.0),
        ("Retargeting > Email > Search_Brand", "At Risk", 130, 410, 36000, 1.0),
        ("Social_Ads > Retargeting > Email", "At Risk", 85, 520, 27500, 1.0),
        ("Retargeting", "At Risk", 30, 640, 25000, 1.0),
        ("Blog > Search_NonBrand > Email", "Potential Loyalists", 110, 280, 33000, 1.0),
        ("Display_Ads > Blog > Retargeting", "At Risk", 65, 490, 28000, 1.0),
        ("Email > Search_Brand", "VIP (Champions)", 8, 2, 52000, -1.0),
        ("Search_Brand", "VIP (Champions)", 5, 1, 49000, -1.0),
        ("Blog > Search_NonBrand > Email", "Potential Loyalists", 10, 10, 33000, 0.0),
        ("Social_Ads > Blog", "New Customers", 8, 12, 18000, 0.0),
        ("Retargeting > Email > Search_Brand", "At Risk", 130, 270, 36000, 1.0),
        ("Social_Ads > Retargeting > Email", "At Risk", 85, 315, 27500, 1.0),
        ("Retargeting", "At Risk", 30, 370, 25000, 1.0),
        ("Email > Search_Brand", "At Risk", 60, 440, 31000, 1.0),
    ]
    df = pd.DataFrame(
        data, columns=["path", "rfm_segment", "conversions", "nulls", "avg_order_value", "margin_mult"]
    )
    df["revenue"] = df["conversions"] * df["avg_order_value"]
    df["net_margin"] = df["revenue"] * 0.30 * df["margin_mult"]
    return df


# -----------------------------------------------------------------------------
# SIDEBAR: DATA SOURCE, FILTERS, ENGINE SETTINGS
# -----------------------------------------------------------------------------
st.title("Markov Journey Mapper & Attribution Simulator")
st.markdown(
    "**Sparse (sparse), higher-order, value-weighted** absorbing Markov chain, "
    "Removal Effect calculator, conversion speed diagnostic, and integration bridges "
    "for MMM, RFM, and MBA."
)

with st.sidebar:
    st.header("Data Source & Filters")
    data_source = st.radio(
        "Choose data source:",
        ["Built-in RFM + Journey demo data", "Upload custom CSV"],
    )

    effective_source = "demo"

    if data_source == "Upload custom CSV":
        st.info(
            "Required columns: `path`, `conversions`, `nulls`. "
            "Optional: `rfm_segment`, `revenue`, `net_margin`."
        )
        uploaded_file = st.file_uploader("Select CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_df = load_csv_journeys(uploaded_file)
                effective_source = "uploaded CSV"
                st.success(
                    f"CSV loaded: {len(raw_df)} rows, "
                    f"{raw_df['path'].nunique()} unique paths."
                )
            except Exception as exc:
                st.error(f"CSV load error: {exc}")
                st.info("The system fell back to the built-in demo data.")
                raw_df = generate_demo_data()
        else:
            st.warning("No CSV uploaded; the built-in demo data is shown.")
            raw_df = generate_demo_data()
    else:
        raw_df = generate_demo_data()

    # RFM segment filter
    segments = ["All segments"] + sorted(raw_df["rfm_segment"].unique().tolist())
    selected_segment = st.selectbox("RFM Segment filter:", segments)

    if selected_segment != "All segments":
        filtered_df = raw_df[raw_df["rfm_segment"] == selected_segment].copy()
    else:
        filtered_df = raw_df.copy()

    st.divider()
    st.header("Markov Engine Settings")
    k_order = st.selectbox("Chain order (k):", [1, 2, 3], index=1)
    collapse_loops = st.checkbox("Remove looped chains (RLE)", value=True)
    st.caption(
        "Lookback window (lookback) is not adjustable because the aggregated input "
        "data does not contain timestamps, so date-based path filtering is not possible."
    )

    min_edge_pct = st.slider(
        "Minimum edge frequency in Sankey chart (%):",
        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
    )

    st.divider()
    st.info(
        f"Data source: {effective_source}. Revenue and net margin values are "
        "supplemented with the loader's defaults of 25,000 Ft / 30% if the source "
        "data does not contain them."
    )

# -----------------------------------------------------------------------------
# ENGINE INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_data
def compute_attribution_cached(filtered_df, k_order, collapse_loops):
    """Cached attribution computation: engine fitting and Removal Effect."""
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

# Engine validation failures (e.g. prohibited channel names) must surface as a
# user-visible message, not an uncaught application error.
engine_error = None
try:
    base_cvr, base_margin, attr_df, edge_counts, all_states, engine = compute_attribution_cached(
        filtered_df, k_order, collapse_loops
    )
except (ValueError, ZeroDivisionError) as exc:
    engine_error = exc
    base_cvr = 0.0
    base_margin = 0.0
    attr_df = pd.DataFrame(
        columns=[
            "Channel",
            "Volume_RE",
            "Margin_RE",
            "Volume_RE_pct",
            "Margin_RE_pct",
        ]
    )
    edge_counts = []
    all_states = []
    engine = None

if engine is None:
    st.error(f"Data processing error: {engine_error}")
    st.info(
        "Check the uploaded file's content (invalid channel name, "
        "negative or non-finite number), or select another data source."
    )
    st.stop()

# Aggregated metrics for KPIs
total_conv = int(filtered_df["conversions"].sum())
total_null = int(filtered_df["nulls"].sum())
total_journeys = total_conv + total_null
total_rev = float(filtered_df["revenue"].sum())

# Add revenue to the full attribution table based on Markov share (base CVR ratio)
if not attr_df.empty:
    attr_df["Markov Conversions"] = attr_df["Volume_RE"] / attr_df["Volume_RE"].sum() * total_conv if attr_df["Volume_RE"].sum() > 0 else 0
    attr_df["Markov Revenue (Ft)"] = attr_df["Volume_RE"] / attr_df["Volume_RE"].sum() * total_rev if attr_df["Volume_RE"].sum() > 0 else 0

top_channel = attr_df.iloc[0]["Channel"] if not attr_df.empty else "-"
top_re = attr_df.iloc[0]["Volume_RE_pct"] if not attr_df.empty else 0.0

# -----------------------------------------------------------------------------
# TOP KPI CARDS
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total paths analyzed", f"{total_journeys:,}".replace(",", " "))
col2.metric(
    "Markov Baseline Conversion Rate",
    f"{base_cvr * 100:.2f}%",
)
col2.caption(f"{total_conv:,} conversions".replace(",", " "))
col3.metric(
    "Total revenue", f"{int(total_rev):,} Ft".replace(",", " ")
)
col4.metric(
    "Channel with highest removal effect",
    top_channel,
    f"-{top_re:.1f}% conversion effect if removed",
    delta_color="normal",
)

st.divider()

# -----------------------------------------------------------------------------
# Configuration limits (input size and performance)
# -----------------------------------------------------------------------------
MAX_DISPLAY_STATES = 30  # Maximum number of transient states shown in heatmap.

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Attribution",
    "Journeys",
    "What-if",
])

# =============================================================================
# TAB 1: ATTRIBUTION DASHBOARD
# =============================================================================
with tab1:
    st.subheader("Markov Removal Effect — Volume & Net Margin")
    st.markdown(
        "Removal Effect is shown in two dimensions: **Volume_RE** (conversion count drop) "
        "and **Margin_RE** (net margin drop). The difference highlights channels that "
        "generate many small conversions vs. those generating high-margin baskets."
    )

    if attr_df.empty:
        st.warning("Not enough data for attribution.")
    else:
        melt_df = attr_df.melt(
            id_vars="Channel",
            value_vars=["Volume_RE_pct", "Margin_RE_pct"],
            var_name="Measure",
            value_name="Removal Effect (%)",
        )
        melt_df["Measure"] = melt_df["Measure"].map({
            "Volume_RE_pct": "Conversion count",
            "Margin_RE_pct": "Net margin",
        })

        fig_bar = px.bar(
            melt_df,
            x="Channel",
            y="Removal Effect (%)",
            color="Measure",
            barmode="group",
            text_auto=".1f",
            color_discrete_sequence=["#2563EB", "#64748B"],
            height=460,
        )
        fig_bar.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94A3B8")
        fig_bar.update_layout(
            legend_title_text="Measure", xaxis_title="", yaxis_title="Removal Effect (%)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### Detailed Attribution Table")
        show_cols = [
            "Channel",
            "Volume_RE_pct",
            "Margin_RE_pct",
            "Markov Conversions",
            "Markov Revenue (Ft)",
        ]
        def _fmt_pct(v):
            if pd.isna(v):
                return "N/A"
            return f"{v:.2f}%"

        st.dataframe(
            attr_df[show_cols].style.format({
                "Volume_RE_pct": _fmt_pct,
                "Margin_RE_pct": _fmt_pct,
                "Markov Conversions": "{:.1f}",
                "Markov Revenue (Ft)": "{:,.0f} Ft",
            }),
            use_container_width=True,
        )
        st.caption(
            "A negative Margin_RE means removing the channel would improve expected "
            "net margin (loss-making channel). The 'N/A' value means the baseline "
            "net margin is not positive, so percentage margin removal effect has no meaning."
        )

# =============================================================================
# TAB 2: SANKEY & TRANSITION MATRIX
# =============================================================================
with tab2:
    col_sankey, col_heat = st.columns([1.15, 1])

    with col_sankey:
        st.subheader("Customer Flow (Sankey chart)")
        # Real transition counts from the estimated chain (not P * journey count estimate).
        state_idx = {s: i for i, s in enumerate(all_states)}

        sources, targets, values, link_colors = [], [], [], []
        total_transitions = 0.0
        if edge_counts:
            total_transitions = sum(w for _, _, w in edge_counts)
            # Threshold: X% of all transitions (keeps significant flows).
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
            f"Edge filter: {min_edge_pct}% of all transitions ({int(total_transitions):,} items). "
            "The filter hides some edges, so the visible chart does not show total flow."
        )

    with col_heat:
        st.subheader("Transition Probability Matrix ($P_{ij}$)")
        n_trans = engine.n_trans
        state_pos = {s: i for i, s in enumerate(engine.transient_states)}
        # Full n_all x n_all matrix densification is memory-intensive for large state
        # spaces (10,000 states -> ~800 MB), so we only show a bounded subset. Ranking
        # is based on estimated transition counts (ties broken by state name), not
        # normalized probabilities, since those all sum to 1.0.
        outgoing_counts = {s: 0.0 for s in engine.transient_states}
        for frm, to, w in engine.transition_counts:
            if frm in outgoing_counts:
                outgoing_counts[frm] += w

        if n_trans > MAX_DISPLAY_STATES:
            ranked = sorted(outgoing_counts.items(), key=lambda kv: kv[1], reverse=True)
            sel_states = [s for s, _ in ranked[:MAX_DISPLAY_STATES]]
            shown_total = sum(w for _, w in ranked[:MAX_DISPLAY_STATES])
            all_total = sum(outgoing_counts.values())
            st.info(
                f"The full state space ({n_trans} transient states) is too large for "
                f"the heatmap; we show the {MAX_DISPLAY_STATES} highest-traffic states "
                f"({shown_total:.0f} / {all_total:.0f} transitions). Rows will not sum "
                "to 100% because transitions to hidden states are omitted."
            )
        else:
            sel_states = engine.transient_states

        display_states = sel_states + [CONVERSION, NULL_STATE]
        n_disp = len(display_states)
        P_disp = np.zeros((n_disp, n_disp))
        conv_col = n_disp - 2
        null_col = n_disp - 1
        for i, frm in enumerate(sel_states):
            ii = state_pos[frm]
            for j, to in enumerate(sel_states):
                jj = state_pos[to]
                P_disp[i, j] = float(engine.Q_conv[ii, jj])
            P_disp[i, conv_col] = float(engine.R_conv[ii, 0])
            P_disp[i, null_col] = float(engine.R_conv[ii, 1])
        # Absorbing states point to themselves
        P_disp[conv_col, conv_col] = 1.0
        P_disp[null_col, null_col] = 1.0

        p_df = pd.DataFrame(
            P_disp * 100,
            index=sel_states + [CONVERSION, NULL_STATE],
            columns=display_states,
        )
        fig_heat = px.imshow(
            p_df,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Target state (j)", y="Source state (i)", color="Transition (%)"),
        )
        fig_heat.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

# =============================================================================
# TAB 3: WHAT-IF & SPEED DIAGNOSTICS
# =============================================================================
with tab3:
    st.subheader("What-If Channel Removal Simulator")
    st.markdown(
        "Select one or more channels and see how the conversion rate changes "
        "when their touchpoints are removed (Removal Effect simulation)."
    )

    disabled_channels = st.multiselect(
        "Select channels to disable for testing:",
        options=engine.channels,
        default=[top_channel] if top_channel != "-" else [],
    )

    try:
        sim_cvr = engine.solve_prob(weight_col="conversions", drop=disabled_channels)
    except ValueError as exc:
        st.error(f"What-if simulation error: {exc}")
        st.info("Select a different channel combination.")
        sim_cvr = None

    s_col1, s_col2, s_col3 = st.columns(3)
    if sim_cvr is None:
        s_col1.metric("Simulated Conversion Rate", "Not available")
        s_col1.caption("Simulation cannot be produced for the current combination.")
        s_col2.metric("Estimated conversion drop", "-")
        s_col3.metric("At-risk Revenue (Removal Loss)", "-")
    else:
        cvr_drop_pct = (1.0 - (sim_cvr / base_cvr)) * 100 if base_cvr > 0 else 0.0
        lost_conversions = total_conv * (cvr_drop_pct / 100.0)
        lost_revenue = total_rev * (cvr_drop_pct / 100.0)

        delta_label = f"{-cvr_drop_pct:.1f}%" if cvr_drop_pct > 0 else "0.0%"
        s_col1.metric(
            "Simulated Conversion Rate",
            f"{sim_cvr * 100:.2f}%",
            delta_label,
            delta_color="normal",
        )
        s_col1.caption(
            f"{int(round(lost_conversions)):,} conversions lost".replace(",", " ")
        )
        s_col2.metric(
            "Estimated conversion drop",
            f"{int(round(lost_conversions)):,} units".replace(",", " "),
        )
        s_col3.metric(
            "At-risk Revenue (Removal Loss)",
            f"{int(round(lost_revenue)):,} Ft".replace(",", " "),
        )

    st.divider()

    st.subheader("Expected Steps to Absorption")
    st.markdown(
        r"Based on the fundamental matrix ($\mathbf{N} \cdot \mathbf{1}$): how many more "
        r"steps on average are needed from a given state to conversion or exit. This is "
        r"a step count, not a time measure, and does not indicate per-channel 'friction'."
    )
    steps_dict = engine.expected_steps_to_absorption()
    if steps_dict:
        steps_df = pd.DataFrame(
            list(steps_dict.items()), columns=["State", "Expected steps"]
        ).sort_values("Expected steps", ascending=False)
        fig_steps = px.bar(
            steps_df.head(15),
            x="Expected steps",
            y="State",
            orientation="h",
            title="Top 15 states requiring most steps",
        )
        fig_steps.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(fig_steps, use_container_width=True)
        st.dataframe(steps_df, use_container_width=True, hide_index=True)
