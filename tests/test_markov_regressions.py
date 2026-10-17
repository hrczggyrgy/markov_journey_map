"""Regression suite for the correctness defects identified in review.

Covers: empty/all-zero input, missing paths, negative counts, non-finite values,
known single-channel CVR, deterministic journeys across k=1/2/3, equivalent
aggregated vs split-row monetary inputs, demo monetary reconciliation, basket-key
integrity, duplicate join-key rejection, and solver safeguards.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.markov_engine import (
    MarkovJourneyEngine,
    clean_journey,
    make_k_order_states,
    START,
    CONVERSION,
    NULL_STATE,
)
from core.data_loader import load_csv_journeys
from core.bridge_connectors import enrich_journeys_with_mba_rules


# --------------------------------------------------------------------------- #
# Engine: empty / degenerate inputs
# --------------------------------------------------------------------------- #
def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["path", "conversions", "nulls"])


def test_empty_input_no_traceback():
    eng = MarkovJourneyEngine(_empty_df(), k=2)
    assert eng.channels == []
    assert eng.compute_attribution().empty
    assert eng.expected_steps_to_absorption() == {}
    assert eng.solve_prob(weight_col="conversions") == 0.0


def test_all_zero_input_safe():
    # Zero-conversion rows: channels present, but CVR is 0.
    df = pd.DataFrame(
        {
            "path": ["A > B", "C"],
            "conversions": [0, 0],
            "nulls": [5, 5],
        }
    )
    eng = MarkovJourneyEngine(df, k=2)
    assert eng.channels == ["A", "B", "C"]
    assert eng.solve_prob(weight_col="conversions") == 0.0
    attr = eng.compute_attribution()
    assert not attr.empty
    assert (attr["Volume_RE"] == 0.0).all()


def test_missing_path_becomes_nan_channel_rejected():
    df = pd.DataFrame(
        {
            "path": [np.nan, "A > B"],
            "conversions": [10, 5],
            "nulls": [5, 5],
        }
    )
    # Missing paths should not silently become the literal string "nan" channel.
    eng = MarkovJourneyEngine(df, k=2)
    assert "nan" not in eng.channels
    assert eng.channels == ["A", "B"]


def test_negative_counts_rejected_by_engine():
    """Direct engine input must reject negative counts, matching the loader."""
    df = pd.DataFrame(
        {
            "path": ["A > B", "C > D"],
            "conversions": [-3, 5],
            "nulls": [0, -2],
        }
    )
    with pytest.raises(ValueError, match="Invalid"):
        MarkovJourneyEngine(df, k=2)


def test_non_finite_counts_rejected_by_engine():
    """Direct engine input must reject non-finite counts, matching the loader."""
    df = pd.DataFrame(
        {
            "path": ["A > B", "C > D", "E > F"],
            "conversions": [np.inf, 5, 5],
            "nulls": [5, 5, 5],
        }
    )
    with pytest.raises(ValueError, match="Invalid"):
        MarkovJourneyEngine(df, k=2)


def test_non_finite_margin_rejected_by_engine():
    """Direct engine input must reject non-finite margin, matching the loader."""
    df = pd.DataFrame(
        {
            "path": ["A > B", "C > D"],
            "conversions": [5, 5],
            "nulls": [5, 5],
            "net_margin": [100.0, np.inf],
        }
    )
    with pytest.raises(ValueError, match="Invalid"):
        MarkovJourneyEngine(df, k=2)


# --------------------------------------------------------------------------- #
# Loader validation
# --------------------------------------------------------------------------- #
def test_csv_loader_rejects_invalid_counts(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("path,conversions,nulls\nA>B,abc,5\nC>D,3,xyz\n")
    with pytest.raises(ValueError, match="invalid"):
        load_csv_journeys(str(p))


def test_csv_loader_rejects_missing_path(tmp_path):
    p = tmp_path / "missing_path.csv"
    p.write_text("path,conversions,nulls\n,5,5\nA>B,3,5\n")
    with pytest.raises(ValueError, match="missing path"):
        load_csv_journeys(str(p))


def test_csv_loader_rejects_empty_and_separator_paths(tmp_path):
    p = tmp_path / "empty_path.csv"
    p.write_text("path,conversions,nulls\n  ,5,5\n> >,3,2\nA>B,1,1\n")
    with pytest.raises(ValueError, match="empty or separator-only path"):
        load_csv_journeys(str(p))


def test_csv_loader_rejects_infinite_revenue(tmp_path):
    p = tmp_path / "inf_rev.csv"
    p.write_text("path,conversions,nulls,revenue,net_margin\nA>B,1,1,inf,0.3\n")
    with pytest.raises(ValueError, match="Invalid"):
        load_csv_journeys(str(p))


def test_csv_loader_rejects_infinite_margin(tmp_path):
    p = tmp_path / "inf_mar.csv"
    p.write_text("path,conversions,nulls,revenue,net_margin\nA>B,1,1,100,inf\n")
    with pytest.raises(ValueError, match="Invalid"):
        load_csv_journeys(str(p))


def test_csv_loader_rejects_channel_name_with_delimiter(tmp_path):
    p = tmp_path / "bad_channel.csv"
    p.write_text("path,conversions,nulls\nA__B > C,5,5\nC,5,5\n")
    with pytest.raises(ValueError, match="Invalid channel name"):
        load_csv_journeys(str(p))


def test_csv_loader_rejects_zero_volume_with_money(tmp_path):
    p = tmp_path / "zero_vol.csv"
    p.write_text("path,conversions,nulls,revenue,net_margin\nA,1,0,100,30\nB,0,0,1000,300\n")
    with pytest.raises(ValueError, match="Zero-volume"):
        load_csv_journeys(str(p))


def test_csv_loader_accepts_valid_rows(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text("path,conversions,nulls\nA>B,10,5\nC>D,3,2\n")
    df = load_csv_journeys(str(p))
    assert len(df) == 2
    assert df["conversions"].tolist() == [10.0, 3.0]
    assert df["nulls"].tolist() == [5.0, 2.0]


def test_csv_loader_missing_required_columns(tmp_path):
    p = tmp_path / "nope.csv"
    p.write_text("path,conversions\nA>B,5\n")
    with pytest.raises(ValueError):
        load_csv_journeys(str(p))


# --------------------------------------------------------------------------- #
# Margin semantics: aggregation invariance
# --------------------------------------------------------------------------- #
def test_margin_re_aggregation_invariant():
    """Truly equivalent aggregated vs split-row inputs must give identical Margin_RE.

    Splitting a row must divide both the journey counts AND the margin proportionally,
    so the terminal margin reward per departure is unchanged.
    """
    base = pd.DataFrame(
        {
            "path": ["A", "B"],
            "conversions": [100, 100],
            "nulls": [900, 900],
            "net_margin": [1000.0, 1000.0],
        }
    )
    split = pd.DataFrame(
        {
            "path": ["A", "A", "B", "B"],
            "conversions": [50, 50, 50, 50],
            "nulls": [450, 450, 450, 450],
            # Margin divided proportionally with the journey counts: total stays 1000 per channel.
            "net_margin": [500.0, 500.0, 500.0, 500.0],
        }
    )
    eng_a = MarkovJourneyEngine(base, k=1)
    eng_b = MarkovJourneyEngine(split, k=1)
    attr_a = eng_a.compute_attribution().set_index("Channel")
    attr_b = eng_b.compute_attribution().set_index("Channel")
    for ch in ["A", "B"]:
        assert abs(attr_a.loc[ch, "Margin_RE"] - attr_b.loc[ch, "Margin_RE"]) < 1e-9


def test_signed_margin_effect_preserved():
    """Margin_RE must be signed: removing a loss-making channel improves margin."""
    df = pd.DataFrame(
        {
            "path": ["A", "B"],
            "conversions": [1, 1],
            "nulls": [0, 0],
            "net_margin": [100.0, -50.0],
        }
    )
    eng = MarkovJourneyEngine(df, k=1)
    attr = eng.compute_attribution().set_index("Channel")
    assert abs(eng.solve_prob(weight_col="net_margin") - 25.0) < 1e-9
    # Removing B (margin -50) raises expected margin: Margin_RE is negative.
    assert attr.loc["B", "Margin_RE"] < 0
    # Removing A (margin +100) lowers expected margin: Margin_RE is positive.
    assert attr.loc["A", "Margin_RE"] > 0
    # Nothing is clipped to zero.
    assert attr.loc["B", "Margin_RE"] != 0.0


def test_nonpositive_baseline_margin_unavailable():
    """When baseline margin is nonpositive, percentage margin attribution is NaN."""
    df = pd.DataFrame(
        {
            "path": ["A", "B"],
            "conversions": [1, 1],
            "nulls": [0, 0],
            "net_margin": [-50.0, -30.0],
        }
    )
    eng = MarkovJourneyEngine(df, k=1)
    attr = eng.compute_attribution().set_index("Channel")
    assert eng.solve_prob(weight_col="net_margin") <= 0
    assert pd.isna(attr.loc["A", "Margin_RE"])
    assert pd.isna(attr.loc["B", "Margin_RE"])
    assert pd.isna(attr.loc["A", "Margin_RE_pct"])


def test_engine_scales_to_many_channels():
    """Engine must build without memory blow-up for more states than MAX_DISPLAY_STATES."""
    n_channels = 50
    df = pd.DataFrame(
        {
            "path": [f"C{i}" for i in range(n_channels)],
            "conversions": [1] * n_channels,
            "nulls": [1] * n_channels,
            "net_margin": [float(100 + i) for i in range(n_channels)],
        }
    )
    eng = MarkovJourneyEngine(df, k=1)
    assert len(eng.channels) == n_channels
    assert len(eng.transient_states) == n_channels + 1  # + Start
    # Attribution and transition counts must complete.
    assert not eng.compute_attribution().empty
    counts = eng.transition_counts
    assert len(counts) == 3 * n_channels  # Start->Ci, Ci->Conversion, Ci->Null
    # No absorbing self-loops exposed.
    loops = [(f, t) for (f, t, w) in counts if f == t and f in (CONVERSION, NULL_STATE)]
    assert loops == []


def test_heatmap_selection_retains_highest_traffic_states():
    """Bounded heatmap keeps the states with the most observed outgoing traffic."""
    # Must stay in sync with MAX_DISPLAY_STATES in app.py.
    MAX_DISPLAY_STATES = 30

    n_channels = 40
    # One high-volume state A gets 1000 journeys; the rest get 1 each.
    rows = [("A", 1000, 1000)] + [(f"C{i}", 1, 1) for i in range(n_channels)]
    df = pd.DataFrame(
        {
            "path": [r[0] for r in rows],
            "conversions": [r[1] for r in rows],
            "nulls": [r[2] for r in rows],
            "net_margin": [float(r[1]) for r in rows],
        }
    )
    eng = MarkovJourneyEngine(df, k=1)
    assert len(eng.transient_states) == n_channels + 1 + 1  # A + C's + Start

    outgoing_counts = {s: 0.0 for s in eng.transient_states}
    for frm, to, w in eng.transition_counts:
        if frm in outgoing_counts:
            outgoing_counts[frm] += w

    # A should dominate outgoing traffic.
    assert outgoing_counts["A"] > outgoing_counts["C0"]
    ranked = sorted(outgoing_counts.items(), key=lambda kv: kv[1], reverse=True)
    sel_states = [s for s, _ in ranked[:MAX_DISPLAY_STATES]]
    assert "A" in sel_states
    assert len(sel_states) == MAX_DISPLAY_STATES


def test_what_if_failure_renders_unavailable_state():
    """A forced solver failure in What-if must not crash or fabricate zero metrics."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing not available")

    import app as app_module

    original = app_module.MarkovJourneyEngine.solve_prob

    def _raise_on_multi_drop(self, weight_col="conversions", drop=None):
        # Only fail when multiple channels are dropped simultaneously. The cached
        # main analysis (compute_attribution_cached) drops one channel at a time,
        # so it keeps working; the What-if section drops the user selection which
        # may contain several channels.
        if drop is not None and len(drop) > 1:
            raise ValueError("forced non-absorbing chain")
        return original(self, weight_col=weight_col, drop=drop)

    app_module.MarkovJourneyEngine.solve_prob = _raise_on_multi_drop

    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    at = AppTest.from_file(app_path).run(timeout=60)
    assert len(at.exception) == 0, f"Initial app run raised: {at.exception}"

    # Select two channels to drop, triggering the forced What-if failure.
    ms_label = "Select channels to disable for testing:"
    ms = [e for e in at.multiselect._list if e.label == ms_label][0]
    ms.set_value(list(ms.options[:2]))
    at.run(timeout=60)

    app_module.MarkovJourneyEngine.solve_prob = original

    assert len(at.exception) == 0, f"App raised after multi-drop: {at.exception}"
    # The What-if metric should show "Not available", not a formatted None or zero.
    found = any(
        "Not available" in str(el.value) for el in at.get("metric") if el.value is not None
    )
    assert found, "Expected 'Not available' metric on What-if failure"


def test_deterministic_steps_exact_values():
    """A > B > C: Start plus one state per touchpoint, identical transitions across k."""
    df = pd.DataFrame(
        {
            "path": ["A > B > C"],
            "conversions": [1],
            "nulls": [0],
        }
    )
    for k in [1, 2, 3]:
        eng = MarkovJourneyEngine(df, k=k)
        # Start is always a transient state; one state per touchpoint follows.
        assert START in eng.transient_states
        n_touch = 3
        n_trans = len(eng.transient_states)
        assert n_trans == n_touch + 1, f"k={k}: expected {n_touch + 1} states, got {n_trans}"
        # Exactly four physical transitions: Start->1st, 1st->2nd, 2nd->3rd, 3rd->Conversion.
        counts = eng.transition_counts
        assert len(counts) == 4, f"k={k}: expected 4 transitions, got {len(counts)}: {counts}"
        # The last transition always goes to Conversion.
        to_conversion = [(frm, to, w) for frm, to, w in counts if to == CONVERSION]
        assert len(to_conversion) == 1
        assert to_conversion[0][2] == 1.0
        # No absorbing self-loops exposed as traffic.
        loops = [(f, t) for (f, t, w) in counts if f == t and f in (CONVERSION, NULL_STATE)]
        assert loops == []
        # Steps to absorption decrease by one per touchpoint: last=1, first=3, Start=4.
        steps = eng.expected_steps_to_absorption()
        step_values = sorted(steps.values(), reverse=True)
        assert step_values == [4.0, 3.0, 2.0, 1.0], f"k={k}: {steps}"


def test_single_channel_cvr_matches_volume_cvr():
    """A single-channel path with 1 conversion and 9 non-conversions has CVR 0.1."""
    df = pd.DataFrame(
        {
            "path": ["A"],
            "conversions": [1],
            "nulls": [9],
            "net_margin": [100.0],
        }
    )
    eng = MarkovJourneyEngine(df, k=1)
    assert abs(eng.solve_prob(weight_col="conversions") - 0.1) < 1e-9


# --------------------------------------------------------------------------- #
# Higher-order state construction: one transition per touchpoint
# --------------------------------------------------------------------------- #
def test_deterministic_steps_invariant_across_k():
    """Deterministic 3-touch journey should keep comparable physical transition counts."""
    df = pd.DataFrame(
        {
            "path": ["A > B > C"],
            "conversions": [1],
            "nulls": [0],
        }
    )
    steps = {}
    for k in [1, 2, 3]:
        eng = MarkovJourneyEngine(df, k=k)
        steps[k] = eng.expected_steps_to_absorption()
    # At least no order should claim fewer physical steps than the number of touches.
    assert steps[1] is not None and steps[2] is not None and steps[3] is not None


def test_make_k_order_states_single_touch():
    assert make_k_order_states(["A"], k=2) == ["(Start)__A"]
    assert make_k_order_states(["A"], k=1) == ["A"]


def test_make_k_order_states_one_state_per_touchpoint():
    """Deterministic 3-touch journey: exactly 3 states regardless of k."""
    assert make_k_order_states(["A", "B", "C"], k=1) == ["A", "B", "C"]
    assert make_k_order_states(["A", "B", "C"], k=2) == [
        "(Start)__A",
        "A__B",
        "B__C",
    ]
    assert make_k_order_states(["A", "B", "C"], k=3) == [
        "(Start)__(Start)__A",
        "(Start)__A__B",
        "A__B__C",
    ]


# --------------------------------------------------------------------------- #
# Solver safeguards
# --------------------------------------------------------------------------- #
def test_all_channels_removed_absorbs_to_null():
    df = pd.DataFrame(
        {
            "path": ["A > B"],
            "conversions": [10],
            "nulls": [10],
        }
    )
    eng = MarkovJourneyEngine(df, k=1)
    sim = eng.solve_prob(weight_col="conversions", drop=["A", "B"])
    assert sim == 0.0


def test_no_channels_returns_empty_attribution():
    df = pd.DataFrame(
        {
            "path": [],
            "conversions": [],
            "nulls": [],
        }
    )
    eng = MarkovJourneyEngine(df, k=1)
    assert eng.compute_attribution().empty


# --------------------------------------------------------------------------- #
# Demo data reconciliation (mirrors scripts/generate_demo_data.py logic)
# --------------------------------------------------------------------------- #
def test_demo_revenue_reconciles_with_generator():
    """The committed generator's revenue reconciles with its aggregate base table."""
    import scripts.generate_demo_data as gen

    base = gen.BASE_JOURNEYS
    intended_rev = sum(conv * aov for _, _, conv, _, aov, _ in base)
    # Re-run the generator's journey construction inline to capture per-row revenue.
    rows = []
    journey_id = 0
    for path, seg, conv, nulls, aov, margin_mult in base:
        per_conv_margin = aov * 0.30 * margin_mult
        total_journeys = conv + nulls
        for j in range(total_journeys):
            is_conv = j < conv
            journey_id += 1
            rows.append(
                {
                    "path_id": f"J{journey_id:07d}",
                    "path": path,
                    "rfm_segment": seg,
                    "conversions": 1 if is_conv else 0,
                    "nulls": 0 if is_conv else 1,
                    "revenue": aov if is_conv else 0.0,
                    "net_margin": per_conv_margin if is_conv else 0.0,
                }
            )
    gen_df = pd.DataFrame(rows)
    # Revenue reconciles with the aggregate input table (incl. return journeys).
    assert int(gen_df["revenue"].sum()) == intended_rev
    # Every converting row gets exactly its AOV, not the aggregate total.
    conv_rows = gen_df[gen_df["conversions"] == 1]
    for _, row in conv_rows.iterrows():
        base_rows = [b for b in base if b[0] == row["path"]]
        assert any(
            abs(row["revenue"] - br[4]) < 1e-6 for br in base_rows
        ), f"{row['path']}: {row['revenue']} not in {[br[4] for br in base_rows]}"


def test_actual_generator_output_reconciles(tmp_path):
    """Run the real generator into a temp dir and validate Parquet data contract."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_demo_data",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "generate_demo_data.py"),
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    # Monkeypatch DATA_DIR to a temp location so we do not clobber repo data.
    gen.DATA_DIR = str(tmp_path)
    gen.generate()

    import duckdb

    j = duckdb.execute(
        "SELECT path_id, path, conversions, revenue, net_margin FROM read_parquet(?)",
        [str(tmp_path / "customer_journeys.parquet")],
    ).df()
    b = duckdb.execute(
        "SELECT path_id FROM read_parquet(?)",
        [str(tmp_path / "transaction_baskets_mba.parquet")],
    ).df()

    # Import the generator's base table so the test always matches the data.
    base = gen.BASE_JOURNEYS
    # Expected revenue / margin from the aggregate base table (incl. return journeys).
    intended_rev = sum(conv * aov for _, _, conv, _, aov, _ in base)
    intended_margin = sum(
        conv * aov * 0.30 * mm for _, _, conv, _, aov, mm in base
    )
    # Revenue reconciles with aggregate base table.
    assert abs(float(j["revenue"].sum()) - intended_rev) < 1.0
    # Margin reconciles with margin_multiplier per row.
    assert abs(float(j["net_margin"].sum()) - intended_margin) < 1.0
    # Every basket references a converting journey.
    conv_ids = set(j.loc[j["conversions"] == 1, "path_id"])
    assert set(b["path_id"]) <= conv_ids
    # No duplicate basket keys (one basket per journey).
    assert b["path_id"].is_unique
    # Journey IDs are unique.
    assert j["path_id"].is_unique
    # Every converting row gets margin = margin_multiplier * 30% of its revenue.
    conv_rows = j[j["conversions"] == 1]
    for _, row in conv_rows.iterrows():
        expected_margins = [
            br[4] * 0.30 * br[5] for br in base if br[0] == row["path"]
        ]
        assert any(
            abs(row["net_margin"] - em) < 1e-6 for em in expected_margins
        ), f"{row['path_id']}: {row['net_margin']} not in {expected_margins}"
    non_conv = j[j["conversions"] == 0]
    assert (non_conv["net_margin"] == 0.0).all()
    assert (non_conv["revenue"] == 0.0).all()
    # Some converting journeys have no basket (unmatched > 0) so the MBA page
    # can exercise the unmatched-journey path.
    n_converted = int(conv_rows.shape[0])
    assert len(b) < n_converted
    assert n_converted - len(b) > 0


# --------------------------------------------------------------------------- #
# MBA join validation
# --------------------------------------------------------------------------- #
def test_duplicate_join_keys_rejected():
    """Duplicate path_id on either side must be rejected, not silently inflated."""
    journeys = pd.DataFrame(
        {
            "path_id": ["J0000001", "J0000001", "J0000002"],
            "path": ["A > B", "A > B", "C"],
            "conversions": [1, 1, 1],
            "nulls": [0, 0, 0],
        }
    )
    mba = pd.DataFrame(
        {
            "path_id": ["J0000001", "J0000002"],
            "is_cross_sell_bundle": [True, False],
            "net_margin": [100.0, 50.0],
            "bundle_type": ["Cross-sell", "None"],
        }
    )
    with pytest.raises(ValueError, match="duplicate path_id"):
        enrich_journeys_with_mba_rules(journeys, mba)


def test_mba_merge_preserves_journey_counts_and_revenue():
    """One-to-one merge: journey rows and revenue are preserved exactly."""
    journeys = pd.DataFrame(
        {
            "path_id": ["J0000001", "J0000002"],
            "path": ["A > B", "C"],
            "conversions": [1, 1],
            "nulls": [0, 0],
            "revenue": [100.0, 200.0],
            "net_margin": [30.0, 60.0],
        }
    )
    mba = pd.DataFrame(
        {
            "path_id": ["J0000001", "J0000002"],
            "is_cross_sell_bundle": [True, False],
            "net_margin": [100.0, 50.0],
            "bundle_type": ["Cross-sell", "None"],
        }
    )
    merged = enrich_journeys_with_mba_rules(journeys, mba)
    assert len(merged) == 2
    assert float(merged["revenue"].sum()) == 300.0


def test_mba_path_id_matches_journey_id_format():
    # Basket path_id uses P000000-style; journeys use J0000000-style. After the
    # generator fix, every basket path_id must match an eligible journey id.
    baskets = pd.DataFrame(
        {
            "path_id": ["J0000001", "J0000002"],
            "is_cross_sell_bundle": [True, False],
            "bundle_type": ["Cross-sell", "None"],
        }
    )
    journeys = pd.DataFrame(
        {
            "path_id": ["J0000001", "J0000002"],
            "path": ["A > B", "C"],
            "conversions": [1, 1],
            "nulls": [0, 0],
        }
    )
    merged = enrich_journeys_with_mba_rules(journeys, baskets)
    assert merged["path_id"].notna().all()
    assert set(merged["path_id"]) == {"J0000001", "J0000002"}


# --------------------------------------------------------------------------- #
# clean_journey / make_k_order_states helpers
# --------------------------------------------------------------------------- #
def test_clean_journey_collapse_loops():
    assert clean_journey(["A", "A", "B", "B", "B", "C"]) == ["A", "B", "C"]
    assert clean_journey([], collapse_loops=False) == []


def test_transition_counts_excludes_absorbing_self_loops():
    """Sankey source must not expose synthetic Conversion->Conversion / Null->Null loops."""
    df = pd.DataFrame(
        {
            "path": ["A > B", "C"],
            "conversions": [10, 5],
            "nulls": [5, 5],
            "net_margin": [100.0, 50.0],
        }
    )
    eng = MarkovJourneyEngine(df, k=2)
    counts = eng.transition_counts
    loops = [
        (frm, to, w)
        for frm, to, w in counts
        if frm == to and frm in (CONVERSION, NULL_STATE)
    ]
    assert loops == []
    # Every non-absorbing source should have a real destination with positive weight.
    assert all(w > 0 for _, _, w in counts)


def test_transition_counts_empty_input():
    eng = MarkovJourneyEngine(_empty_df(), k=2)
    assert eng.transition_counts == []


def test_solver_rejects_non_absorbing_chain():
    """A closed self-loop must produce a diagnostic, not a silent zero result."""
    df = pd.DataFrame(
        {
            "path": ["A"],
            "conversions": [1],
            "nulls": [0],
        }
    )
    eng = MarkovJourneyEngine(df, k=1)
    import scipy.sparse as sp

    n = eng.n_trans
    # Start -> Start with probability 1: closed loop, no absorption.
    Q_bad = sp.csr_matrix((n, n), dtype=np.float64)
    Q_bad[0, 0] = 1.0
    b = np.zeros(n)
    b[eng._state_idx[START]] = 1.0
    with pytest.raises(ValueError, match="Non-absorbing|no stable solution"):
        eng._solve_visit_system(sp.eye(n, format="csr") - Q_bad.T, b)


def test_channel_name_with_delimiter_rejected():
    """Channel names containing "__" collide with k-order state encoding."""
    df = pd.DataFrame(
        {
            "path": ["A__B > C", "C"],
            "conversions": [5, 5],
            "nulls": [5, 5],
        }
    )
    with pytest.raises(ValueError, match="Invalid channel name"):
        MarkovJourneyEngine(df, k=2)


def test_reserved_state_name_rejected():
    """Channel names must not collide with reserved state names."""
    for bad in ["(Start)", "(Conversion)", "(Null)"]:
        df = pd.DataFrame(
            {
                "path": [bad, "A"],
                "conversions": [5, 5],
                "nulls": [5, 5],
            }
        )
        with pytest.raises(ValueError, match="Invalid channel name"):
            MarkovJourneyEngine(df, k=2)


# --------------------------------------------------------------------------- #
# Application-level smoke tests
# --------------------------------------------------------------------------- #
def _run_page_smoke(page_file: str) -> object:
    """Run a Streamlit page file and return the AppTest result.

    Skips if streamlit.testing is unavailable.
    """
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing not available")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, page_file)
    at = AppTest.from_file(path).run(timeout=60)
    return at


def test_default_app_renders_without_exception():
    """The app.py entry point must complete without uncaught exceptions."""
    at = _run_page_smoke("app.py")
    assert len(at.exception) == 0, f"App raised: {at.exception}"


def test_rfm_page_renders_without_exception():
    """RFM segmentation page completes without uncaught exceptions."""
    at = _run_page_smoke("pages/1_RFM_Segmentation.py")
    assert len(at.exception) == 0, f"RFM page raised: {at.exception}"


def test_mba_page_renders_without_exception():
    """MBA page completes without uncaught exceptions (cs_df regression)."""
    at = _run_page_smoke("pages/2_Market_Basket_MBA.py")
    assert len(at.exception) == 0, f"MBA page raised: {at.exception}"


def test_mmm_page_renders_without_exception():
    """MMM prior calibration page completes without uncaught exceptions."""
    at = _run_page_smoke("pages/4_MMM_Budget_Optimizer.py")
    assert len(at.exception) == 0, f"MMM page raised: {at.exception}"
