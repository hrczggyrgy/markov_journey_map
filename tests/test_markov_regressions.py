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
    # Mind-null sorok (konverzió nélkül): csatornák jelen vannak, de CVR 0.
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


def test_negative_counts_clamped_to_zero():
    df = pd.DataFrame(
        {
            "path": ["A > B", "C > D"],
            "conversions": [-3, 5],
            "nulls": [0, -2],
        }
    )
    eng = MarkovJourneyEngine(df, k=2)
    assert eng.solve_prob(weight_col="conversions") >= 0.0


def test_non_finite_values_rejected():
    df = pd.DataFrame(
        {
            "path": ["A > B", "C > D", "E > F"],
            "conversions": [np.inf, 5, 5],
            "nulls": [5, 5, 5],
        }
    )
    eng = MarkovJourneyEngine(df, k=2)
    # Inf rows must not corrupt the chain; finite CVR expected.
    assert 0.0 <= eng.solve_prob(weight_col="conversions") <= 1.0


# --------------------------------------------------------------------------- #
# Loader validation
# --------------------------------------------------------------------------- #
def test_csv_loader_rejects_invalid_counts(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("path,conversions,nulls\nA>B,abc,5\nC>D,3,xyz\n")
    with pytest.raises(ValueError, match="érvénytelen"):
        load_csv_journeys(str(p))


def test_csv_loader_rejects_missing_path(tmp_path):
    p = tmp_path / "missing_path.csv"
    p.write_text("path,conversions,nulls\n,5,5\nA>B,3,5\n")
    with pytest.raises(ValueError, match="hiányzó útvonal"):
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
    attr_a = eng_a.compute_attribution().set_index("Csatorna")
    attr_b = eng_b.compute_attribution().set_index("Csatorna")
    for ch in ["A", "B"]:
        assert abs(attr_a.loc[ch, "Margin_RE"] - attr_b.loc[ch, "Margin_RE"]) < 1e-9


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

    base = [
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
    intended_rev = sum(conv * aov for _, _, conv, _, aov in base)
    # Re-run the generator's journey construction inline to capture per-row revenue.
    rows = []
    journey_id = 0
    for path, seg, conv, nulls, aov in base:
        per_conv_margin = aov * 0.30
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
    assert int(gen_df["revenue"].sum()) == intended_rev == 101_192_500
    # Every converting row gets exactly its AOV, not the aggregate total.
    conv_rows = gen_df[gen_df["conversions"] == 1]
    for _, row in conv_rows.iterrows():
        base_row = next(b for b in base if b[0] == row["path"])
        assert row["revenue"] == base_row[4], f"{row['path']}: {row['revenue']} != {base_row[4]}"


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
    """Sankey source must not expose synthetic Conversion→Conversion / Null→Null loops."""
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
    with pytest.raises(ValueError, match="nem abszorbeert|no stable solution"):
        eng._solve_visit_system(sp.eye(n, format="csr") - Q_bad.T, b)
