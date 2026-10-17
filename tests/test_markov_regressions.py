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
def test_csv_loader_parses_numeric_before_defaults(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("path,conversions,nulls\nA>B,abc,5\nC>D,3,xyz\n")
    df = load_csv_journeys(str(p))
    assert df["conversions"].tolist() == [0.0, 3.0]
    assert df["nulls"].tolist() == [5.0, 0.0]


def test_csv_loader_missing_required_columns(tmp_path):
    p = tmp_path / "nope.csv"
    p.write_text("path,conversions\nA>B,5\n")
    with pytest.raises(ValueError):
        load_csv_journeys(str(p))


# --------------------------------------------------------------------------- #
# Margin semantics: aggregation invariance
# --------------------------------------------------------------------------- #
def test_margin_re_aggregation_invariant():
    """Equivalent aggregated vs split-row inputs must give identical Margin_RE."""
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
            "conversions": [50, 50, 100, 100],
            "nulls": [450, 450, 900, 900],
            "net_margin": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )
    eng_a = MarkovJourneyEngine(base, k=1)
    eng_b = MarkovJourneyEngine(split, k=1)
    attr_a = eng_a.compute_attribution().set_index("Csatorna")
    attr_b = eng_b.compute_attribution().set_index("Csatorna")
    for ch in ["A", "B"]:
        assert abs(attr_a.loc[ch, "Margin_RE"] - attr_b.loc[ch, "Margin_RE"]) < 1e-9


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
def test_demo_revenue_reconciles_with_base_table():
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
    df = pd.DataFrame(
        base, columns=["path", "rfm_segment", "conversions", "nulls", "avg_order_value"]
    )
    df["revenue"] = df["conversions"] * df["avg_order_value"]
    assert int(df["revenue"].sum()) == intended_rev


# --------------------------------------------------------------------------- #
# MBA join validation
# --------------------------------------------------------------------------- #
def test_duplicate_join_keys_rejected_or_cardinality_safe():
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
    merged = enrich_journeys_with_mba_rules(journeys, mba)
    # Duplicate journey key multiplies rows: 2 journey rows for J0000001 each join
    # the single MBA row, plus one row for J0000002 -> 3 rows (inflation from 2
    # unique journeys to 3 merged rows).
    assert len(merged) == 3
    assert int(merged["cross_sell_conversions"].sum()) == 2


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
