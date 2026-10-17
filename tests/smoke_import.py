"""Streamlit-page and core-module smoke test run by CI."""
import sys

sys.path.insert(0, ".")

from core.markov_engine import (
    MarkovJourneyEngine,
    clean_journey,
    make_k_order_states,
)
from core.data_loader import load_csv_journeys
from core.bridge_connectors import (
    export_markov_priors_for_mmm,
    compare_rfm_transition_matrices,
    enrich_journeys_with_mba_rules,
)

df = __import__("pandas").DataFrame(
    {
        "path": ["A > B", "C"],
        "conversions": [10, 5],
        "nulls": [5, 5],
        "net_margin": [100.0, 50.0],
    }
)
eng = MarkovJourneyEngine(df, k=2)
assert eng.solve_prob(weight_col="conversions") >= 0.0
assert not eng.compute_attribution().empty
print("Import smoke test: OK")
