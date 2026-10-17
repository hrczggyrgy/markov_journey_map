"""Integrációs modulok: Markov <-> MMM, RFM <-> Markov, MBA <-> Markov."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.markov_engine import MarkovJourneyEngine


def export_markov_priors_for_mmm(
    attr_df: pd.DataFrame,
    spend_dict: dict,
    confidence_weight: float = 0.3,
) -> dict:
    """Markov-alapú ROAS-ből Bayes-i prior paramétereket generál az MMM-hez.

    Visszatér egy dict-el: channel -> {markov_roas_estimate, prior_distribution, mu, sigma}
    """
    priors = {}
    for _, row in attr_df.iterrows():
        ch = row["Csatorna"]
        if ch not in spend_dict or spend_dict[ch] <= 0:
            continue

        markov_roas = row.get("Markov Bevétel (Ft)", row.get("revenue_attributed", 0)) / spend_dict[ch]

        sigma = max(0.15, 1.0 - confidence_weight)
        mu = np.log(max(markov_roas, 0.01)) - 0.5 * (sigma ** 2)

        priors[ch] = {
            "markov_roas_estimate": round(markov_roas, 3),
            "prior_distribution": "LogNormal",
            "mu": round(float(mu), 4),
            "sigma": round(float(sigma), 4),
        }
    return priors


def compare_rfm_transition_matrices(
    df: pd.DataFrame, seg_a: str, seg_b: str, path_sep: str = ">", k: int = 2
) -> pd.DataFrame:
    """Két RFM szegmens átmeneti valószínűségi mátrixának különbsége (delta-P).

    Pozitív érték: a 'seg_a' szegmensben gyakoribb az adott lépés.
    """
    eng_a = MarkovJourneyEngine(
        df[df["rfm_segment"] == seg_a], path_sep=path_sep, k=k
    )
    eng_b = MarkovJourneyEngine(
        df[df["rfm_segment"] == seg_b], path_sep=path_sep, k=k
    )

    common_states = sorted(
        list(set(eng_a.all_states).union(set(eng_b.all_states)))
    )

    def _prob_matrix(eng: MarkovJourneyEngine) -> pd.DataFrame:
        # Q + R egyben: átmeneti állapotok sorai, minden célállapot oszlopban
        n_trans = eng.n_trans
        P = np.zeros((len(eng.all_states), len(eng.all_states)))
        Q = eng.Q_conv.toarray() if eng.Q_conv is not None else np.zeros((n_trans, n_trans))
        R = eng.R_conv.toarray() if eng.R_conv is not None else np.zeros((n_trans, 2))
        P[:n_trans, :n_trans] = Q
        P[:n_trans, n_trans:] = R
        df_p = pd.DataFrame(P, index=eng.all_states, columns=eng.all_states)
        return df_p.reindex(index=common_states, columns=common_states, fill_value=0.0)

    P_a = _prob_matrix(eng_a)
    P_b = _prob_matrix(eng_b)

    delta_P = (P_a - P_b) * 100.0
    return delta_P


def enrich_journeys_with_mba_rules(
    journeys_df: pd.DataFrame, orders_mba_df: pd.DataFrame
) -> pd.DataFrame:
    """Útvonalak összekötése az MBA kimenettel (cross-sell súlyozás).

    orders_mba_df oszlopai: ['path_id', 'is_cross_sell_bundle', 'net_margin', 'anchor_category']
    """
    merged = journeys_df.merge(
        orders_mba_df,
        on="path_id",
        how="left",
        suffixes=("", "_mba"),
    )

    # Net margin: journeys-ből preferred, ha nem voller akkor MBA-ből
    if "net_margin_mba" in merged.columns:
        if "net_margin" not in merged.columns:
            merged["net_margin"] = merged["net_margin_mba"]
        else:
            merged["net_margin"] = merged["net_margin"].fillna(merged["net_margin_mba"])

    if "is_cross_sell_bundle" in merged.columns and "conversions" in merged.columns:
        merged["cross_sell_conversions"] = np.where(
            merged["is_cross_sell_bundle"] == True,
            merged["conversions"],
            0,
        )
    else:
        merged["cross_sell_conversions"] = 0.0

    return merged
