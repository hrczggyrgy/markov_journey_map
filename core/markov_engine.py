"""Ritkamátrixos (Sparse), k-adik rendű és fedezetsúlyozott Markov-útvonal motor.

Erőforrás-takarékos megvalósítás a np.linalg.pinv(I - Q) helyett:
  * (I - Q^T) x = e_start ritka lineáris egyenletrendszer megoldása scipy.sparse-szal
  * k-rendű állapotok (pl. 'Meta__Blog -> Search')
  * párhuzamos átmeneti mátrixok: konverziós sűrűség és nettó fedezet
  * N * 1 vektorból átlépések számának és átlagos időnek a kiszámolása
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

START = "(Start)"
CONVERSION = "(Conversion)"
NULL_STATE = "(Null)"


def clean_journey(raw_steps: list[str], collapse_loops: bool = True) -> list[str]:
    """Ötvözi az egymást közvetlenül követő duplikált csatornákat (Run-Length Encoding)."""
    if not collapse_loops or not raw_steps:
        return list(raw_steps)
    return [channel for channel, _ in groupby(raw_steps)]


def make_k_order_states(steps: list[str], k: int = 2) -> list[str]:
    """Útvonal átalakítása k-adik rendű állapotokká.

    Pl. ['Meta', 'Blog', 'Search'], k=2 -> ['(Start)__Meta', 'Meta__Blog', 'Blog__Search']
    """
    padded = [START] + list(steps)
    if len(padded) <= k:
        return ["__".join(padded)]
    return ["__".join(padded[i : i + k]) for i in range(len(padded) - k + 1)]


@dataclass
class ChainFit:
    """Eltárolt sűrűségi mátrixok és állapottér egy illesztésre."""

    states: list[str]
    state_idx: dict[str, int]
    Q_conv: sp.csr_matrix
    R_conv: sp.csr_matrix  # oszlop 0: Conversion, oszlop 1: Null
    Q_margin: sp.csr_matrix
    R_margin: sp.csr_matrix
    transient_states: list[str]
    absorbing_states: list[str]
    n_trans: int
    n_all: int
    path_weights: dict[tuple[str, str], dict[str, float]]


class MarkovJourneyEngine:
    """Higher-order, sparse, value-weighted Markov journey engine."""

    def __init__(
        self,
        df: pd.DataFrame,
        path_sep: str = ">",
        k: int = 2,
        collapse_loops: bool = True,
        lookback_days: int | None = None,
        date_col: str | None = None,
        conversion_date_col: str | None = None,
    ):
        self.df = df.copy()
        self.sep = path_sep
        self.k = max(1, int(k))
        self.collapse_loops = collapse_loops
        self.lookback_days = lookback_days
        self.date_col = date_col
        self.conversion_date_col = conversion_date_col

        self._states: list[str] = []
        self._state_idx: dict[str, int] = {}
        self.Q_conv: sp.csr_matrix | None = None
        self.R_conv: sp.csr_matrix | None = None
        self.Q_margin: sp.csr_matrix | None = None
        self.R_margin: sp.csr_matrix | None = None
        self.transient_states: list[str] = []
        self.absorbing_states: list[str] = [CONVERSION, NULL_STATE]
        self.all_states: list[str] = []
        self.n_trans = 0
        self.n_all = 0
        self.channels: list[str] = []

        self._fit()

    # ------------------------------------------------------------------ #
    # Bemeneti útvonalak előfeldolgozása
    # ------------------------------------------------------------------ #
    def _parse_steps(self, path_str: str) -> list[str]:
        steps = [s.strip() for s in str(path_str).split(self.sep) if s.strip()]
        return clean_journey(steps, self.collapse_loops)

    def _expand_to_edges(
        self,
        steps: list[str],
        total: float,
        conv: float,
        nulls: float,
        margin: float,
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Egy útvonal bontása élek/retepárakká k-rendű állapotokkal.

        Visszatér egy dict-el: {(from_state, to_state): {'conv': w, 'margin': w*margin}}
        """
        k_states = make_k_order_states(steps, self.k)
        edges: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"conv": 0.0, "margin": 0.0}
        )

        prev = START
        # (Start) -> első k-rendű állapot (teljes útvonal-számmal)
        first = k_states[0] if k_states else NULL_STATE
        edges[(prev, first)]["conv"] += total
        edges[(prev, first)]["margin"] += total * margin

        # Középsi átmenetek k-rendű állapottól állapothoz (teljes útvonal-számmal)
        for i in range(len(k_states) - 1):
            frm, to = k_states[i], k_states[i + 1]
            edges[(frm, to)]["conv"] += total
            edges[(frm, to)]["margin"] += total * margin

        # Utolsó állapot -> elnyelő állapotok: konverzió és null szétválasztva
        last = k_states[-1] if k_states else NULL_STATE
        edges[(last, CONVERSION)]["conv"] += conv
        edges[(last, CONVERSION)]["margin"] += conv * margin
        edges[(last, NULL_STATE)]["conv"] += nulls
        edges[(last, NULL_STATE)]["margin"] += 0.0  # null útvonal nem hoz bevételt

        return edges

    # ------------------------------------------------------------------ #
    # Mátrix építés
    # ------------------------------------------------------------------ #
    def _fit(self) -> None:
        rows_conv: list[int] = []
        cols_conv: list[int] = []
        data_conv: list[float] = []
        rows_margin: list[int] = []
        cols_margin: list[int] = []
        data_margin: list[float] = []

        channels_set: set[str] = set()
        edge_weights: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"conv": 0.0, "margin": 0.0}
        )

        for _, row in self.df.iterrows():
            steps = self._parse_steps(str(row["path"]))
            if not steps:
                continue

            conv = float(row.get("conversions", 0) or 0)
            nulls = float(row.get("nulls", 0) or 0)
            total = conv + nulls
            if total <= 0:
                continue

            margin = float(row.get("net_margin", row.get("avg_order_value", 0)) or 0)
            # Több csatornát is tartalmazó útvonalnál a margin-t osztjuk el
            w_margin = total * margin

            for ch in steps:
                channels_set.add(ch)

            edge_block = self._expand_to_edges(steps, total, conv, nulls, margin)
            for (frm, to), vals in edge_block.items():
                edge_weights[(frm, to)]["conv"] += vals["conv"]
                edge_weights[(frm, to)]["margin"] += vals["margin"]

        self.channels = sorted(channels_set)
        # Állapottér: Start + k-rendű állapotok, amiket a gyűjtött élek határoznak meg
        k_order_states: set[str] = set()
        for (frm, to) in edge_weights:
            k_order_states.add(frm)
            k_order_states.add(to)

        self.transient_states = [START] + sorted(k_order_states - {CONVERSION, NULL_STATE, START})
        self.all_states = self.transient_states + self.absorbing_states
        self._state_idx = {s: i for i, s in enumerate(self.all_states)}
        self.n_trans = len(self.transient_states)
        self.n_all = len(self.all_states)

        n = self.n_all
        for (frm, to), vals in edge_weights.items():
            i = self._state_idx[frm]
            j = self._state_idx[to]
            rows_conv.append(i)
            cols_conv.append(j)
            data_conv.append(vals["conv"])
            rows_margin.append(i)
            cols_margin.append(j)
            data_margin.append(vals["margin"])

        # Absorbing állapotok önmagukra mutassanak 1.0-sсал
        for abs_state in self.absorbing_states:
            idx = self._state_idx[abs_state]
            rows_conv.extend([idx, idx])
            cols_conv.extend([idx, idx])
            data_conv.extend([1.0, 0.0])
            rows_margin.extend([idx, idx])
            cols_margin.extend([idx, idx])
            data_margin.extend([1.0, 0.0])

        freq_conv = sp.csr_matrix(
            (data_conv, (rows_conv, cols_conv)), shape=(n, n), dtype=np.float64
        )
        freq_margin = sp.csr_matrix(
            (data_margin, (rows_margin, cols_margin)), shape=(n, n), dtype=np.float64
        )

        # Sorok normalizálása valószínűségi mátrixokká
        def _row_normalize(freq: sp.csr_matrix) -> sp.csr_matrix:
            row_sums = np.asarray(freq.sum(axis=1)).flatten()
            inv = np.where(row_sums > 0, 1.0 / np.maximum(row_sums, 1e-12), 0.0)
            return freq.multiply(sp.csr_matrix(inv[:, None]))

        P_conv = _row_normalize(freq_conv)
        P_margin = _row_normalize(freq_margin)

        # Kanonikus felbontás
        self.Q_conv = P_conv[: self.n_trans, : self.n_trans].tocsr()
        self.R_conv = P_conv[: self.n_trans, self.n_trans :].tocsr()
        self.Q_margin = P_margin[: self.n_trans, : self.n_trans].tocsr()
        self.R_margin = P_margin[: self.n_trans, self.n_trans :].tocsr()

        self._states = self.all_states
        self._state_idx = self._state_idx

    # ------------------------------------------------------------------ #
    # Ritka lineáris egyenletrendszer: (I - Q^T) y = e_start
    # A megoldásból y^T @ R kapjuk a (Start) sor elnyelési valóscheinlichségeit.
    # ------------------------------------------------------------------ #
    def _absorption_from_start(
        self, Q: sp.csr_matrix, R: sp.csr_matrix
    ) -> np.ndarray:
        """Visszatér a (Start) állapotból az elnyelési valóscheinlichségek vektorával."""
        n = Q.shape[0]
        I_minus_Q = sp.eye(n, format="csr") - Q.T
        e_start = np.zeros(n)
        e_start[self._state_idx[START]] = 1.0
        y = spla.spsolve(I_minus_Q, e_start)
        # y^T @ R  ->  (start-sor elnyelési valóscheinlichségei minden absorbing állapotra)
        return np.asarray(y @ R).flatten()

    def solve_prob(
        self,
        weight_col: str = "conversions",
        drop: Iterable[str] | None = None,
    ) -> float:
        """Elnyelési valószínűség a (Start) állapotból (Conversion oszlop).

        weight_col: 'conversions' | 'net_margin' (mely mátrixpárt használjuk)
        drop: kicsapandó csatornák (kimenő éleik Null-ba irányítása)
        """
        drop = set(drop or [])
        Q = self.Q_conv.copy()
        R = self.R_conv.copy()
        Qm = self.Q_margin.copy()
        Rm = self.R_margin.copy()

        if drop:
            for ch in drop:
                for state in list(self.transient_states):
                    if state == START:
                        continue
                    if ch in state.split("__"):
                        idx = self._state_idx[state]
                        Q[idx, :] = 0.0
                        Qm[idx, :] = 0.0
                        R[idx, 0] = 0.0
                        R[idx, 1] = 1.0
                        Rm[idx, 0] = 0.0

        abs_probs = self._absorption_from_start(
            Qm if weight_col == "net_margin" else Q,
            Rm if weight_col == "net_margin" else R,
        )
        # Oszlop 0 = (Conversion)
        return float(abs_probs[0])

    # ------------------------------------------------------------------ #
    # Attribúció: Removal Effect (dabszám- és fedezet-alapú)
    # ------------------------------------------------------------------ #
    def compute_attribution(self) -> pd.DataFrame:
        base_cvr = self.solve_prob(weight_col="conversions")
        base_margin = self.solve_prob(weight_col="net_margin")

        rows = []
        for ch in self.channels:
            rem_cvr = self.solve_prob(weight_col="conversions", drop=[ch])
            rem_margin = self.solve_prob(weight_col="net_margin", drop=[ch])
            re_vol = 1.0 - (rem_cvr / base_cvr) if base_cvr > 0 else 0.0
            re_margin = 1.0 - (rem_margin / base_margin) if base_margin > 0 else 0.0
            rows.append(
                {
                    "Csatorna": ch,
                    "Volume_RE": max(0.0, re_vol),
                    "Margin_RE": max(0.0, re_margin),
                    "Volume_RE_pct": max(0.0, re_vol) * 100,
                    "Margin_RE_pct": max(0.0, re_margin) * 100,
                }
            )
        return pd.DataFrame(rows).sort_values("Volume_RE", ascending=False)

    # ------------------------------------------------------------------ #
    # Fundamentális mátrix: várható lépések száma az elnyelésig
    # ------------------------------------------------------------------ #
    def expected_steps_to_absorption(self) -> dict[str, float]:
        """Mekkora átlagos lépésszám szükséges minden átmeneti állapotból a végpontig."""
        if self.Q_conv is None:
            return {}
        n = self.n_trans
        I = sp.eye(n, format="csr")
        # N = (I - Q)^-1  ->  N * 1 = x, ahol (I - Q) x = 1
        ones = np.ones(n)
        x = spla.spsolve(I - self.Q_conv, ones)
        return {
            state: float(x[self._state_idx[state]])
            for state in self.transient_states
        }
