"""Ritkamátrixos (sparse), k-adik rendű Markov-útvonal motor terminális fedezet-jutalommal.

Erőforrás-takarékos megvalósítás np.linalg.pinv(I - Q) helyett:
  * (I - Q^T) x = e_start ritka lineáris egyenletrendszer scipy.sparse-szal
  * k-rendű állapotok (pl. 'Meta__Blog -> Search')
  * átmeneti lánc szorzós alapú (conversions, nulls) valószínűségi lánc
  * nettó fedezet terminális jutalomként: v = Qv + r_margin
  * N * 1 vektorból átlépések számának a kiszámolása
"""

from __future__ import annotations

import warnings
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

    Egyetlen állapot generálódik megfigyelt érintkezésenként, a k-1 pêldányos
    START kiesítéssel a korábbi értékek helyett. Így egy deterministic
    három-érintkezéses útvonal pontosan három állapotot és három fizikai
    átmenetetopoietic, függetlenül k-tol.

    Pl. ['Meta', 'Blog', 'Search'], k=2 -> ['(Start)__Meta', 'Meta__Blog', 'Blog__Search']
    """
    k = max(1, int(k))
    if not steps:
        return []
    padded = [START] * (k - 1) + list(steps)
    return ["__".join(padded[i : i + k]) for i in range(len(steps))]


@dataclass
class ChainFit:
    """Eltárolt sűrűségi mátrixok és állapottér egy illesztésre."""

    states: list[str]
    state_idx: dict[str, int]
    Q_conv: sp.csr_matrix
    R_conv: sp.csr_matrix  # oszlop 0: Conversion, oszlop 1: Null
    r_margin: np.ndarray  # állapotonkénti várható azonnali fedezet-jutalom
    transient_states: list[str]
    absorbing_states: list[str]
    n_trans: int
    n_all: int


class MarkovJourneyEngine:
    """Higher-order, sparse, count-based Markov journey engine with terminal margin reward.

    The transition chain (Q_conv, R_conv) is purely count-based: a real Markov
    chain over journeys. Margin is modelled as a *terminal reward* earned on the
    transition into Conversion (expected immediate margin per state), not as money
    flowing through every intermediate edge. Expected margin from Start is then

        v = Q v + r_margin   <=>   v = (I - Q)^-1 r_margin

    which is aggregation-invariant and keeps non-converting outcomes in the chain.
    """

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
        # lookback/date args are accepted for API compatibility but not applied:
        # the aggregate CSV/Parquet input carries no per-journey timestamps, so a
        # date filter cannot be meaningfully computed here.
        self.lookback_days = lookback_days
        self.date_col = date_col
        self.conversion_date_col = conversion_date_col

        self._states: list[str] = []
        self._state_idx: dict[str, int] = {}
        self.Q_conv: sp.csr_matrix | None = None
        self.R_conv: sp.csr_matrix | None = None
        self.r_margin: np.ndarray | None = None
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
    ) -> tuple[list[tuple[str, str, float]], str | None]:
        """Bontja az útvonalat élekre (from, to, journey_count).

        Visszatér az élekkel és az utolsó állapottal (amelyről a konverziós
        fedezet-jutalom attach-olódik terminálisan).
        """
        k_states = make_k_order_states(steps, self.k)
        edges: list[tuple[str, str, float]] = []

        if not k_states:
            return edges, None

        prev = START
        first = k_states[0]
        edges.append((prev, first, total))

        for i in range(len(k_states) - 1):
            edges.append((k_states[i], k_states[i + 1], total))

        last = k_states[-1]
        edges.append((last, CONVERSION, conv))
        edges.append((last, NULL_STATE, nulls))

        return edges, last

    # ------------------------------------------------------------------ #
    # Mátrix építés
    # ------------------------------------------------------------------ #
    def _fit(self) -> None:
        raw_frm: list[str] = []
        raw_to: list[str] = []
        raw_w: list[float] = []

        channels_set: set[str] = set()
        # conv_margin[state] = fedezet-osszeg a Conversionba iranyulo atmeneten;
        # departures[state] = osszes atmenet szama az allapotbol.
        conv_margin: dict[str, float] = defaultdict(float)
        departures: dict[str, float] = defaultdict(float)

        for _, row in self.df.iterrows():
            path_val = row["path"]
            if pd.isna(path_val):
                continue
            steps = self._parse_steps(str(path_val))
            if not steps:
                continue

            conv = float(row.get("conversions", 0) or 0)
            nulls = float(row.get("nulls", 0) or 0)
            # Konverziós/kilépési számoknak végleges, nem negatív értékeknek
            # kell lenniük (sosem lehet negatív átmeneti valószínűség).
            # Négatív pénzügyi értékek (net_margin) különlegesen kezelhetők,
            # azok nem eredményeznek negatív átmenetet.
            if not (np.isfinite(conv) and np.isfinite(nulls)) or conv < 0 or nulls < 0:
                continue
            total = conv + nulls
            if total <= 0:
                continue

            # net_margin egy soron aggregated fedezet-osszeg a konverziorol.
            margin_dollars = float(row.get("net_margin", 0.0) or 0.0)

            for ch in steps:
                channels_set.add(ch)

            edges, last = self._expand_to_edges(steps, total, conv, nulls)
            for frm, to, w in edges:
                raw_frm.append(frm)
                raw_to.append(to)
                raw_w.append(w)
                departures[frm] += w

            if last is not None and conv > 0:
                conv_margin[last] += margin_dollars

        self.channels = sorted(channels_set)

        # Állapottér: Start + minden intermédiaire állapot, amire egy élek hivatkozik.
        referenced: set[str] = set(raw_frm) | set(raw_to)
        k_order_states = {s for s in referenced if s not in (CONVERSION, NULL_STATE, START)}
        self.transient_states = [START] + sorted(k_order_states)
        self.all_states = self.transient_states + self.absorbing_states
        self._state_idx = {s: i for i, s in enumerate(self.all_states)}
        self.n_trans = len(self.transient_states)
        self.n_all = len(self.all_states)

        n = self.n_all
        rows_num: list[int] = []
        cols_num: list[int] = []
        data_num: list[float] = []
        for frm, to, w in zip(raw_frm, raw_to, raw_w):
            rows_num.append(self._state_idx[frm])
            cols_num.append(self._state_idx[to])
            data_num.append(w)

        # Absorbing állapotok önmagakra mutassanak: Conversion->Conversion 1.0,
        # Null->Null 1.0 (tartsa meg a sorösszet 1-essé a normalizáláshoz).
        for abs_state in self.absorbing_states:
            idx = self._state_idx[abs_state]
            rows_num.extend([idx, idx])
            cols_num.extend([idx, idx])
            data_num.extend([1.0, 0.0])

        freq_conv = sp.csr_matrix(
            (data_num, (rows_num, cols_num)), shape=(n, n), dtype=np.float64
        )
        # A becsolt átmenet-szárazat tartalmazza a valós útvonal-elekeket és a
        # szintetikus elnyelő önmagakra mutató éleket (Conversion→Conversion,
        # Null→Null), amelyek a sorok normalizálásához szükségesek, de nem
        # jelölnek megfigyelt forgalmat. A publikus transition_counts accessibility
        # ezen utolsót szűri ki.
        self._freq_conv = freq_conv

        # Sorok normalizálása valószínűségi mátrixokká
        row_sums = np.asarray(freq_conv.sum(axis=1)).flatten()
        inv = np.where(row_sums > 0, 1.0 / np.maximum(row_sums, 1e-12), 0.0)
        P_conv = freq_conv.multiply(sp.csr_matrix(inv[:, None]))

        # Kanonikus felbontás: Q (átmeneti × átmeneti), R (átmeneti × elnyelő)
        self.Q_conv = P_conv[: self.n_trans, : self.n_trans].tocsr()
        self.R_conv = P_conv[: self.n_trans, self.n_trans :].tocsr()

        # Terminal margin reward: r[i] = várható azonnali fedezet az állapotból
        # kilepéskor = fedezet-összeg a konverzióról / összes kilepés számából.
        self.r_margin = np.zeros(self.n_trans, dtype=np.float64)
        for state, val in conv_margin.items():
            if state in self._state_idx:
                i = self._state_idx[state]
                dep = departures.get(state, 0.0)
                if dep > 0:
                    self.r_margin[i] = val / dep

        self._states = self.all_states

    @property
    def transition_counts(self) -> list[tuple[str, str, float]]:
        """Null-nélküli, megfigyelt átmenetek (forrás állapot, cél állapot, darabszám).

        A szintetikus elnyelő önmagakra mutató élei (Conversion→Conversion,
        Null→Null) nem fuhrnak forgalom, ezért kiszűrésre kerülnek.
        """
        if self._freq_conv is None:
            return []
        coo = self._freq_conv.tocoo()
        absorbing = {CONVERSION, NULL_STATE}
        return [
            (self.all_states[int(i)], self.all_states[int(j)], float(w))
            for i, j, w in zip(coo.row, coo.col, coo.data)
            if w > 0
            and not (
                self.all_states[int(i)] in absorbing
                and self.all_states[int(j)] in absorbing
                and i == j
            )
        ]

    # ------------------------------------------------------------------ #
    # Ritka lineáris egyenletrendszer: (I - Q^T) y = e_start
    # y[j] = várható látogatásszám a j állapotra a Start-ból indulva.
    # ------------------------------------------------------------------ #
    def _solve_visit_system(self, A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
        """Megoldja A x = b a rendszerrel, és érvényesíti a megoldást.

        Először közvetlen ritka megoldást próbál (szinguláris mátriz figyelmeztetés
        hibaként), aztán iteratív legkisebb négyszögekkel egybecel convergencia és
        residuális ellenőrzéssel. Ha a rendszer nem stabilizing, egy kontrollált
        hibaüzenettel áll meg lugará_null helyett, hogy nullázatlan eredetet adna.
        """
        b_norm = float(np.linalg.norm(b))

        # 1. Közvetlen ritka megoldás.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", spla.MatrixRankWarning)
                x = spla.spsolve(A, b)
            x = np.asarray(x).flatten()
            if np.all(np.isfinite(x)):
                return x
        except Exception:
            pass  # fallback felé

        # 2. Iteratív legkisebb négyszögek convergencia- és residuális-ellenőrzéssel.
        result = spla.lsqr(A, b, atol=1e-10, btol=1e-10)
        x = np.asarray(result[0]).flatten()
        istop = int(result[1])
        r1norm = float(result[3])  # ||r|| = ||b - A x||

        # istop == 1: converged approximate solution. Residuális kell, hogy kis legyen
        # ||b|| relativben (itt ||b|| = 1.0 az indulási vektor Normja).
        converged = istop == 1 and r1norm <= 1e-6 * max(b_norm, 1e-12)
        if not converged or not np.all(np.isfinite(x)):
            raise ValueError(
                "Markov-lánc nem abszorbeert: (I - Q^T) x = b rendszernek nincs stabil "
                "megoldása. Ez akkor fordulhat elő, ha egy állapotkészlet záró hurokot "
                "épít, amelyről nem lehet konvertálni vagy kilépni. Residuális: "
                f"{r1norm:.3e}, solver állapot: {istop}."
            )

        return x

    def _visit_vector_from_start(self, Q: sp.csr_matrix) -> np.ndarray:
        """Visszatér a Start-ból indulva várható állapot-látogatásokkal."""
        n = Q.shape[0]
        I_minus_Q = sp.eye(n, format="csr") - Q.T
        e_start = np.zeros(n)
        start_idx = self._state_idx.get(START)
        if start_idx is None or n == 0:
            return np.zeros(n)
        e_start[start_idx] = 1.0
        return self._solve_visit_system(I_minus_Q, e_start)

    def _absorption_prob(self, Q: sp.csr_matrix, R: sp.csr_matrix) -> float:
        """Start-ból a Conversionba való elnyelési valószínűség."""
        y = self._visit_vector_from_start(Q)
        conv_col = np.asarray(R[:, 0].toarray()).ravel()
        return float(np.dot(y, conv_col))

    def _margin_value(self, Q: sp.csr_matrix, r_margin: np.ndarray) -> float:
        """Start-ból indulva várható összes fedezet (Ft)."""
        y = self._visit_vector_from_start(Q)
        return float(y @ r_margin)

    # ------------------------------------------------------------------ #
    # Elnyelési valószínűség / várható fedezet a Start állapotból
    # ------------------------------------------------------------------ #
    def solve_prob(
        self,
        weight_col: str = "conversions",
        drop: Iterable[str] | None = None,
    ) -> float:
        """Valószínűség ('conversions') vagy várható fedezet ('net_margin') a Start-ból.

        weight_col: 'conversions' | 'net_margin'
        drop: kicsapandó csatornák (kimenő éjeik Null-ba irányítása,
              és a konverziós jutalomjuk törlése).
        """
        drop = set(drop or [])
        Q = self.Q_conv.copy()
        R = self.R_conv.copy()
        r_margin = self.r_margin.copy() if self.r_margin is not None else np.zeros(self.n_trans)

        if drop:
            for ch in drop:
                for state in list(self.transient_states):
                    if state == START:
                        continue
                    if ch in state.split("__"):
                        idx = self._state_idx[state]
                        Q[idx, :] = 0.0
                        R[idx, 0] = 0.0
                        R[idx, 1] = 1.0
                        r_margin[idx] = 0.0

        if weight_col == "net_margin":
            return self._margin_value(Q, r_margin)
        return self._absorption_prob(Q, R)

    # ------------------------------------------------------------------ #
    # Attribúció: Removal Effect (dabszám- és fedezet-alapú)
    # ------------------------------------------------------------------ #
    def compute_attribution(self) -> pd.DataFrame:
        empty = pd.DataFrame(
            columns=[
                "Csatorna",
                "Volume_RE",
                "Margin_RE",
                "Volume_RE_pct",
                "Margin_RE_pct",
            ]
        )
        if self.n_trans == 0 or not self.channels:
            return empty

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
        if self.Q_conv is None or self.n_trans == 0 or not self.channels:
            return {}
        n = self.n_trans
        I = sp.eye(n, format="csr")
        ones = np.ones(n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", spla.MatrixRankWarning)
                x = spla.spsolve(I - self.Q_conv, ones)
        except Exception:
            x = spla.lsqr(I - self.Q_conv, ones, atol=1e-10, btol=1e-10)[0]
        x = np.asarray(x).flatten()
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            state: float(x[self._state_idx[state]])
            for state in self.transient_states
        }
