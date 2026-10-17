"""Sparse (sparse), higher-order Markov journey engine with terminal margin reward.

Resource-efficient implementation instead of np.linalg.pinv(I - Q):
  * (I - Q^T) x = e_start sparse linear system via scipy.sparse
  * k-order states (e.g. 'Meta__Blog -> Search')
  * count-based (conversions, nulls) probabilistic chain
  * net margin as terminal reward: v = Qv + r_margin
  * N * 1 vector for computing transition counts
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


def validate_journey_row(
    path: str,
    conversions: float,
    nulls: float,
    net_margin: float,
    revenue: float | None = None,
) -> None:
    """Validate a journey row according to the model contract.

    The CSV loader and the engine boundary both enforce the same checks, so
    direct API usage gets the same strictness as the loader.
    """
    if not np.isfinite(conversions) or not np.isfinite(nulls) or conversions < 0 or nulls < 0:
        raise ValueError(
            f"Invalid conversion/null count on path {path!r}: "
            f"conversions={conversions!r}, nulls={nulls!r}"
        )
    if not np.isfinite(net_margin):
        raise ValueError(
            f"Invalid infinite financial value on path {path!r}: "
            f"net_margin={net_margin!r}"
        )
    if revenue is not None and not np.isfinite(revenue):
        raise ValueError(
            f"Invalid infinite financial value on path {path!r}: "
            f"revenue={revenue!r}"
        )
    # A zero-volume row (conversions=0, nulls=0) does not represent a journey,
    # but KPI aggregates could still pick up its money. Explicit contract:
    # a zero-volume journey cannot generate revenue or margin.
    total = conversions + nulls
    if total <= 0 and revenue is not None and (revenue != 0.0 or net_margin != 0.0):
        raise ValueError(
            f"Zero-volume journey '{path}' contains financial value: "
            f"revenue={revenue!r}, net_margin={net_margin!r}. Zero-volume "
            "journey cannot be associated with revenue or margin."
        )


def clean_journey(raw_steps: list[str], collapse_loops: bool = True) -> list[str]:
    """Collapse consecutive duplicate channels (Run-Length Encoding)."""
    if not collapse_loops or not raw_steps:
        return list(raw_steps)
    return [channel for channel, _ in groupby(raw_steps)]


def make_k_order_states(steps: list[str], k: int = 2) -> list[str]:
    """Transform a path into k-order states.

    One state is generated per observed touchpoint, with k-1 padding STARTs
    replacing earlier values. A deterministic three-touch path therefore keeps
    exactly three states and three physical transitions, regardless of k.

    E.g. ['Meta', 'Blog', 'Search'], k=2 -> ['(Start)__Meta', 'Meta__Blog', 'Blog__Search']
    """
    k = max(1, int(k))
    if not steps:
        return []
    padded = [START] * (k - 1) + list(steps)
    return ["__".join(padded[i : i + k]) for i in range(len(steps))]


@dataclass
class ChainFit:
    """Stored density matrices and state space for one fit."""

    states: list[str]
    state_idx: dict[str, int]
    Q_conv: sp.csr_matrix
    R_conv: sp.csr_matrix  # column 0: Conversion, column 1: Null
    r_margin: np.ndarray  # per-state expected immediate margin reward
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
    # Input path preprocessing
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
        """Break a path into edges (from, to, journey_count).

        Returns the edges and the last state (to which the terminal margin
        reward is attached).
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
    # Matrix construction
    # ------------------------------------------------------------------ #
    def _fit(self) -> None:
        raw_frm: list[str] = []
        raw_to: list[str] = []
        raw_w: list[float] = []

        channels_set: set[str] = set()
        # conv_margin[state] = margin sum on transition into Conversion;
        # departures[state] = total number of transitions out of state.
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
            # net_margin is the aggregate margin sum for this row's conversion.
            margin_dollars = float(row.get("net_margin", 0.0) or 0.0)
            revenue = float(row.get("revenue", 0.0) or 0.0)

            # Shared validation contract: the CSV loader and the direct API
            # usage perform the same checks. Invalid rows raise an exception
            # here instead of being silently skipped or coerced to zero.
            validate_journey_row(str(path_val), conv, nulls, margin_dollars, revenue)

            total = conv + nulls
            if total <= 0:
                continue

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

        # k-order state names join channel names with "__", and channel removal
        # searches "__"-split state names. The input contract therefore rejects
        # channel names containing "__" and reserved state names.
        reserved = {START, CONVERSION, NULL_STATE}
        bad_channels = [
            ch
            for ch in self.channels
            if "__" in ch or ch in reserved
        ]
        if bad_channels:
            raise ValueError(
                "Invalid channel name: "
                + ", ".join(repr(ch) for ch in bad_channels)
                + ". Channel name must not contain '__' characters and cannot "
                "equal reserved state names: "
                + ", ".join(repr(s) for s in sorted(reserved))
            )

        # State space: Start + every intermediate state referenced by edges.
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

        # Absorbing states point to themselves: Conversion->Conversion 1.0,
        # Null->Null 1.0 (keeps row sums at 1.0 for normalization).
        for abs_state in self.absorbing_states:
            idx = self._state_idx[abs_state]
            rows_num.extend([idx, idx])
            cols_num.extend([idx, idx])
            data_num.extend([1.0, 0.0])

        freq_conv = sp.csr_matrix(
            (data_num, (rows_num, cols_num)), shape=(n, n), dtype=np.float64
        )
        # The estimated count matrix contains real path edges and synthetic
        # absorbing self-loops (Conversion->Conversion, Null->Null) needed for
        # row normalization, but they do not represent observed traffic. The
        # public transition_counts accessor filters these out.
        self._freq_conv = freq_conv

        # Normalize rows into probability matrices
        row_sums = np.asarray(freq_conv.sum(axis=1)).flatten()
        inv = np.where(row_sums > 0, 1.0 / np.maximum(row_sums, 1e-12), 0.0)
        P_conv = freq_conv.multiply(sp.csr_matrix(inv[:, None]))

        # Canonical decomposition: Q (transient × transient), R (transient × absorbing)
        self.Q_conv = P_conv[: self.n_trans, : self.n_trans].tocsr()
        self.R_conv = P_conv[: self.n_trans, self.n_trans :].tocsr()

        # Terminal margin reward: r[i] = expected immediate margin on exit from
        # state i = conversion margin sum / number of exits.
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
        """Non-zero, observed transitions (source state, target state, count).

        Synthetic absorbing self-loops (Conversion->Conversion, Null->Null) do
        not represent traffic, so they are filtered out.
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
    # Sparse linear system: (I - Q^T) y = e_start
    # y[j] = expected number of visits to state j from Start.
    # ------------------------------------------------------------------ #
    def _solve_visit_system(self, A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
        """Solve A x = b and validate the result.

        Tries a direct sparse solve first (treating a singular-matrix warning as
        failure), then falls back to an iterative least-squares solve with
        convergence and residual checks. Raises a controlled ValueError when the
        system has no stable solution (e.g. a closed, non-absorbing state class)
        instead of manufacturing a finite but wrong result.
        """
        b_norm = float(np.linalg.norm(b))
        tol = 1e-6 * max(b_norm, 1e-12)

        def _validate(x: np.ndarray, source: str) -> np.ndarray:
            x = np.asarray(x).flatten()
            if not np.all(np.isfinite(x)):
                raise ValueError(
                    f"Non-absorbing Markov chain: {source} produced a non-finite "
                    "result for (I - Q^T) x = b."
                )
            residual = float(np.linalg.norm(b - A @ x))
            if residual > tol:
                raise ValueError(
                    f"Non-absorbing Markov chain: {source} residual {residual:.3e} "
                    f"exceeds tolerance {tol:.3e}. This can happen when a closed "
                    "state class cannot reach conversion or exit."
                )
            return x

        # 1. Direct sparse solve (with residual validation).
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", spla.MatrixRankWarning)
                x = spla.spsolve(A, b)
            return _validate(x, "spsolve")
        except spla.MatrixRankWarning:
            pass  # singular matrix: iterative fallback
        except (ValueError, RuntimeError, ArithmeticError):
            pass  # anticipated numerical failure: iterative fallback

        # 2. Iterative least-squares solve with convergence and residual checks.
        result = spla.lsqr(A, b, atol=1e-10, btol=1e-10)
        x = np.asarray(result[0]).flatten()
        istop = int(result[1])
        r1norm = float(result[3])  # ||r|| = ||b - A x||

        converged = istop == 1 and r1norm <= tol
        if not converged or not np.all(np.isfinite(x)):
            raise ValueError(
                f"Non-absorbing Markov chain: (I - Q^T) x = b has no stable "
                f"solution (istop={istop}, residual={r1norm:.3e}). This can happen "
                "when a closed state class cannot reach conversion or exit."
            )

        return _validate(x, "lsqr")

    def _visit_vector_from_start(self, Q: sp.csr_matrix) -> np.ndarray:
        """Return expected state visits from Start."""
        n = Q.shape[0]
        I_minus_Q = sp.eye(n, format="csr") - Q.T
        e_start = np.zeros(n)
        start_idx = self._state_idx.get(START)
        if start_idx is None or n == 0:
            return np.zeros(n)
        e_start[start_idx] = 1.0
        return self._solve_visit_system(I_minus_Q, e_start)

    def _absorption_prob(self, Q: sp.csr_matrix, R: sp.csr_matrix) -> float:
        """Probability of absorption into Conversion from Start."""
        y = self._visit_vector_from_start(Q)
        conv_col = np.asarray(R[:, 0].toarray()).ravel()
        return float(np.dot(y, conv_col))

    def _margin_value(self, Q: sp.csr_matrix, r_margin: np.ndarray) -> float:
        """Expected total margin (Ft) from Start."""
        y = self._visit_vector_from_start(Q)
        return float(y @ r_margin)

    # ------------------------------------------------------------------ #
    # Absorption probability / expected margin from Start state
    # ------------------------------------------------------------------ #
    def solve_prob(
        self,
        weight_col: str = "conversions",
        drop: Iterable[str] | None = None,
    ) -> float:
        """Probability ('conversions') or expected margin ('net_margin') from Start.

        weight_col: 'conversions' | 'net_margin'
        drop: channels to remove (outgoing edges redirected to Null,
              and their conversion reward removed).
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
    # Attribution: Removal Effect (count- and margin-based)
    # ------------------------------------------------------------------ #
    def compute_attribution(self) -> pd.DataFrame:
        empty = pd.DataFrame(
            columns=[
                "Channel",
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
        margin_reportable = base_margin > 0
        for ch in self.channels:
            rem_cvr = self.solve_prob(weight_col="conversions", drop=[ch])
            rem_margin = self.solve_prob(weight_col="net_margin", drop=[ch])
            re_vol = 1.0 - (rem_cvr / base_cvr) if base_cvr > 0 else 0.0
            # Margin_RE is signed: a channel removal that *improves* expected
            # margin yields a negative removal effect. Clipping it to zero would
            # hide a loss-making channel or an unsuitable baseline.
            if margin_reportable:
                re_margin = 1.0 - (rem_margin / base_margin)
            else:
                # A nonpositive baseline margin makes percentage attribution
                # unsuitable; report NaN instead of a misleading zero effect.
                re_margin = float("nan")
            rows.append(
                {
                    "Channel": ch,
                    "Volume_RE": max(0.0, re_vol),
                    "Margin_RE": re_margin,
                    "Volume_RE_pct": max(0.0, re_vol) * 100,
                    "Margin_RE_pct": re_margin * 100 if margin_reportable else float("nan"),
                }
            )
        return pd.DataFrame(rows).sort_values("Volume_RE", ascending=False)

    # ------------------------------------------------------------------ #
    # Fundamental matrix: expected number of steps to absorption
    # ------------------------------------------------------------------ #
    def expected_steps_to_absorption(self) -> dict[str, float]:
        """Expected number of steps from each transient state to absorption."""
        if self.Q_conv is None or self.n_trans == 0 or not self.channels:
            return {}
        n = self.n_trans
        I = sp.eye(n, format="csr")
        ones = np.ones(n)
        # Solve (I - Q) x = 1 with the same validated sparse solver as the visit vector.
        x = self._solve_visit_system(I - self.Q_conv, ones)
        x = np.asarray(x).flatten()
        return {
            state: float(x[self._state_idx[state]])
            for state in self.transient_states
        }
