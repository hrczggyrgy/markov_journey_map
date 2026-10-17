"""DuckDB-based data layer: lightning-fast path filtering from Parquet files
without memory overflow."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import duckdb

from core.markov_engine import validate_journey_row, START, CONVERSION, NULL_STATE


def load_aggregated_edges(
    parquet_path: str,
    rfm_segment: str = "ALL",
    lookback_days: int | None = None,
    group_by_col: str = "path",
) -> pd.DataFrame:
    """Fast SQL-based path filtering from Parquet without memory overflow.

    Returns a DataFrame with columns:
      <group_by_col>, conversions, nulls, revenue, net_margin
    """
    where_clauses = []
    params: list = [parquet_path]

    if rfm_segment and rfm_segment != "ALL":
        where_clauses.append("rfm_segment = ?")
        params.append(rfm_segment)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    if lookback_days is not None and lookback_days > 0:
        where_sql += " AND journey_date >= CURRENT_DATE - ?"
        params.append(lookback_days)

    query = f"""
        SELECT
            {group_by_col},
            SUM(conversions) AS conversions,
            SUM(nulls) AS nulls,
            SUM(revenue) AS revenue,
            SUM(net_margin) AS net_margin
        FROM read_parquet(?)
        WHERE {where_sql}
        GROUP BY {group_by_col}
    """

    return duckdb.execute(query, params).df()


def load_rfm_segments(parquet_path: str) -> pd.DataFrame:
    """Load RFM segment table (user_id, rfm_label, clv_tier)."""
    query = "SELECT * FROM read_parquet(?)"
    return duckdb.execute(query, [parquet_path]).df()


def load_mba_rules(parquet_path: str) -> pd.DataFrame:
    """Load MBA rules (order_id, bundle_type, net_margin, is_cross_sell_bundle)."""
    query = "SELECT * FROM read_parquet(?)"
    return duckdb.execute(query, [parquet_path]).df()


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame as Parquet using DuckDB."""
    duckdb.execute(f"COPY (SELECT * FROM df) TO '{path}' (FORMAT PARQUET)")


def load_csv_journeys(
    file,
    default_revenue_per_conversion: float = 25000.0,
    default_margin_rate: float = 0.30,
    default_segment: str = "All customers",
) -> pd.DataFrame:
    """
    CSV journey file loading and validation.

    Required columns: `path`, `conversions`, `nulls`.
    Optional (filled if missing): `rfm_segment`, `revenue`, `net_margin`.

    Returns a normalized DataFrame ready for MarkovJourneyEngine.
    """
    df = pd.read_csv(file)

    required = ["path", "conversions", "nulls"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Read required numeric columns FIRST, before deriving any defaults.
    # Bad values are converted to NaN so they can be validated and reported,
    # not silently converted to zero.
    conversions_num = pd.to_numeric(df["conversions"], errors="coerce")
    nulls_num = pd.to_numeric(df["nulls"], errors="coerce")

    # Validation FIRST, before string conversion:
    # - missing path (NaN) must not silently become the "nan" channel
    # - empty or separator-only path means nothing
    path_str = df["path"].astype(str)

    def _has_meaningful_path(p) -> bool:
        s = str(p)
        return bool([t for t in s.split(">") if t.strip()])

    path_is_empty = ~path_str.map(_has_meaningful_path)
    path_is_missing = df["path"].isna()
    counts_bad = (
        conversions_num.isna()
        | nulls_num.isna()
        | ~np.isfinite(conversions_num)
        | ~np.isfinite(nulls_num)
        | (conversions_num < 0)
        | (nulls_num < 0)
    )
    bad_rows = path_is_missing | path_is_empty | counts_bad
    rejected = []
    if bad_rows.any():
        for idx in df.index[bad_rows]:
            reasons = []
            if path_is_missing.loc[idx]:
                reasons.append("missing path")
            if path_is_empty.loc[idx]:
                reasons.append(f"empty or separator-only path: {path_str.loc[idx]!r}")
            c = conversions_num.loc[idx]
            n = nulls_num.loc[idx]
            if pd.isna(c) or not np.isfinite(c) or c < 0:
                reasons.append(f"invalid conversion count: {c!r}")
            if pd.isna(n) or not np.isfinite(n) or n < 0:
                reasons.append(f"invalid null count: {n!r}")
            rejected.append(f"row {idx}: {'; '.join(reasons)}")
        raise ValueError(
            "CSV contains invalid rows that cannot be loaded:\n"
            + "\n".join(rejected[:10])
            + (f"\n... and {len(rejected) - 10} more rows" if len(rejected) > 10 else "")
        )

    # All remaining rows are valid: convert types.
    df["conversions"] = conversions_num.astype(float)
    df["nulls"] = nulls_num.astype(float)

    # Supplement optional columns with defaults (on clean numeric bases).
    if "rfm_segment" not in df.columns:
        df["rfm_segment"] = default_segment
    if "revenue" not in df.columns:
        df["revenue"] = df["conversions"].astype(float) * default_revenue_per_conversion
    if "net_margin" not in df.columns:
        df["net_margin"] = df["revenue"].astype(float) * default_margin_rate

    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").astype(float)
    df["net_margin"] = pd.to_numeric(df["net_margin"], errors="coerce").astype(float)
    # Shared validation contract: counts and margin are checked with the same
    # rules in both the CSV loader and direct engine usage.
    for idx, row in df.iterrows():
        validate_journey_row(
            str(row["path"]),
            float(row["conversions"]),
            float(row["nulls"]),
            float(row["net_margin"]),
            float(row["revenue"]),
        )

    # Financial finality: NaN or infinite values are not acceptable, but
    # negative financial amounts (e.g. refunds) are allowed.
    rev_bad = df["revenue"].isna() | ~np.isfinite(df["revenue"])
    if rev_bad.any():
        bad_idx = df.index[rev_bad].tolist()
        raise ValueError(
            "Invalid financial value in rows: " + ", ".join(map(str, bad_idx[:10]))
        )

    # Channel name constraints must be checked BEFORE reporting successful load:
    # k-order state names join with "__", and channel removal splits on "__".
    # The Engine would later reject with an error; here we stop with a usable
    # message instead.
    df["path"] = df["path"].astype(str)
    df["rfm_segment"] = df["rfm_segment"].astype(str)
    reserved = {START, CONVERSION, NULL_STATE}
    all_channels: set[str] = set()
    for steps in df["path"].str.split(">"):
        all_channels.update(ch.strip() for ch in steps if ch.strip())
    bad_channels = [ch for ch in all_channels if "__" in ch or ch in reserved]
    if bad_channels:
        raise ValueError(
            "Invalid channel name: "
            + ", ".join(repr(ch) for ch in bad_channels)
            + ". Channel name must not contain '__' characters and cannot "
            "equal reserved state names: "
            + ", ".join(repr(s) for s in sorted(reserved))
        )

    return df


def load_ga4_journeys(
    file,
    default_revenue_per_conversion: float = 25000.0,
    default_margin_rate: float = 0.30,
    default_segment: str = "All customers",
) -> pd.DataFrame:
    """
    Load a Google Analytics 4 (GA4) BigQuery export CSV and build journey data.

    GA4 export columns (expected): event_date, event_timestamp, event_name,
    user_pseudo_id, session_id, page_location, page_referrer, source, medium,
    campaign, event_params (JSON), items (JSON).

    Builds per-user journeys ordered by event_timestamp, marks purchase events
    as conversions, and aggregates to path-level counts/money.

    Returns a DataFrame with columns: path, conversions, nulls, revenue,
    net_margin, rfm_segment — ready for MarkovJourneyEngine.
    """
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["event_name", "user_pseudo_id", "event_timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "GA4 export missing required columns: " + ", ".join(missing)
        )

    def _extract_param(event_params_str, key):
        """Extract a param value from GA4 event_params JSON array."""
        if pd.isna(event_params_str):
            return None
        try:
            params = json.loads(event_params_str)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(params, list):
            return None
        for param in params:
            if isinstance(param, dict) and param.get("key") == key:
                val = param.get("value", {})
                if isinstance(val, dict):
                    return (
                        val.get("string_value")
                        or val.get("int_value")
                        or val.get("double_value")
                    )
                return val
        return None

    # Source/medium: from columns or event_params JSON
    if "source" not in df.columns:
        df["source"] = df["event_params"].apply(
            lambda x: _extract_param(x, "source")
        )
    if "medium" not in df.columns:
        df["medium"] = df["event_params"].apply(
            lambda x: _extract_param(x, "medium")
        )

    # Replace empty strings with NaN, then fillna
    df["source"] = df["source"].replace("", pd.NA).fillna("(direct)").astype(str)
    df["medium"] = df["medium"].replace("", pd.NA).fillna("(none)").astype(str)
    df["channel"] = df["source"] + " / " + df["medium"]

    # Page location (not used in path building, but kept for completeness)
    if "page_location" not in df.columns:
        df["page_location"] = df["event_params"].apply(
            lambda x: _extract_param(x, "page_location")
        )

    # Event value: GA4 stores in micros (1/1,000,000 of currency unit)
    if "value" not in df.columns:
        df["event_value"] = df["event_params"].apply(
            lambda x: _extract_param(x, "value")
        )
        df["event_value"] = pd.to_numeric(df["event_value"], errors="coerce")
    else:
        df["event_value"] = pd.to_numeric(df["event_value"], errors="coerce")
    df["event_value"] = df["event_value"].fillna(0.0)

    # Identify conversion events (purchase)
    df["is_conversion"] = df["event_name"].eq("purchase")

    # Sort by user and timestamp to build journeys
    df["event_timestamp"] = pd.to_numeric(df["event_timestamp"], errors="coerce")
    df = df.sort_values(["user_pseudo_id", "event_timestamp"])

    # Build one journey row per user
    journey_rows = []
    for _user_id, group in df.groupby("user_pseudo_id", sort=False):
        steps = []
        revenue = 0.0
        is_conv = False
        for _, row in group.iterrows():
            channel = str(row["channel"]).strip()
            if channel:
                steps.append(channel)
            if row["is_conversion"]:
                is_conv = True
                revenue += float(row["event_value"])

        path = " > ".join(steps)
        if not path:
            continue  # skip users with no channel info

        if is_conv:
            conversions = 1
            nulls = 0
        else:
            conversions = 0
            nulls = 1

        journey_rows.append(
            {
                "path": path,
                "conversions": conversions,
                "nulls": nulls,
                "revenue": revenue if is_conv else 0.0,
                "net_margin": (revenue * default_margin_rate) if is_conv else 0.0,
                "rfm_segment": default_segment,
            }
        )

    if not journey_rows:
        raise ValueError("No usable journeys found in GA4 export.")

    journeys_df = pd.DataFrame(journey_rows)

    # Aggregate by path string
    agg = (
        journeys_df.groupby("path")
        .agg(
            conversions=("conversions", "sum"),
            nulls=("nulls", "sum"),
            revenue=("revenue", "sum"),
            net_margin=("net_margin", "sum"),
            rfm_segment=("rfm_segment", "first"),
        )
        .reset_index()
    )

    # Validate through the shared contract
    for _, row in agg.iterrows():
        validate_journey_row(
            str(row["path"]),
            float(row["conversions"]),
            float(row["nulls"]),
            float(row["net_margin"]),
            float(row["revenue"]),
        )

    return agg

