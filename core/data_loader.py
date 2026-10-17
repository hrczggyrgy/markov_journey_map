"""DuckDB-alapú adatreteg: Parquet fájlokból villámgyors útvonal-szűrés."""

from __future__ import annotations

import pandas as pd
import duckdb


def load_aggregated_edges(
    parquet_path: str,
    rfm_segment: str = "ALL",
    lookback_days: int | None = None,
    group_by_col: str = "path",
) -> pd.DataFrame:
    """Villámgyors SQL-alapú útvonal-szűrés Parquet fájlból memóriatúlcsordulás nélkül.

    Visszatér egy DataFrame-del oszlopokkal:
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
    """RFM szegmens tábla betöltése (user_id, rfm_label, clv_tier)."""
    query = "SELECT * FROM read_parquet(?)"
    return duckdb.execute(query, [parquet_path]).df()


def load_mba_rules(parquet_path: str) -> pd.DataFrame:
    """MBA szabályok betöltése (order_id, bundle_type, net_margin, is_cross_sell_bundle)."""
    query = "SELECT * FROM read_parquet(?)"
    return duckdb.execute(query, [parquet_path]).df()


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """DataFrame mentése Parquet-ként DuckDB-rel."""
    duckdb.execute(f"COPY (SELECT * FROM df) TO '{path}' (FORMAT PARQUET)")


def load_csv_journeys(
    file,
    default_revenue_per_conversion: float = 25000.0,
    default_margin_rate: float = 0.30,
    default_segment: str = "Minden vásárló",
) -> pd.DataFrame:
    """
    CSV journey fájl betöltése és érvényesítése.

    Kötelező oszlopok: `path`, `conversions`, `nulls`.
    Opcionális (kiegészítve ha hiányzik): `rfm_segment`, `revenue`, `net_margin`.

    Visszatér egy normalizált DataFrame-del, amely már használható a
    MarkovJourneyEngine-hez is.
    """
    df = pd.read_csv(file)

    required = ["path", "conversions", "nulls"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó kötelező oszlopok: {', '.join(missing)}")

    # Opcionális oszlopok kiegészítése alapértékekkel
    if "rfm_segment" not in df.columns:
        df["rfm_segment"] = default_segment
    if "revenue" not in df.columns:
        df["revenue"] = df["conversions"].astype(float) * default_revenue_per_conversion
    if "net_margin" not in df.columns:
        df["net_margin"] = df["revenue"].astype(float) * default_margin_rate

    # Típusok homogénítása
    df["conversions"] = pd.to_numeric(df["conversions"], errors="coerce").fillna(0.0)
    df["nulls"] = pd.to_numeric(df["nulls"], errors="coerce").fillna(0.0)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    df["net_margin"] = pd.to_numeric(df["net_margin"], errors="coerce").fillna(0.0)
    df["path"] = df["path"].astype(str)
    df["rfm_segment"] = df["rfm_segment"].astype(str)

    return df
