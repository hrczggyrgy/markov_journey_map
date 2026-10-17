"""DuckDB-alapú adatreteg: Parquet fájlokból villámgyors útvonal-szűrés."""

from __future__ import annotations

import numpy as np
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

    # Szükséges numerikus oszlopok felolvasása ELŐRE, mielőtt bármilyen
    # alapértékeket deriváltunk belőlük. Hibás értékeket NaN-ra konvertáljuk,
    # hogy ezeket érvényesíthetjük és jeeljük, nem csendes Null-ra.
    conversions_num = pd.to_numeric(df["conversions"], errors="coerce")
    nulls_num = pd.to_numeric(df["nulls"], errors="coerce")

    # Érvényességi ellenőrzések ELSŐSORban, a string konverzió elött:
    # - hiányzó útvonal (NaN) nem láthatóaként "nan" csatorná váljon
    # - üres vagy szeparátoros csakból álló útvonal nem jelent semmit
    # - konverziós/kilépési számok végleges, nem negatív, nem negatív
    path_str = df["path"].astype(str)

    def _has_meaningful_path(p) -> bool:
        s = str(p)
        return bool([t for t in s.split(">") if t.strip()])

    path_is_empty = ~path_str.map(_has_meaningful_path)
    path_is_missing = df["path"].isna()
    counts_bad = conversions_num.isna() | nulls_num.isna() | ~np.isfinite(conversions_num) | ~np.isfinite(nulls_num) | (conversions_num < 0) | (nulls_num < 0)
    bad_rows = path_is_missing | path_is_empty | counts_bad
    rejected = []
    if bad_rows.any():
        for idx in df.index[bad_rows]:
            reasons = []
            if path_is_missing.loc[idx]:
                reasons.append("hiányzó útvonal")
            if path_is_empty.loc[idx]:
                reasons.append(f"üres vagy csak szeparátorból álló útvonal: {path_str.loc[idx]!r}")
            c = conversions_num.loc[idx]
            n = nulls_num.loc[idx]
            if pd.isna(c) or not np.isfinite(c) or c < 0:
                reasons.append(f"érvénytelen konverziós szám: {c!r}")
            if pd.isna(n) or not np.isfinite(n) or n < 0:
                reasons.append(f"érvénytelen kilépési szám: {n!r}")
            rejected.append(f"sor {idx}: {'; '.join(reasons)}")
        raise ValueError(
            "A CSV tartalmaz érvénytelen sorokat, amelyeket nem lehet betölteni:\n"
            + "\n".join(rejected[:10])
            + (f"\n... és még {len(rejected) - 10} további sor" if len(rejected) > 10 else "")
        )

    # Minden megmaradt sor érvényes: konvertáljuk típust.
    df["conversions"] = conversions_num.astype(float)
    df["nulls"] = nulls_num.astype(float)

    # Opcionális oszlopok kiegészítése alapértékekkel (már tisztán numerikus alapokon)
    if "rfm_segment" not in df.columns:
        df["rfm_segment"] = default_segment
    if "revenue" not in df.columns:
        df["revenue"] = df["conversions"].astype(float) * default_revenue_per_conversion
    if "net_margin" not in df.columns:
        df["net_margin"] = df["revenue"].astype(float) * default_margin_rate

    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").astype(float)
    df["net_margin"] = pd.to_numeric(df["net_margin"], errors="coerce").astype(float)
    # Pénzügyi értékek véglegességének ellenőrzése: NaN vagy végtelen érték nem
    # elfogadható, de a negatív pénzügyi összeg (pl. visszafizetés) engedélyezett.
    rev_bad = df["revenue"].isna() | ~np.isfinite(df["revenue"])
    mar_bad = df["net_margin"].isna() | ~np.isfinite(df["net_margin"])
    if rev_bad.any() or mar_bad.any():
        bad_idx = df.index[rev_bad | mar_bad].tolist()
        raise ValueError(
            "Érvénytelen pénzügyi érték a következő sorokban: "
            + ", ".join(map(str, bad_idx[:10]))
        )
    df["path"] = df["path"].astype(str)
    df["rfm_segment"] = df["rfm_segment"].astype(str)

    return df
