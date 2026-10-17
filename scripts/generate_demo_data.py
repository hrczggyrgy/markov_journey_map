"""Minta Parquet adatok generálása a data_lake/ mappába."""

from __future__ import annotations

import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_lake")


def generate() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 5000

    # 1. unified_customers_rfm.parquet
    rfm = pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(n)],
        "rfm_segment": rng.choice(
            ["Új vásárlók", "Potenciális hűségesek", "VIP (Champions)", "Lemorzsolódó (At Risk)"],
            n, p=[0.4, 0.3, 0.2, 0.1],
        ),
        "clv_tier": rng.choice(["Bronze", "Silver", "Gold", "Platinum"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "recency_days": rng.exponential(30, n).round(0),
        "frequency": rng.poisson(3, n) + 1,
        "monetary": rng.exponential(60000, n).round(0),
    })
    duckdb.execute(f"COPY (SELECT * FROM rfm) TO '{DATA_DIR}/unified_customers_rfm.parquet' (FORMAT PARQUET)")

    # 2. transaction_baskets_mba.parquet
    baskets = pd.DataFrame({
        "order_id": [f"O{i:06d}" for i in range(n)],
        "path_id": [f"P{i:06d}" for i in range(n)],
        "is_cross_sell_bundle": rng.choice([True, False], n, p=[0.35, 0.65]),
        "bundle_type": rng.choice(["None", "Cross-sell", "Upsell"], n, p=[0.65, 0.25, 0.10]),
        "net_margin": rng.exponential(18000, n).round(0),
        "anchor_category": rng.choice(["Electronics", "Fashion", "Home", "Beauty"], n),
    })
    duckdb.execute(f"COPY (SELECT * FROM baskets) TO '{DATA_DIR}/transaction_baskets_mba.parquet' (FORMAT PARQUET)")

    # 3. customer_journeys.parquet
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
    rows = []
    for path, seg, conv, nulls, aov in base:
        rev = conv * aov
        margin = rev * 0.3
        journeys = conv + nulls
        for j in range(journeys):
            is_conv = j < conv
            rows.append({
                "path_id": f"J{len(rows):07d}",
                "path": path,
                "rfm_segment": seg,
                "conversions": 1 if is_conv else 0,
                "nulls": 0 if is_conv else 1,
                "revenue": rev if is_conv else 0,
                "net_margin": margin if is_conv else 0,
                "journey_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 90))),
            })
    journeys = pd.DataFrame(rows)
    duckdb.execute(f"COPY (SELECT * FROM journeys) TO '{DATA_DIR}/customer_journeys.parquet' (FORMAT PARQUET)")

    # 4. channel_daily_spend_mmm.parquet
    channels = ["Social_Ads", "Search_NonBrand", "Search_Brand", "Email", "Retargeting", "Display_Ads", "Blog"]
    spend = pd.DataFrame({
        "channel": np.repeat(channels, 90),
        "date": pd.date_range("2026-01-01", periods=90).tolist() * len(channels),
        "spend": rng.exponential(400000, 90 * len(channels)).round(0),
        "impressions": rng.exponential(2e6, 90 * len(channels)).round(0),
        "clicks": rng.exponential(40000, 90 * len(channels)).round(0),
    })
    duckdb.execute(f"COPY (SELECT * FROM spend) TO '{DATA_DIR}/channel_daily_spend_mmm.parquet' (FORMAT PARQUET)")

    print(f"Demo Parquet fájlok generálva: {DATA_DIR}")


if __name__ == "__main__":
    generate()