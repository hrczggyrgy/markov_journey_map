"""Minta Parquet adatok generálása a data_lake/ mappába.

Generálja a demonstációs adatbázist a következő szempontok szerint:

* Egy sor = egy valós journey (path_id egyedileg azonosítja az utjat).
* Konverziós journey-k kapnak a teljes AOV-értékű bevételt és a proportionalis
  nettó fedezetet; nem konverziós journey-k 0-t.
* A MBA kosárak csak konverziós journey-khoz kapcsolódnak, és a kosár
  nettó fedezete a journey-hoz proportionalis (cross-sell kosarak magasabb,
  egytermékes alacsonyabb), így az enrich adott journey-hoz tartozik.
* Nem minden konverziós journey kap kosarat, így az "Egytermékes" és a
  "Nem csatolt" kategóriák is üresek maradnak.
* Egy kis részhalmaz journey-k negatív nettó fedezettel rendelkezik
  (visszafizetés/retúra), ezzel tesztelhető a signed Margin_RE megjelenítése.
"""

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

    # ------------------------------------------------------------------ #
    # 1. unified_customers_rfm.parquet
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2. customer_journeys.parquet — one row per actual journey
    # ------------------------------------------------------------------ #
    # (path, rfm_segment, conversions, nulls, aov, margin_multiplier)
    # margin_multiplier 1.0 = normál 30% árrés; <0 = visszafizetés (negatív margin).
    base = [
        ("Social_Ads > Blog > Search_NonBrand > Email", "Új vásárlók", 140, 420, 24000, 1.0),
        ("Social_Ads > Search_NonBrand > Search_Brand", "Új vásárlók", 190, 510, 26500, 1.0),
        ("Social_Ads > Blog", "Új vásárlók", 15, 890, 18000, 1.0),
        ("Social_Ads", "Új vásárlók", 20, 1450, 15000, 1.0),
        ("Display_Ads > Social_Ads > Search_NonBrand > Retargeting", "Új vásárlók", 95, 380, 29000, 1.0),
        ("Display_Ads", "Új vásárlók", 5, 1100, 14000, 1.0),
        ("Search_NonBrand > Blog > Email > Search_Brand", "Potenciális hűségesek", 210, 310, 34000, 1.0),
        ("Search_NonBrand > Retargeting > Search_Brand", "Potenciális hűségesek", 175, 290, 31000, 1.0),
        ("Search_NonBrand", "Potenciális hűségesek", 45, 720, 22000, 1.0),
        ("Email > Search_Brand", "VIP (Champions)", 420, 180, 52000, 1.0),
        ("Search_Brand > Email", "VIP (Champions)", 310, 120, 48000, 1.0),
        ("Email", "VIP (Champions)", 260, 240, 45000, 1.0),
        ("Search_Brand", "VIP (Champions)", 290, 190, 49000, 1.0),
        ("Retargeting > Email > Search_Brand", "Lemorzsolódó (At Risk)", 130, 410, 36000, 1.0),
        ("Social_Ads > Retargeting > Email", "Lemorzsolódó (At Risk)", 85, 520, 27500, 1.0),
        ("Retargeting", "Lemorzsolódó (At Risk)", 30, 640, 25000, 1.0),
        ("Blog > Search_NonBrand > Email", "Potenciális hűségesek", 110, 280, 33000, 1.0),
        ("Display_Ads > Blog > Retargeting", "Lemorzsolódó (At Risk)", 65, 490, 28000, 1.0),
        # Visszafizetés / retúra: negatív nettó fedezet वाली journey.
        ("Email > Search_Brand", "VIP (Champions)", 3, 2, 52000, -1.0),
        ("Search_Brand", "VIP (Champions)", 2, 1, 49000, -1.0),
        # Nulla-margin journey (ingyenes kampány).
        ("Blog > Search_NonBrand > Email", "Potenciális hűségesek", 5, 5, 33000, 0.0),
    ]
    rows = []
    journey_id = 0
    for path, seg, conv, nulls, aov, margin_mult in base:
        per_conv_margin = aov * 0.30 * margin_mult
        total_journeys = conv + nulls
        for j in range(total_journeys):
            is_conv = j < conv
            journey_id += 1
            rows.append({
                "path_id": f"J{journey_id:07d}",
                "path": path,
                "rfm_segment": seg,
                "conversions": 1 if is_conv else 0,
                "nulls": 0 if is_conv else 1,
                "revenue": aov if is_conv else 0.0,
                "net_margin": per_conv_margin if is_conv else 0.0,
                "journey_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 90))),
            })
    journeys = pd.DataFrame(rows)
    duckdb.execute(f"COPY (SELECT * FROM journeys) TO '{DATA_DIR}/customer_journeys.parquet' (FORMAT PARQUET)")

    # ------------------------------------------------------------------ #
    # 3. transaction_baskets_mba.parquet
    #
    # Kosárak csak konverziós journey-khoz kapcsolódnak. Nem minden konverziós
    # journey kap kosarat (~70% fedettség), így az "Egytermékes" és a
    # "Nem csatolt" kategóriák is jelentősek. A kosár nettó fedezete a journey
    # per-konverziós fedezetéből dériválódik (cross-sell magasabb, egytermékes
    # alacsonyabb), így az enrich adott journey-hoz tartozik.
    # ------------------------------------------------------------------ #
    converting_journey_ids = journeys.loc[journeys["conversions"] == 1, "path_id"].tolist()
    converting_journey_rows = journeys[journeys["conversions"] == 1].set_index("path_id")

    # Csak egy részhalmaznak kapunk kosarat (~70%).
    n_baskets_target = int(len(converting_journey_ids) * 0.70)
    basket_ids = rng.choice(converting_journey_ids, size=n_baskets_target, replace=False)

    basket_rows = []
    order_id = 0
    for pid in basket_ids:
        order_id += 1
        journey_row = converting_journey_rows.loc[pid]
        is_cross = bool(rng.choice([True, False], p=[0.35, 0.65]))
        journey_margin = float(journey_row["net_margin"])
        if is_cross:
            # Cross-sell kosár: magasabb árrés (1.1-1.5x a journey marginje).
            basket_margin = journey_margin * rng.uniform(1.1, 1.5)
            bundle_type = "Cross-sell"
        else:
            # Egytermékes kosár: alacsonyabb árrés (0.3-0.7x a journey marginje).
            basket_margin = journey_margin * rng.uniform(0.3, 0.7)
            bundle_type = "None"
        basket_rows.append({
            "order_id": f"O{order_id:06d}",
            "path_id": pid,
            "is_cross_sell_bundle": is_cross,
            "bundle_type": bundle_type,
            "net_margin": round(basket_margin, 0),
            "anchor_category": rng.choice(["Electronics", "Fashion", "Home", "Beauty"]),
        })
    baskets = pd.DataFrame(basket_rows)
    duckdb.execute(f"COPY (SELECT * FROM baskets) TO '{DATA_DIR}/transaction_baskets_mba.parquet' (FORMAT PARQUET)")

    # ------------------------------------------------------------------ #
    # 4. channel_daily_spend_mmm.parquet
    # ------------------------------------------------------------------ #
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
    print(f"  journeyek: {len(journeys)} (konverziós: {int((journeys['conversions']==1).sum())})")
    print(f"  kosarak: {len(baskets)} (cross-sell: {int(baskets['is_cross_sell_bundle'].sum())})")
    print(f"  journeyek revenue osszege: {journeys['revenue'].sum():,.0f}")
    print(f"  journeyek margin osszege: {journeys['net_margin'].sum():,.0f}")


if __name__ == "__main__":
    generate()
