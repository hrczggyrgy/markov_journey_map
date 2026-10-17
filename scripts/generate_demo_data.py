"""Minta Parquet adatok generálása a data_lake/ mappába.

A generált adatok célja, hogy a Streamlit alkalmazás vizuálisan kifejezőek
legyenek:

* **Sankey**: gazdaságos, hierarchikus forgalom (Start → figyelmeztetés →
  gondolkozás → konverzió → Konverzió/Null), thick és readable élekkel.
* **Attribúciós bar chart**: Volume_RE és Margin_RE eltérő színekkel, különböző
  csatornák getragenek (nincs egyformán 50%-os RE).
* **Hőtérkép**: korlátozott állapotszám, értelmezhető átmeneti valószínűségek.
* **What-if**: betekintő csatorna-kiesés (nincs 0%-os vagy 100%-os hatás).

Bemeneti szerződés:
* Egy sor = egy journey (path_id egyedi).
* Konverziós journey kap AOV-értékű bevételt és proportionalis nettó
  fedezetet; nem konverziós 0-t.
* Negatív margin = visszafizetés/retúra, null margin = ingyenes kampány.
* Nem minden konverziós journey kap MBA kosarat („Nem csatolt” kategória).
"""

from __future__ import annotations

import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_lake")

# --------------------------------------------------------------------------- #
# Path-sablonok: (útvonal, rfm_segment, konverziók, nulls, aov, margin_multiplier)
#
# A sablonokat úgy választjuk ki, hogy:
#   * rövid útvonalak (1-2 érintkezés): nagy forgalom, alacsony CVR (figyelmeztetés)
#   * középső útvonalak (3-4 érintkezés): közepes forgalom, magasabb CVR (gondolkozás)
#   * hosszú útvonalak (5-6 érintkezés): alacsony forgalom, legmagasabb CVR (loyalty)
#   * némely útvonal negatív margin-multiplikátorral (visszafizetés)
#   * némely null-marginral (ingyenes kampány)
BASE_JOURNEYS = [
    # --- Figyelmeztetés (awareness): hosszú null-halozat, alacsony CVR ---
    ("Social_Ads", "Új vásárlók", 40, 960, 15000, 1.0),
    ("Display_Ads", "Új vásárlók", 25, 975, 14000, 1.0),
    ("Social_Ads > Blog", "Új vásárlók", 15, 890, 18000, 1.0),
    ("Display_Ads > Social_Ads", "Új vásárlók", 20, 480, 17000, 1.0),
    ("Social_Ads > Search_NonBrand", "Új vásárlók", 35, 465, 20000, 1.0),
    ("Display_Ads > Search_NonBrand", "Új vásárlók", 18, 482, 19000, 1.0),
    # --- Gondolkozás (consideration): közepes forgalom, közepes CVR ---
    ("Social_Ads > Blog > Search_NonBrand", "Potenciális hűségesek", 60, 340, 24000, 1.0),
    ("Display_Ads > Social_Ads > Search_NonBrand", "Potenciális hűségesek", 50, 350, 25000, 1.0),
    ("Search_NonBrand > Blog > Email", "Potenciális hűségesek", 70, 230, 31000, 1.0),
    ("Social_Ads > Retargeting > Email", "Potenciális hűségesek", 55, 245, 26000, 1.0),
    ("Blog > Search_NonBrand > Email", "Potenciális hűségesek", 65, 235, 30000, 1.0),
    ("Display_Ads > Blog > Retargeting", "Potenciális hűségesek", 40, 360, 27000, 1.0),
    # --- Konverzió (conversion): alacsony forgalom, magas CVR ---
    ("Email > Search_Brand", "VIP (Champions)", 420, 80, 52000, 1.0),
    ("Search_Brand > Email", "VIP (Champions)", 310, 90, 48000, 1.0),
    ("Email", "VIP (Champions)", 260, 140, 45000, 1.0),
    ("Search_Brand", "VIP (Champions)", 290, 110, 49000, 1.0),
    ("Retargeting > Email > Search_Brand", "VIP (Champions)", 130, 70, 36000, 1.0),
    ("Search_NonBrand > Search_Brand", "VIP (Champions)", 90, 60, 32000, 1.0),
    ("Blog > Search_NonBrand > Email > Search_Brand", "VIP (Champions)", 55, 45, 38000, 1.0),
    ("Social_Ads > Blog > Search_NonBrand > Email > Search_Brand", "VIP (Champions)", 40, 60, 41000, 1.0),
    ("Direct", "VIP (Champions)", 180, 20, 43000, 1.0),
    ("Direct > Search_Brand", "VIP (Champions)", 95, 55, 40000, 1.0),
    # --- Visszafizetés / retúra: negatív nettó fedezet ---
    ("Email > Search_Brand", "VIP (Champions)", 8, 2, 52000, -1.0),
    ("Search_Brand", "VIP (Champions)", 5, 1, 49000, -1.0),
    # --- Ingyenes kampány: nulla margin ---
    ("Blog > Search_NonBrand > Email", "Potenciális hűségesek", 10, 10, 33000, 0.0),
    ("Social_Ads > Blog", "Új vásárlók", 8, 12, 18000, 0.0),
    # --- Lemorzsolódó / At Risk: alacsony CVR, magas null-halozat ---
    ("Retargeting > Email > Search_Brand", "Lemorzsolódó (At Risk)", 130, 270, 36000, 1.0),
    ("Social_Ads > Retargeting > Email", "Lemorzsolódó (At Risk)", 85, 315, 27500, 1.0),
    ("Retargeting", "Lemorzsolódó (At Risk)", 30, 370, 25000, 1.0),
    ("Email > Search_Brand", "Lemorzsolódó (At Risk)", 60, 440, 31000, 1.0),
]


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
    # 2. customer_journeys.parquet — egy sor = egy journey
    # ------------------------------------------------------------------ #
    rows = []
    journey_id = 0
    for path, seg, conv, nulls, aov, margin_mult in BASE_JOURNEYS:
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
    # 3. transaction_baskets_mba.parquet — csak konverziós journey-khoz
    # ------------------------------------------------------------------ #
    converting_journey_rows = journeys[journeys["conversions"] == 1].set_index("path_id")

    # Csak egy részhalmaz konverziós journey kap kosarat (~70%), így az
    # „Egytermékes” és a „Nem csatolt” kategóriák is jelentősek maradnak.
    n_baskets_target = int(len(converting_journey_rows) * 0.70)
    basket_ids = rng.choice(converting_journey_rows.index.tolist(), size=n_baskets_target, replace=False)

    basket_rows = []
    order_id = 0
    for pid in basket_ids:
        order_id += 1
        journey_row = converting_journey_rows.loc[pid]
        is_cross = bool(rng.choice([True, False], p=[0.35, 0.65]))
        journey_margin = float(journey_row["net_margin"])
        if is_cross:
            basket_margin = journey_margin * rng.uniform(1.1, 1.5)
            bundle_type = "Cross-sell"
        else:
            basket_margin = journey_margin * rng.uniform(0.3, 0.7)
            bundle_type = "None"
        basket_rows.append({
            "order_id": f"O{order_id:06d}",
            "path_id": pid,
            "is_cross_sell_bundle": is_cross,
            "bundle_type": bundle_type,
            "net_margin": round(max(basket_margin, 0.0), 0),
            "anchor_category": rng.choice(["Electronics", "Fashion", "Home", "Beauty"]),
        })
    baskets = pd.DataFrame(basket_rows)
    duckdb.execute(f"COPY (SELECT * FROM baskets) TO '{DATA_DIR}/transaction_baskets_mba.parquet' (FORMAT PARQUET)")

    # ------------------------------------------------------------------ #
    # 4. channel_daily_spend_mmm.parquet
    # ------------------------------------------------------------------ #
    channels_spend = ["Social_Ads", "Search_NonBrand", "Search_Brand", "Email", "Retargeting", "Display_Ads", "Blog", "Direct"]
    spend = pd.DataFrame({
        "channel": np.repeat(channels_spend, 90),
        "date": pd.date_range("2026-01-01", periods=90).tolist() * len(channels_spend),
        "spend": rng.exponential(400000, 90 * len(channels_spend)).round(0),
        "impressions": rng.exponential(2e6, 90 * len(channels_spend)).round(0),
        "clicks": rng.exponential(40000, 90 * len(channels_spend)).round(0),
    })
    duckdb.execute(f"COPY (SELECT * FROM spend) TO '{DATA_DIR}/channel_daily_spend_mmm.parquet' (FORMAT PARQUET)")

    print(f"Demo Parquet fájlok generálva: {DATA_DIR}")
    print(f"  journeyek: {len(journeys)} (konverziós: {int((journeys['conversions']==1).sum())})")
    print(f"  kosarak: {len(baskets)} (cross-sell: {int(baskets['is_cross_sell_bundle'].sum())})")
    print(f"  journeyek revenue osszege: {journeys['revenue'].sum():,.0f}")
    print(f"  journeyek margin osszege: {journeys['net_margin'].sum():,.0f}")


if __name__ == "__main__":
    generate()
