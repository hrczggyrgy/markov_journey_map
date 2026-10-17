# Markov Journey Mapper & Attribution Simulator

A ritkamátrixos (sparse), k-adik rendű és fedezetsúlyozott elnyelő Markov-lánc alapú vásárlói útvonal-elemző, Removal Effect kalkulátor és attribúciós igazságmérleg. Integrált RFM, MBA (Market Basket Analysis) és MMM (Marketing Mix Modeling) hídakkal.

## Főbb jellemzők

| Komponendent | Leírás |
|-------------|--------|
| **Sparse Higher-Order Motor** | k-adik rendű állapotok (`k=1,2,3`), `scipy.sparse` lineáris egyenletrendszer-megoldás `(I − Qᵀ)y = e_start` helyett `np.linalg.pinv(I − Q)` |
| **Loop Collapse** | Ötvöző hurkok eltávolítása Run-Length Encoding-gel |
| **Lookback ablak** | Csak a konverzió előtti X napi események láncolása |
| **Fedezett-súlyozott attribúció** | Volume_RE (dabszám) + Margin_RE (nettó fedezet) Removal Effect egyszerre |
| **Konverziós sebesség** | Várható lépésszám az elnyelésig (`N·1`), fricció pontok és journey accelerátorok |
| **DuckDB adatreteg** | Villámgyors Parquet-alapú útvonal-szűrés memóriatúlcsordulás nélkül |
| **RFM → Markov** | Transition Drift (ΔP hőtérkép) két szegmens között |
| **MBA → Markov** | Kosár-affinitás: cross-sell kosárakra specializált Removal Effect |
| **Markov → MMM** | Bayes-i LogNormal prior generálás a csatornaegyütthatókhoz |

## Mappastruktúra

```
markov_journey/
├── app.py                      # Központi Streamlit Executive Dashboard
├── core/
│   ├── __init__.py
│   ├── markov_engine.py        # Sparse, k-rendű, fedezetsúlyozott Markov motor
│   ├── data_loader.py          # DuckDB + Parquet adatreteg, CSV import
│   └── bridge_connectors.py    # MMM/RFM/MBA integrációs hídak
├── pages/                      # Streamlit oldalak (multipage)
│   ├── __init__.py
│   ├── 1_RFM_Segmentation.py   # RFM → Markov Transition Drift
│   ├── 2_Market_Basket_MBA.py  # MBA → Markov kosár-affinitás
│   └── 4_MMM_Budget_Optimizer.py  # Markov → MMM Bayes-i prior
├── data_lake/                  # Parquet adattár (minták)
├── scripts/                    # Adatgenerálási segédeszközök
├── .github/workflows/ci.yml    # GitHub Actions CI
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Gyorsindítás

```bash
# Klónozd a repot (cserélj le a ténylees repository URL-jevel)
git clone https://github.com/you/markov_journey.git
cd markov_journey

# Virtuális környezet
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Függőségek telepítése
pip install -r requirements.txt

# Alapadatok generálása (ha nincsenek a data_lake/-ben)
python scripts/generate_demo_data.py

# Alkalmazás indítása
streamlit run app.py
```

A böngészőben nyisd meg: `http://localhost:8501`

## Bemeneti adatformátum (CSV)

Kötelező oszlopok:
- `path` — útvonal szekvencia (pl. `Social_Ads > Blog > Email`)
- `conversions` — konverziók száma
- `nulls` — kilépések / nem konvertáló útvonalak száma

Opcionális oszlopok (ha hiányoznak, alapértékekkel kiegészül):
- `rfm_segment` — szegmens címke (alap: `"Minden vásárló"`)
- `revenue` — bevétel (alap: `conversions × 25 000 Ft`)
- `net_margin` — nettó fedezet (alap: `revenue × 0,30`)

## Példa használat (Python API)

```python
import pandas as pd
from core.markov_engine import MarkovJourneyEngine

df = pd.read_csv("journeys.csv")
engine = MarkovJourneyEngine(df, k=2, collapse_loops=True)

base_cvr = engine.solve_prob(weight_col="conversions")
base_margin = engine.solve_prob(weight_col="net_margin")

attr = engine.compute_attribution()
print(attr[["Csatorna", "Volume_RE_pct", "Margin_RE_pct"]])

steps = engine.expected_steps_to_absorption()
```

## CI / Tesztelés

A repository GitHub Actions folyamatot tartalmaz, amely minden push-re ellenőrzi a Python fájlok fordíthatóságát.

## Licenc

MIT License — lásd: `LICENSE`

## Címezés

Ha ezt az eszközt kutató vagy üzleti célra használsz, kérlek hivatkozz a repository-ra.