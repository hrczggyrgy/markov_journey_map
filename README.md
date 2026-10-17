# Markov Journey Mapper & Attribution Simulator

A sparse (sparse), higher-order, value-weighted absorbing Markov chain for customer journey analysis, Removal Effect calculation, and attribution dashboard. Integrated RFM, MBA (Market Basket Analysis), and MMM (Marketing Mix Modeling) bridges.

## Features

| Component | Description |
|-----------|-------------|
| **Sparse Higher-Order Engine** | k-order states (`k=1,2,3`), `scipy.sparse` linear system solver `(I − Qᵀ)y = e_start` instead of `np.linalg.pinv(I − Q)` |
| **Loop Collapse** | Removes consecutive duplicate channels via Run-Length Encoding |
| **Lookback Window** | Not applicable — aggregated input data contains no timestamps, so date-based path filtering is not possible |
| **Value-Weighted Attribution** | Volume_RE (count) + Margin_RE (net margin) Removal Effect computed together |
| **Conversion Speed** | Expected steps to absorption (`N·1`), step counts — not a time measure or per-channel "friction" |
| **DuckDB Data Layer** | Lightning-fast Parquet-based path filtering without memory overflow |
| **RFM → Markov** | Transition Drift (ΔP heatmap) between two segments |
| **MBA → Markov** | Basket affinity: cross-sell basket specialized Removal Effect |
| **Markov → MMM** | Bayesian LogNormal prior generation for channel coefficients |

## Project Structure

```
markov_journey/
├── app.py                      # Central Streamlit Executive Dashboard
├── core/
│   ├── __init__.py
│   ├── markov_engine.py        # Sparse, k-order, value-weighted Markov engine
│   ├── data_loader.py          # DuckDB + Parquet data layer, CSV import
│   └── bridge_connectors.py    # MMM/RFM/MBA integration bridges
├── pages/                      # Streamlit pages (multipage)
│   ├── __init__.py
│   ├── 1_RFM_Segmentation.py   # RFM → Markov Transition Drift
│   ├── 2_Market_Basket_MBA.py  # MBA → Markov basket affinity
│   └── 4_MMM_Budget_Optimizer.py  # Markov → MMM Bayesian prior
├── data_lake/                  # Parquet data store (samples)
├── scripts/                    # Data generation utilities
├── .github/workflows/ci.yml    # GitHub Actions CI
├── tests/                      # Regression and smoke test suite
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/hrczggyrgy/markov_journey_map.git
cd markov_journey_map

# Virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Generate sample data (if missing from data_lake/)
python scripts/generate_demo_data.py

# Run the application
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## Input Data Format (CSV)

Required columns:
- `path` — path sequence (e.g. `Social_Ads > Blog > Email`)
- `conversions` — number of conversions
- `nulls` — exits / non-converting journeys

Optional columns (filled with defaults if missing):
- `rfm_segment` — segment label (default: `"All customers"`)
- `revenue` — revenue (default: `conversions × 25,000 Ft`)
- `net_margin` — net margin (default: `revenue × 0.30`)

## Example Usage (Python API)

```python
import pandas as pd
from core.markov_engine import MarkovJourneyEngine

df = pd.read_csv("journeys.csv")
engine = MarkovJourneyEngine(df, k=2, collapse_loops=True)

base_cvr = engine.solve_prob(weight_col="conversions")
base_margin = engine.solve_prob(weight_col="net_margin")

attr = engine.compute_attribution()
print(attr[["Channel", "Volume_RE_pct", "Margin_RE_pct"]])

steps = engine.expected_steps_to_absorption()
```

## Margin Reward Model

Margin is modeled as a **terminal reward** earned on the transition into Conversion, not as money flowing through every intermediate edge. Expected margin from Start is computed as:

```
v = Qv + r_margin  ⟺  v = (I − Q)⁻¹ r_margin
```

This is aggregation-invariant: splitting an aggregated row into equivalent sub-rows does not change Margin_RE. Margin_RE can be negative (removing a loss-making channel improves expected margin). When the baseline margin is non-positive, percentage margin attribution is reported as `N/A`.

## Tab Reference

| Tab | Purpose |
|-----|---------|
| **Attribution** | Removal Effect by channel — volume and margin, side-by-side bar chart and detailed table |
| **Journeys** | Sankey diagram using real transition counts (not probability-scaled estimates), plus a bounded transition probability heatmap |
| **What-if** | Channel removal simulation — simulated conversion rate, conversion drop, and at-risk revenue |

## Testing

```bash
pytest tests/
```

The suite (43 tests) covers empty/all-zero input, missing paths, negative and non-finite counts, known single-channel CVR, deterministic journeys across k=1/2/3, equivalent aggregation, demo monetary reconciliation, basket-key integrity, duplicate join-key rejection, solver safeguards, channel-name delimiter validation, and application-level smoke tests for all pages.

CI runs on every push and pull request: dependency installation, compile check, `ruff check --select F821` (undefined-name lint), and `pytest`.

## License

MIT License — see `LICENSE`.

## Citation

If you use this tool for research or commercial purposes, please cite the repository.