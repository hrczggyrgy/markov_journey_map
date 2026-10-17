# Data Lake

Ez a mappá tartalmazza a mintául szolgáló Parquet adatfájlokat, amelykel az alkalmazás azonnal futtatható anélkül, hogy valós adatra lenne szükség.

| Fájl | Leírás | Használat |
|------|--------|-----------|
| `customer_journeys.parquet` | Útvonalak, konverziók, nullák, bevétel, nettó fedezet, RFM szegmens | Markov bemenet |
| `unified_customers_rfm.parquet` | 5 000 felhasználó RFM szegmenssel és CLV tierrel | RFM → Markov híd |
| `transaction_baskets_mba.parquet` | Kosár-affinitási adatok (cross-sell / egytermékes, anchor kategória) | MBA → Markov híd |
| `channel_daily_spend_mmm.parquet` | Csatornánkénti napi spend, impressions, clicks | MMM → Markov híd |

> **Megjegyzés:** Ha saját adatokat szeretnél használni, cseréld le ezeket a fájlokat (ugyanazon oszlopnevekkel), vagy tölts fel CSV-t az app oldalsávján (kötelező oszlopok: `path`, `conversions`, `nulls`).

Az adatfájlok generálása:
```python
# Lásd: scripts/generate_demo_data.py (közelében készülő)
```