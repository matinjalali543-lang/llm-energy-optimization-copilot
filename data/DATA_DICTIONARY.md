# Data dictionary

`data/processed/representative_days.csv` — **360 rows, 15 local days × 24 hours.**
The only data file this repository reads at runtime.

## Columns

| Column | Type | Range | Meaning |
|---|---|---|---|
| `local_timestamp` | str | 360 distinct | ISO 8601 with offset, `Europe/Berlin` |
| `day` | str | 15 distinct | Local calendar date, the key a request means by "tomorrow" |
| `hour` | int | 0 – 23 | Hour within the local day |
| `selected_because` | str | 10 distinct | Why this day is in the set — see below |
| `price_eur_mwh` | float | −90.010 – 200.040 | Day-ahead price. **Real, unchanged.** |
| `load_mw` | float | 0.208 – 0.492 | Site demand. Real shape, scaled to 0.50 MW peak. |
| `pv_mw` | float | 0.000 – 0.300 | Site PV. Real shape, scaled to 0.30 MWp. |
| `carbon_g_kwh` | float | 51.2 – 692.4 | **Proxy, not source data.** See `ATTRIBUTION.md`. |

Timestamps carry an offset and are local. Days are whole local days: the
preparation script drops the 23- and 25-hour clock-change days outright, because
a copilot asked to optimize "tomorrow" on such a day would silently plan the
wrong number of periods.

## The 15 days, and why each is here

| Day | Selected because | Price spread | Mean price | Negative hours | PV yield (MWh) |
|---|---|---:|---:|---:|---:|
| 2019-01-11 | dullest | 20.72 | 49.92 | 0 | 0.05 |
| 2019-01-24 | most expensive | 65.09 | 85.80 | 0 | 0.13 |
| 2019-06-08 | cheapest | 117.32 | **−42.24** | 19 | 1.72 |
| 2019-07-27 | narrowest price spread | **6.69** | 36.16 | 0 | 1.62 |
| 2019-09-13 | typical | 25.83 | 39.02 | 0 | 1.17 |
| 2019-10-13 | typical | 27.42 | 22.14 | 0 | 1.06 |
| 2019-11-20 | dirtiest grid | 32.44 | 53.25 | 0 | 0.20 |
| 2020-02-16 | most negative-price hours | 32.88 | −8.27 | **23** | 0.40 |
| 2020-02-22 | cleanest grid | 19.69 | −1.51 | 13 | 0.72 |
| 2020-04-30 | typical | 25.49 | 20.00 | 0 | 1.14 |
| 2020-06-01 | sunniest | 71.54 | 6.07 | 3 | **2.64** |
| 2020-06-09 | typical | 28.83 | 36.68 | 0 | 1.18 |
| 2020-07-15 | typical | 28.65 | 41.71 | 0 | 1.16 |
| 2020-09-21 | **widest price spread** | **164.58** | 61.10 | 0 | 1.75 |
| 2020-09-23 | typical | 28.35 | 46.43 | 0 | 1.10 |

Selection is a deterministic ranking over the whole snapshot — widest and
narrowest spread, sunniest and dullest, cheapest and most expensive, cleanest
and dirtiest, most negative-price hours, then six days nearest the median on
both spread and PV yield at once. Nothing was hand-picked.

The range matters more than the count. A copilot asked to "reduce cost tomorrow"
has 164 EUR/MWh of spread to work with on 2020-09-21 and 6.69 on 2019-07-27, and
a dataset containing only comfortable days would flatter the optimizer into
looking better than it is.

## Regenerating

```bash
python data/download_opsd.py     # ~130 MB, verifies the pinned sha256
python data/prepare_dataset.py
```

`prepare_dataset.py` verifies its own output before writing: every day exactly
24 hours, no missing values, no negative load or PV, carbon intensity within its
own emission factor, site load within the configured peak, and at least one day
with a usable price spread. A dataset that fails any of those is not written.
