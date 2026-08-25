# Attribution

## Source

**Open Power System Data** — Time series data package, hourly resolution.

- Package `opsd_time_series`, snapshot **`2020-10-06`**
- Landing page: https://data.open-power-system-data.org/time_series/2020-10-06
- File used: `time_series_60min_singleindex.csv` — 130,339,665 bytes, sha256
  `6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8`
  (verified against the file, not copied on trust — `data/download_opsd.py
  --verify-only` re-checks it)
- Package author: Jonathan Muehlenpfordt, Neon Neue Energieökonomik

**Upstream source of the series used here: the ENTSO-E Transparency Platform.**
OPSD republishes it; it did not originate the measurements.

The snapshot is pinned to a dated release rather than `latest` because OPSD
stopped updating the package, and because a source that changes underneath a
committed derivation is not a derivation.

## Columns taken

| OPSD column | Used for |
|---|---|
| `DE_LU_price_day_ahead` | `price_eur_mwh`, unchanged |
| `DE_load_actual_entsoe_transparency` | shape of `load_mw`, rescaled |
| `DE_solar_generation_actual` | shape of `pv_mw`, rescaled |
| `DE_wind_generation_actual` | carbon-intensity proxy only |

## What is real and what is this project's choice

Being explicit, because the distinction matters more than the data:

**Real, unchanged.** The day-ahead price, in EUR/MWh, including its negative
hours. One selected day has 19 hours below zero and another has 23; those are
measurements, not decoration.

**Real shape, chosen scale.** Load and PV. OPSD publishes a ~55 GW national
system; this project is one commercial site, so the series are rescaled to a
0.50 MW peak load and a 0.30 MWp array. Rescaling preserves the correlation
between price, sun and demand, which is the part dispatch depends on.

**Not source data at all.** `carbon_g_kwh`. See below.

## Carbon intensity is a proxy

The snapshot carries load, solar and wind. It carries **no fossil generation by
fuel**, so a genuine fuel-mix carbon intensity cannot be derived from it. This
was checked before the emissions objective was designed rather than after.

What the committed column contains:

```
fossil_share     = max(0, 1 - (wind + solar) / load)
carbon_g_kwh     = fossil_share * 700
```

This treats the entire non-wind, non-solar remainder as an average fossil unit
at 700 gCO₂/kWh. It **ignores nuclear, hydro and biomass**, and therefore reads
high — German nuclear alone was a substantial low-carbon share across this
period.

It is an assumption of this project. Results from the emissions objective
demonstrate that the objective changes the schedule; they are not a claim about
grams of CO₂ in Germany, and the README does not make one.

## Licence

OPSD's time series package is released under a free licence permitting reuse
with attribution; the upstream ENTSO-E data carries its own terms. This
repository redistributes only a **small derived extract** — 15 days, rescaled,
with a computed column — not the source file.

## Reproducing

```bash
python data/download_opsd.py     # ~130 MB, verifies the pinned hash
python data/prepare_dataset.py   # writes data/processed/representative_days.csv
```

The raw file is gitignored. The derived CSV is committed, which is what lets a
clone run everything with no network access.
