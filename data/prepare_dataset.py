"""Derive the committed site dataset from the pinned OPSD snapshot.

    python data/prepare_dataset.py

Run once. The CSV it writes is committed, so a clone needs neither the 130 MB
raw snapshot nor a network connection.

## Why days rather than years

A copilot's horizon is "tomorrow". It never needs two years of history the way a
forecasting project does, so this script does not commit two years: it selects a
small set of **representative local days** and commits those. The selection is a
documented, deterministic procedure over the whole snapshot, not a hand-picked
week that happened to look good.

The days are chosen to span the regimes that change the answer:

* the widest and narrowest day-ahead price spreads
* the sunniest and dullest days
* the day with the most negative-price hours
* the most and least expensive days by mean price
* plus typical days nearest the median on both spread and PV yield

A copilot asked to "reduce cost tomorrow" gives a different and more interesting
answer on a 90 EUR/MWh spread than on a 4 EUR/MWh one, and a dataset that
contained only comfortable days would flatter the optimizer.

## The site

OPSD publishes the German system: ~55 GW of load. This project is about a single
commercial site, so the *shapes* are real and the *scale* is chosen — stated
here rather than implied. Scaling preserves the correlation between price, PV
and load, which is the part that matters for dispatch.

## Carbon intensity is a proxy, not data

The snapshot carries load, solar and wind. It carries **no fossil generation by
fuel**, so a true fuel-mix carbon intensity cannot be derived from it. This was
checked before the emissions objective was designed.

What is computed instead is stated in one line:

    carbon_intensity = fossil_share * FOSSIL_EMISSION_FACTOR
    fossil_share     = max(0, 1 - (wind + solar) / load)

That treats the entire non-wind, non-solar remainder as though it were served by
an average fossil unit, which ignores nuclear, hydro and biomass and therefore
runs high. It is an **assumption of this project**, labelled as one everywhere
it appears, in the way project 03 labelled the line ratings its source did not
provide. Results from the emissions objective are about the mechanism, not about
grams of CO2 in Germany.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DEFAULT_RAW = HERE / "raw" / "time_series_60min_singleindex.csv"
OUT_PATH = HERE / "processed" / "representative_days.csv"

# --- the site. Shapes are real; these scales are a choice of this project. ---
SITE_PEAK_LOAD_MW = 0.50      # a small commercial site
SITE_PV_PEAK_MW = 0.30        # roof array, deliberately smaller than peak load
LOCAL_TZ = "Europe/Berlin"

# --- the carbon proxy. NOT source data. See the module docstring. ---
FOSSIL_EMISSION_FACTOR_G_PER_KWH = 700.0

COLUMNS = {
    "utc_timestamp": "utc_timestamp",
    "DE_LU_price_day_ahead": "price_eur_mwh",
    "DE_load_actual_entsoe_transparency": "system_load_mw",
    "DE_solar_generation_actual": "system_solar_mw",
    "DE_wind_generation_actual": "system_wind_mw",
}

N_TYPICAL_DAYS = 6


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is a 130 MB download and is deliberately not "
            "committed; fetch it with `python data/download_opsd.py`. The "
            "processed CSV this script writes IS committed, so you only need "
            "the raw file to re-derive."
        )
    frame = pd.read_csv(path, usecols=list(COLUMNS), parse_dates=["utc_timestamp"])
    frame = frame.rename(columns=COLUMNS).set_index("utc_timestamp").sort_index()
    # Every column must be present in a row for that hour to be usable. Dropping
    # here rather than interpolating keeps the committed data free of invented
    # values; the gaps are at the snapshot's edges, not scattered through it.
    return frame.dropna()


def to_site(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale the German system down to one site and add the carbon proxy."""
    out = pd.DataFrame(index=frame.index)
    out["price_eur_mwh"] = frame["price_eur_mwh"]
    out["load_mw"] = frame["system_load_mw"] / frame["system_load_mw"].max() * SITE_PEAK_LOAD_MW
    out["pv_mw"] = frame["system_solar_mw"] / frame["system_solar_mw"].max() * SITE_PV_PEAK_MW

    renewable = frame["system_wind_mw"] + frame["system_solar_mw"]
    fossil_share = (1.0 - renewable / frame["system_load_mw"]).clip(lower=0.0)
    out["carbon_g_kwh"] = fossil_share * FOSSIL_EMISSION_FACTOR_G_PER_KWH
    return out


def day_features(site: pd.DataFrame) -> pd.DataFrame:
    """One row per local day, with the properties the selection ranks on."""
    local = site.tz_convert(LOCAL_TZ)
    grouped = local.groupby(local.index.date)

    features = pd.DataFrame(
        {
            "hours": grouped.size(),
            "price_spread": grouped["price_eur_mwh"].max() - grouped["price_eur_mwh"].min(),
            "price_mean": grouped["price_eur_mwh"].mean(),
            "negative_hours": grouped["price_eur_mwh"].apply(lambda s: int((s < 0).sum())),
            "pv_yield_mwh": grouped["pv_mw"].sum(),
            "carbon_mean": grouped["carbon_g_kwh"].mean(),
        }
    )
    # Only whole days. A 23- or 25-hour day is a clock change, and a copilot
    # asked about "tomorrow" on such a day would silently optimize the wrong
    # number of periods.
    return features[features["hours"] == 24]


def select_days(features: pd.DataFrame) -> pd.DataFrame:
    """Pick the representative set, deterministically."""
    picks: dict[object, str] = {}

    def take(day, reason):
        # First reason wins, so a day that is both sunniest and cheapest is
        # labelled once and not committed twice.
        picks.setdefault(day, reason)

    take(features["price_spread"].idxmax(), "widest price spread")
    take(features["price_spread"].idxmin(), "narrowest price spread")
    take(features["pv_yield_mwh"].idxmax(), "sunniest")
    take(features["pv_yield_mwh"].idxmin(), "dullest")
    take(features["price_mean"].idxmax(), "most expensive")
    take(features["price_mean"].idxmin(), "cheapest")
    take(features["carbon_mean"].idxmax(), "dirtiest grid")
    take(features["carbon_mean"].idxmin(), "cleanest grid")
    if features["negative_hours"].max() > 0:
        take(features["negative_hours"].idxmax(), "most negative-price hours")

    # Typical days: nearest the median on both axes at once, by normalised
    # distance, so "typical" means typical in both respects rather than average
    # in one and extreme in the other.
    spread = features["price_spread"]
    yield_ = features["pv_yield_mwh"]
    distance = (
        ((spread - spread.median()) / spread.std()) ** 2
        + ((yield_ - yield_.median()) / yield_.std()) ** 2
    ) ** 0.5
    for day in distance.sort_values().index:
        if len(picks) >= 9 + N_TYPICAL_DAYS:
            break
        take(day, "typical")

    chosen = features.loc[sorted(picks)].copy()
    chosen["reason"] = [picks[day] for day in sorted(picks)]
    return chosen


def build(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    site = to_site(load_raw(raw_path))
    features = day_features(site)
    chosen = select_days(features)

    local = site.tz_convert(LOCAL_TZ)
    frames = []
    for day, row in chosen.iterrows():
        block = local[local.index.date == day].copy()
        block.insert(0, "day", str(day))
        block.insert(1, "hour", range(len(block)))
        block.insert(2, "selected_because", row["reason"])
        frames.append(block)

    dataset = pd.concat(frames)
    dataset.index.name = "local_timestamp"
    return dataset.reset_index(), chosen


def verify(dataset: pd.DataFrame) -> None:
    """Check the committed file before anyone depends on it."""
    problems = []

    counts = dataset.groupby("day").size()
    if not (counts == 24).all():
        problems.append(f"days without 24 hours: {counts[counts != 24].to_dict()}")
    if dataset["hour"].min() != 0 or dataset["hour"].max() != 23:
        problems.append("hour column is not 0..23")
    if dataset[["price_eur_mwh", "load_mw", "pv_mw", "carbon_g_kwh"]].isna().any().any():
        problems.append("missing values in the committed dataset")
    if (dataset["load_mw"] <= 0).any():
        problems.append("non-positive site load")
    if (dataset["pv_mw"] < 0).any():
        problems.append("negative PV")
    if not (0 <= dataset["carbon_g_kwh"]).all():
        problems.append("negative carbon intensity")
    if dataset["carbon_g_kwh"].max() > FOSSIL_EMISSION_FACTOR_G_PER_KWH + 1e-9:
        problems.append("carbon proxy exceeds its own emission factor")
    if dataset["load_mw"].max() > SITE_PEAK_LOAD_MW + 1e-9:
        problems.append("site load exceeds the configured peak")
    # The dataset is worthless for a cost-optimizing copilot if every day is
    # flat: there would be nothing to arbitrage and every objective would agree.
    spreads = dataset.groupby("day")["price_eur_mwh"].agg(lambda s: s.max() - s.min())
    if spreads.max() < 10:
        problems.append(f"no day has a usable price spread (max {spreads.max():.1f})")

    if problems:
        raise ValueError("the prepared dataset is not usable:\n  " + "\n  ".join(problems))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    dataset, chosen = build(args.raw)
    verify(dataset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {args.out}  ({len(dataset)} rows, {dataset['day'].nunique()} days)")
    print()
    print(
        chosen[["reason", "price_spread", "price_mean", "negative_hours", "pv_yield_mwh"]]
        .round(2)
        .to_string()
    )
    print(
        f"\nsite: {SITE_PEAK_LOAD_MW} MW peak load, {SITE_PV_PEAK_MW} MWp PV "
        "-- shapes are real, scales are this project's choice"
    )
    print(
        f"carbon intensity is a PROXY at {FOSSIL_EMISSION_FACTOR_G_PER_KWH:.0f} "
        "gCO2/kWh on the non-renewable remainder, not source data"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
