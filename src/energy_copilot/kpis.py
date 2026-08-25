"""Before, after, and the difference — all of it arithmetic, none of it prose.

§14 is unusually direct about this: "Calculate all numerical values in Python.
Do not ask the LLM to calculate financial KPIs." That instruction is the whole
reason this module exists as a separate thing rather than a paragraph in the
prompt.

The explanation layer downstream is given these numbers and told to describe
them. It is never given the schedule and asked what the saving was, because a
language model asked to sum a column will produce a number that looks like a
sum, and the failure is invisible: the wrong figure is formatted exactly like
the right one.

## What "before" means

The baseline is the honest do-nothing case, computed here without a solver:

* PV serves load directly, hour by hour
* surplus is exported up to the export limit, and curtailed beyond it
* the shortfall is imported
* the battery sits idle
* the flexible block runs on its default earliest-first schedule

It is not a strawman — it is what the site does with no optimizer at all, which
is the comparison a reader actually wants. Where the optimizer wins by very
little, that shows up here rather than being hidden by a worse baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from energy_copilot.domain.config import SiteConfig
from energy_copilot.optimization.solve import DispatchResult

HOURS_PER_PERIOD = 1.0


@dataclass(frozen=True)
class KPIs:
    """The §14 quantities, for one schedule."""

    cost_eur: float
    peak_mw: float
    grid_import_mwh: float
    grid_export_mwh: float
    pv_curtailed_mwh: float
    pv_used_mwh: float
    self_consumed_mwh: float
    battery_throughput_mwh: float
    emissions_kg: float
    demand_mwh: float

    def as_dict(self) -> dict[str, float]:
        return {key: round(float(value), 6) for key, value in asdict(self).items()}


def baseline_kpis(site: SiteConfig, profile: pd.DataFrame, hours: int = 24) -> KPIs:
    """The do-nothing case, computed directly. No LP, no solver, no LLM."""
    profile = profile.iloc[:hours].reset_index(drop=True)

    fixed_flex = [0.0] * len(profile)
    for block in site.flexible_loads:
        for hour, power in enumerate(block.default_schedule(len(profile))):
            fixed_flex[hour] += power

    demand = profile["load_mw"].to_numpy() + pd.Series(fixed_flex).to_numpy()
    pv = profile["pv_mw"].to_numpy()
    price = profile["price_eur_mwh"].to_numpy()
    carbon = profile["carbon_g_kwh"].to_numpy()

    pv_to_load = pv.clip(max=demand)
    surplus = (pv - pv_to_load).clip(min=0.0)
    exported = surplus.clip(max=site.grid_export_limit_mw)
    curtailed = surplus - exported
    imported = (demand - pv_to_load).clip(min=0.0)

    cost = float(
        (price * imported * HOURS_PER_PERIOD).sum()
        - (price * site.export_price_ratio * exported * HOURS_PER_PERIOD).sum()
    )
    return KPIs(
        cost_eur=cost,
        peak_mw=float(imported.max()),
        grid_import_mwh=float(imported.sum() * HOURS_PER_PERIOD),
        grid_export_mwh=float(exported.sum() * HOURS_PER_PERIOD),
        pv_curtailed_mwh=float(curtailed.sum() * HOURS_PER_PERIOD),
        pv_used_mwh=float((pv_to_load + exported).sum() * HOURS_PER_PERIOD),
        self_consumed_mwh=float((pv_to_load).sum() * HOURS_PER_PERIOD),
        battery_throughput_mwh=0.0,
        emissions_kg=float((carbon * imported * HOURS_PER_PERIOD).sum()),
        demand_mwh=float(demand.sum() * HOURS_PER_PERIOD),
    )


def result_kpis(result: DispatchResult) -> KPIs:
    """The same quantities, read off a solved schedule."""
    schedule = result.schedule
    return KPIs(
        cost_eur=result.total_cost_eur,
        peak_mw=result.peak_mw,
        grid_import_mwh=result.grid_import_mwh,
        grid_export_mwh=result.grid_export_mwh,
        pv_curtailed_mwh=result.pv_curtailed_mwh,
        pv_used_mwh=float(schedule["pv_used_mw"].sum() * HOURS_PER_PERIOD),
        self_consumed_mwh=result.self_consumed_mwh,
        battery_throughput_mwh=result.battery_throughput_mwh,
        emissions_kg=result.emissions_kg,
        demand_mwh=float(schedule["total_demand_mw"].sum() * HOURS_PER_PERIOD),
    )


def improvement(before: KPIs, after: KPIs) -> dict[str, float]:
    """Absolute and percentage change, per §14.

    Percentages are omitted rather than faked when the baseline is zero or when
    it straddles zero. A day whose baseline cost is −4.10 EUR (the site is paid
    to consume) and whose optimized cost is −19.98 has no meaningful "percentage
    saving": the usual formula returns −387%, which reads as a catastrophe and
    is in fact an improvement. Reporting the absolute change and withholding the
    ratio is the honest option, and this is a real case in the committed data,
    not a hypothetical.
    """
    out: dict[str, float] = {
        "cost_saving_eur": before.cost_eur - after.cost_eur,
        "peak_reduction_mw": before.peak_mw - after.peak_mw,
        "emissions_saving_kg": before.emissions_kg - after.emissions_kg,
        "curtailment_reduction_mwh": before.pv_curtailed_mwh - after.pv_curtailed_mwh,
        "self_consumption_gain_mwh": after.self_consumed_mwh - before.self_consumed_mwh,
        "import_reduction_mwh": before.grid_import_mwh - after.grid_import_mwh,
    }
    if before.cost_eur > 0:
        out["cost_saving_pct"] = 100.0 * out["cost_saving_eur"] / before.cost_eur
        if after.cost_eur < 0:
            # 2020-06-01 goes from a 63.80 EUR cost to a 19.81 EUR credit, which
            # the formula reports as a "131% saving" -- arithmetically right and
            # close to meaningless as prose. The flag lets the explanation say
            # "the bill became a credit" instead of quoting a percentage above
            # one hundred and hoping the reader works it out.
            out["cost_became_credit"] = 1.0
    if before.peak_mw > 0:
        out["peak_reduction_pct"] = 100.0 * out["peak_reduction_mw"] / before.peak_mw
    if before.emissions_kg > 0:
        out["emissions_saving_pct"] = 100.0 * out["emissions_saving_kg"] / before.emissions_kg
    return {key: round(float(value), 6) for key, value in out.items()}


def comparison_table(before: KPIs, after: KPIs) -> pd.DataFrame:
    """The §14 before/after/improvement block, as a frame."""
    rows = []
    labels = {
        "cost_eur": "Cost (EUR)",
        "peak_mw": "Peak import (MW)",
        "grid_import_mwh": "Grid import (MWh)",
        "grid_export_mwh": "Grid export (MWh)",
        "pv_curtailed_mwh": "PV curtailed (MWh)",
        "self_consumed_mwh": "PV self-consumed (MWh)",
        "battery_throughput_mwh": "Battery throughput (MWh)",
        "emissions_kg": "Emissions (kg, proxy)",
    }
    before_d, after_d = before.as_dict(), after.as_dict()
    for key, label in labels.items():
        rows.append(
            {
                "metric": label,
                "before": before_d[key],
                "after": after_d[key],
                "change": round(after_d[key] - before_d[key], 6),
            }
        )
    return pd.DataFrame(rows)


def check_demand_is_conserved(before: KPIs, after: KPIs, tolerance: float = 1e-6) -> list[str]:
    """Both sides must serve the same energy, or the comparison is meaningless.

    This is the guard against the most flattering possible bug: an "optimized"
    case that quietly serves less load than the baseline and books the
    difference as a saving.
    """
    gap = abs(before.demand_mwh - after.demand_mwh)
    if gap > tolerance:
        return [
            f"baseline serves {before.demand_mwh:.6f} MWh but the optimized case "
            f"serves {after.demand_mwh:.6f} MWh; a saving found by serving less "
            "load is not a saving"
        ]
    return []
