"""Scenario comparison: "compare a 1 MWh battery with a 2 MWh battery" (§17).

Each variant becomes its **own deterministic optimization** against the same
day, the same prices and the same load. The LLM is not asked what a bigger
battery would do; two models are built and solved, and the difference between
them is subtraction.

## Variants override the site, and only in ways the schema allows

`ScenarioVariant` is a closed set of fields — battery energy, battery power, PV
peak — so "what if the battery were twice the size" is expressible and "what if
this attribute were something else" is not. The override produces a **new frozen
`SiteConfig`**; the base site is never mutated, so a comparison cannot leave the
session in a different state than it found it.

## The base case is always included

A comparison of two hypotheticals with no reference point invites the reader to
assume the better one is an improvement on what they have. It might be worse
than both. The base site is solved alongside the variants and labelled, so the
question "is either of these actually better than today" has an answer on the
same table.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from energy_copilot.domain.config import Battery, SiteConfig, day_profile, scale_to_site
from energy_copilot.kpis import KPIs, baseline_kpis, improvement, result_kpis
from energy_copilot.optimization.solve import InfeasibleRequest, solve_intent
from energy_copilot.schemas.intent import OptimizationIntent, ScenarioVariant

BASE_LABEL = "as configured"


def apply_variant(site: SiteConfig, variant: ScenarioVariant) -> SiteConfig:
    """A new site with the variant's overrides. The original is untouched."""
    battery = site.battery
    if variant.battery_energy_mwh is not None or variant.battery_power_mw is not None:
        if battery is None:
            raise ValueError(
                f"variant '{variant.label}' changes the battery, but this site has "
                "none to change"
            )
        battery = replace(
            battery,
            energy_mwh=(
                variant.battery_energy_mwh
                if variant.battery_energy_mwh is not None
                else battery.energy_mwh
            ),
            power_mw=(
                variant.battery_power_mw
                if variant.battery_power_mw is not None
                else battery.power_mw
            ),
        )
    return replace(
        site,
        battery=battery,
        pv_peak_mw=variant.pv_peak_mw if variant.pv_peak_mw is not None else site.pv_peak_mw,
    )


@dataclass(frozen=True)
class ScenarioOutcome:
    """One arm of a comparison."""

    label: str
    site: SiteConfig
    kpis: KPIs | None
    baseline: KPIs | None
    delta: dict[str, float]
    failure: str | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None


def run_comparison(
    intent: OptimizationIntent,
    site: SiteConfig | None = None,
    profile: pd.DataFrame | None = None,
) -> list[ScenarioOutcome]:
    """Solve the base case and every variant against the same day."""
    site = site or SiteConfig()
    if profile is None:
        profile = day_profile(intent.horizon.day) if intent.horizon.day else day_profile(
            "2020-09-21"
        )

    # The comparison itself is stripped from the intent before solving: a
    # variant's model is an ordinary single-case model, and leaving the list in
    # place would invite a builder to recurse.
    single = intent.model_copy(update={"comparison": []})

    # (label, site or None, failure or None). One impossible variant must not
    # discard the arms that did work, so a failure is recorded and carried
    # rather than raised.
    arms: list[tuple[str, SiteConfig | None, str | None]] = [(BASE_LABEL, site, None)]
    for variant in intent.comparison:
        try:
            arms.append((variant.label, apply_variant(site, variant), None))
        except ValueError as exc:
            arms.append((variant.label, None, str(exc)))

    outcomes: list[ScenarioOutcome] = []
    for label, arm_site, failure in arms:
        if failure is not None or arm_site is None:
            outcomes.append(
                ScenarioOutcome(
                    label=label, site=site, kpis=None, baseline=None,
                    delta={}, failure=failure or "no site for this variant",
                )
            )
            continue

        arm_profile = scale_to_site(profile, arm_site)
        arm_baseline = baseline_kpis(arm_site, arm_profile, hours=len(arm_profile))
        try:
            result = solve_intent(single, arm_site, arm_profile)
        except InfeasibleRequest as exc:
            outcomes.append(
                ScenarioOutcome(
                    label=label, site=arm_site, kpis=None, baseline=arm_baseline,
                    delta={}, failure=str(exc),
                )
            )
            continue
        arm_kpis = result_kpis(result)
        outcomes.append(
            ScenarioOutcome(
                label=label,
                site=arm_site,
                kpis=arm_kpis,
                baseline=arm_baseline,
                delta=improvement(arm_baseline, arm_kpis),
            )
        )
    return outcomes


def comparison_frame(outcomes: list[ScenarioOutcome]) -> pd.DataFrame:
    """The arms side by side, with the base case first."""
    rows = []
    base = next((o for o in outcomes if o.label == BASE_LABEL and o.ok), None)
    for outcome in outcomes:
        if not outcome.ok:
            rows.append({"scenario": outcome.label, "status": f"failed: {outcome.failure[:60]}"})
            continue
        row = {
            "scenario": outcome.label,
            "status": "ok",
            "battery_mwh": outcome.site.battery.energy_mwh if outcome.site.battery else 0.0,
            "battery_mw": outcome.site.battery.power_mw if outcome.site.battery else 0.0,
            "pv_mwp": outcome.site.pv_peak_mw,
            "cost_eur": round(outcome.kpis.cost_eur, 3),
            "peak_mw": round(outcome.kpis.peak_mw, 4),
            "emissions_kg": round(outcome.kpis.emissions_kg, 1),
            "saving_vs_own_baseline_eur": outcome.delta.get("cost_saving_eur", 0.0),
        }
        if base is not None:
            # The number a reader actually wants: better or worse than today.
            row["vs_base_eur"] = round(base.kpis.cost_eur - outcome.kpis.cost_eur, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def describe_comparison(outcomes: list[ScenarioOutcome]) -> str:
    """A sentence about which arm won, with the margin.

    Deliberately says "the same" when the margin is under a cent: presenting a
    0.004 EUR difference as a winner is how a comparison stops being useful.
    """
    solved = [o for o in outcomes if o.ok]
    if len(solved) < 2:
        return "Not enough scenarios solved to compare."
    best = min(solved, key=lambda o: o.kpis.cost_eur)
    worst = max(solved, key=lambda o: o.kpis.cost_eur)
    margin = worst.kpis.cost_eur - best.kpis.cost_eur
    if margin < 0.01:
        return (
            f"All {len(solved)} scenarios cost the same to within a cent "
            f"({best.kpis.cost_eur:.2f} EUR); the difference between them does not "
            "matter on this day."
        )
    return (
        f"'{best.label}' is cheapest at {best.kpis.cost_eur:.2f} EUR, "
        f"{margin:.2f} EUR below '{worst.label}' at {worst.kpis.cost_eur:.2f} EUR."
    )


def default_battery_sweep(sizes_mwh: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)) -> list[dict]:
    """Variants for a quick battery-sizing sweep, used by the figures."""
    return [
        {"label": f"{size} MWh", "battery_energy_mwh": size, "battery_power_mw": size / 2}
        for size in sizes_mwh
    ]


def sweep_site(size_mwh: float, site: SiteConfig | None = None) -> SiteConfig:
    site = site or SiteConfig()
    base = site.battery or Battery()
    return replace(site, battery=replace(base, energy_mwh=size_mwh, power_mw=size_mwh / 2))
