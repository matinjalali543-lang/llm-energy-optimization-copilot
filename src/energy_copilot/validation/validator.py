"""Deterministic validation: is this intent possible *at this site*?

The schema already established that the intent is meaningful. This layer asks
the different question of whether it is achievable here, and it is the last
thing between a request and a solver.

Three properties it deliberately has:

**It has never heard of the LLM.** It takes an `OptimizationIntent` and a
`SiteConfig` and nothing else. There is no prompt, no model name, no
conversation. That is what makes the security claim in `docs/decisions.md` §A
true by construction rather than by inspection: this code behaves identically
whether the intent came from a careful user, a helpful model, or an attacker
with complete control of the model's output.

**It reports every problem, not the first.** A request with three impossible
limits should come back with three reasons. Failing fast is right for a compiler
and wrong for a conversation, where each round trip costs the user a turn.

**Every rejection carries a reason and a code.** §10 requires the validator to
explain why. "Rejected" tells a user nothing; "this battery's floor is 10%, so a
minimum of 5% cannot be held" tells them what to ask for instead.

## Rejections and warnings are different

A rejection means the request cannot be honoured as stated. A warning means it
can, but the answer will be duller than the user expects — a cost objective on a
day with a 6.69 EUR/MWh spread is going to save almost nothing, and saying so up
front is better than presenting a rounding error as a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from energy_copilot.domain.config import SiteConfig, available_days, day_profile
from energy_copilot.domain.ontology import Asset, Objective
from energy_copilot.schemas.intent import OptimizationIntent, Rejection

# A price spread below this is not worth arbitraging after round-trip losses and
# degradation; the optimizer will return a schedule that barely moves. Not an
# error -- a warning, because the user is entitled to ask.
FLAT_DAY_SPREAD_EUR_MWH = 10.0


@dataclass
class ValidationReport:
    """Everything wrong with a request, and everything worth mentioning."""

    rejections: list[Rejection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejections

    def reject(self, code: str, reason: str, offending: object = None) -> None:
        self.rejections.append(
            Rejection(
                code=code,
                reason=reason,
                offending_value=None if offending is None else str(offending)[:200],
            )
        )

    def report(self) -> str:
        lines = []
        if self.rejections:
            lines.append("Rejected:")
            lines += [f"  - {rejection}" for rejection in self.rejections]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {warning}" for warning in self.warnings]
        return "\n".join(lines) if lines else "Valid."


def validate_intent(
    intent: OptimizationIntent,
    site: SiteConfig,
    data_path=None,
) -> ValidationReport:
    """Check an intent against the site. No LLM, no network, no prompt."""
    report = ValidationReport()

    _check_horizon(intent, report, data_path)
    _check_assets(intent, site, report)
    _check_objective(intent, site, report)
    _check_battery_constraints(intent, site, report)
    _check_grid_constraints(intent, site, report)
    _check_energy_requirement(intent, site, report)
    _check_comparison(intent, site, report)
    _warn_about_a_dull_day(intent, report, data_path)

    return report


# --- horizon ---------------------------------------------------------------


def _check_horizon(intent: OptimizationIntent, report: ValidationReport, data_path) -> None:
    horizon = intent.horizon
    if horizon.day is None:
        return
    try:
        day_profile(horizon.day, data_path)
    except KeyError:
        report.reject(
            "unknown_day",
            f"no data for {horizon.day}. This project ships 15 representative "
            f"days; available: {', '.join(available_days(data_path))}",
            horizon.day,
        )
    except ValueError as exc:
        report.reject("bad_day", str(exc), horizon.day)


# --- assets ----------------------------------------------------------------


def _check_assets(intent: OptimizationIntent, site: SiteConfig, report: ValidationReport) -> None:
    """An asset in the intent must exist at the site.

    The ontology already refused assets this project has never heard of. This
    catches the subtler case: a real asset type that this *particular* site does
    not have.
    """
    for asset in intent.assets:
        if asset is Asset.BATTERY and not site.has_battery:
            report.reject(
                "absent_asset",
                "this site has no battery configured, so nothing can be charged "
                "or discharged",
                asset.value,
            )
        elif asset is Asset.PV and site.pv_peak_mw <= 0:
            report.reject(
                "absent_asset", "this site has no PV array configured", asset.value
            )
        elif asset is Asset.FLEXIBLE_LOAD and not site.has_flexible_load:
            report.reject(
                "absent_asset",
                "this site has no flexible load configured, so there is nothing "
                "to shift",
                asset.value,
            )

    if intent.shift_flexible_loads and not site.has_flexible_load:
        report.reject(
            "absent_asset",
            "the request asks to shift flexible loads, but this site has none "
            "configured",
            "shift_flexible_loads",
        )


# --- objective -------------------------------------------------------------


def _check_objective(
    intent: OptimizationIntent, site: SiteConfig, report: ValidationReport
) -> None:
    """Is this objective answerable with what the site has?

    Self-consumption without PV is the case worth catching: the objective is
    perfectly well-defined and its answer is trivially zero, which would look
    like a working optimizer producing a useless schedule.
    """
    if intent.objective is Objective.MAXIMIZE_SELF_CONSUMPTION and site.pv_peak_mw <= 0:
        report.reject(
            "objective_not_answerable",
            "maximising self-consumption needs on-site generation, and this site "
            "has no PV. The answer would be zero at every hour.",
            intent.objective.value,
        )

    if intent.objective is Objective.MINIMIZE_PEAK and not site.has_battery:
        if not site.has_flexible_load:
            report.reject(
                "objective_not_answerable",
                "reducing peak demand needs a battery or a shiftable load, and "
                "this site has neither. Nothing can move the peak.",
                intent.objective.value,
            )
        else:
            report.warnings.append(
                "no battery configured, so peak reduction can only come from "
                "shifting flexible load"
            )


# --- battery ---------------------------------------------------------------


def _check_battery_constraints(
    intent: OptimizationIntent, site: SiteConfig, report: ValidationReport
) -> None:
    constraints = intent.constraints
    battery = site.battery

    asks_about_battery = any(
        value is not None
        for value in (
            constraints.minimum_soc_frac,
            constraints.maximum_soc_frac,
            constraints.terminal_soc_frac,
        )
    )
    if battery is None:
        if asks_about_battery:
            report.reject(
                "absent_asset",
                "the request constrains the state of charge, but this site has "
                "no battery",
                "state of charge",
            )
        return

    # A floor below the battery's own reserve is not a preference the optimizer
    # can honour: the equipment will not go there.
    if constraints.minimum_soc_frac is not None:
        if constraints.minimum_soc_frac < battery.soc_min_frac:
            report.warnings.append(
                f"requested SOC floor of {constraints.minimum_soc_frac:.0%} is below "
                f"this battery's reserve of {battery.soc_min_frac:.0%}; the reserve "
                "governs and the request cannot make the battery go lower"
            )
        if constraints.minimum_soc_frac > battery.soc_max_frac:
            report.reject(
                "infeasible_soc",
                f"a minimum state of charge of {constraints.minimum_soc_frac:.0%} is "
                f"above this battery's usable ceiling of {battery.soc_max_frac:.0%}; "
                "no schedule can satisfy it",
                constraints.minimum_soc_frac,
            )

    if constraints.maximum_soc_frac is not None:
        if constraints.maximum_soc_frac < battery.soc_min_frac:
            report.reject(
                "infeasible_soc",
                f"a maximum state of charge of {constraints.maximum_soc_frac:.0%} is "
                f"below this battery's floor of {battery.soc_min_frac:.0%}; the "
                "battery cannot be discharged that far",
                constraints.maximum_soc_frac,
            )

    if constraints.terminal_soc_frac is not None:
        terminal = constraints.terminal_soc_frac
        if terminal > battery.soc_max_frac:
            report.reject(
                "infeasible_soc",
                f"a closing state of charge of {terminal:.0%} exceeds this "
                f"battery's usable ceiling of {battery.soc_max_frac:.0%}",
                terminal,
            )
        elif terminal < battery.soc_min_frac:
            report.reject(
                "infeasible_soc",
                f"a closing state of charge of {terminal:.0%} is below this "
                f"battery's floor of {battery.soc_min_frac:.0%}",
                terminal,
            )
        else:
            # Reachability: can the battery physically get there in the horizon?
            needed = (terminal - battery.initial_soc_frac) * battery.energy_mwh
            hours = intent.horizon.hours
            reachable = battery.power_mw * hours * battery.charge_efficiency
            if needed > reachable + 1e-9:
                report.reject(
                    "unreachable_soc",
                    f"reaching {terminal:.0%} from {battery.initial_soc_frac:.0%} "
                    f"needs {needed:.3f} MWh, but at {battery.power_mw} MW over "
                    f"{hours} h the battery can absorb at most {reachable:.3f} MWh",
                    terminal,
                )


# --- grid ------------------------------------------------------------------


def _check_grid_constraints(
    intent: OptimizationIntent, site: SiteConfig, report: ValidationReport
) -> None:
    constraints = intent.constraints

    if constraints.max_grid_import_mw is not None:
        requested = constraints.max_grid_import_mw
        if requested > site.grid_import_limit_mw:
            # Almost always a unit error: 200 meaning 200 kW on a 0.6 MW site.
            report.reject(
                "above_connection_limit",
                f"a grid import limit of {requested} MW is above this site's "
                f"connection capacity of {site.grid_import_limit_mw} MW. A request "
                "cannot raise the size of the connection; if you meant kW, say so.",
                requested,
            )
        elif requested <= 0:
            report.reject(
                "infeasible_import",
                "a grid import limit of zero leaves the site with no supply when "
                "PV and battery cannot cover the load",
                requested,
            )

    if constraints.max_grid_export_mw is not None:
        if constraints.max_grid_export_mw > site.grid_export_limit_mw:
            report.reject(
                "above_connection_limit",
                f"an export limit of {constraints.max_grid_export_mw} MW exceeds "
                f"this site's export capacity of {site.grid_export_limit_mw} MW",
                constraints.max_grid_export_mw,
            )

    if constraints.allow_export and site.grid_export_limit_mw <= 0:
        report.reject(
            "export_not_permitted",
            "the request allows export, but this site's export limit is zero",
            "allow_export",
        )

    if constraints.max_peak_mw is not None and constraints.max_peak_mw > site.grid_import_limit_mw:
        report.warnings.append(
            f"a peak cap of {constraints.max_peak_mw} MW is above the connection "
            f"limit of {site.grid_import_limit_mw} MW, so it will never bind"
        )


# --- energy ----------------------------------------------------------------


def _check_energy_requirement(
    intent: OptimizationIntent, site: SiteConfig, report: ValidationReport
) -> None:
    required = intent.constraints.required_energy_mwh
    if required is None:
        return
    hours = intent.horizon.hours
    deliverable = site.grid_import_limit_mw * hours + site.pv_peak_mw * hours
    if required > deliverable:
        report.reject(
            "unreachable_energy",
            f"{required} MWh cannot be delivered in {hours} h: the connection and "
            f"PV together can supply at most {deliverable:.2f} MWh",
            required,
        )


# --- scenario comparison ---------------------------------------------------


def _check_comparison(
    intent: OptimizationIntent, site: SiteConfig, report: ValidationReport
) -> None:
    for variant in intent.comparison:
        if variant.battery_energy_mwh is not None and not site.has_battery:
            report.reject(
                "absent_asset",
                f"comparison '{variant.label}' varies the battery, but the base "
                "site has none to vary",
                variant.label,
            )
        if (
            variant.battery_energy_mwh is not None
            and variant.battery_power_mw is not None
            and variant.battery_power_mw > variant.battery_energy_mwh
        ):
            report.warnings.append(
                f"comparison '{variant.label}' has {variant.battery_power_mw} MW on "
                f"{variant.battery_energy_mwh} MWh — under one hour of storage, "
                "which is a very unusual C-rate"
            )


# --- warnings about the day ------------------------------------------------


def _warn_about_a_dull_day(
    intent: OptimizationIntent, report: ValidationReport, data_path
) -> None:
    """Say up front when the answer is going to be boring.

    A cost objective on the 6.69 EUR/MWh day will save almost nothing. That is a
    true and useful thing to know before the schedule appears, and reporting a
    rounding error as a triumph is how these systems lose trust.
    """
    if intent.objective is not Objective.MINIMIZE_COST or not intent.horizon.day:
        return
    try:
        profile = day_profile(intent.horizon.day, data_path)
    except (KeyError, ValueError):
        return  # already rejected in _check_horizon
    spread = float(profile["price_eur_mwh"].max() - profile["price_eur_mwh"].min())
    if spread < FLAT_DAY_SPREAD_EUR_MWH:
        report.warnings.append(
            f"{intent.horizon.day} has a price spread of only {spread:.2f} EUR/MWh. "
            "After round-trip losses and degradation there is very little to "
            "arbitrage, so expect a small saving."
        )


def summarise_profile(profile: pd.DataFrame) -> dict[str, float]:
    """Facts about a day, for warnings and explanations."""
    return {
        "price_min": float(profile["price_eur_mwh"].min()),
        "price_max": float(profile["price_eur_mwh"].max()),
        "price_spread": float(profile["price_eur_mwh"].max() - profile["price_eur_mwh"].min()),
        "negative_hours": int((profile["price_eur_mwh"] < 0).sum()),
        "pv_yield_mwh": float(profile["pv_mw"].sum()),
        "load_mwh": float(profile["load_mw"].sum()),
    }
