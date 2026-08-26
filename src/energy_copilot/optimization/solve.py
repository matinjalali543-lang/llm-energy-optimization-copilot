"""Solving, and reading the answer back as data rather than a live model.

A `DispatchResult` is a frozen snapshot: dataframes and floats, nothing that
holds a reference to a Pyomo object. Everything downstream — KPIs, explanations,
the UI — works from it, so no part of the system can quietly re-solve, mutate a
model, or read a variable that was never loaded.

`load_solutions=False` followed by an explicit `load_from` is deliberate. The
convenient path loads whatever the solver produced, including the values left
over from an infeasible attempt, and those look exactly like an answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
from pyomo.environ import SolverFactory, TerminationCondition, value

from energy_copilot.domain.config import SiteConfig
from energy_copilot.schemas.intent import OptimizationIntent

SOLVER_NAME = "appsi_highs"

# Below this, a flow is solver noise rather than a decision. Interior-point and
# simplex codes both return values a hair off their bounds; project 01 in this
# series had a test fail on -8.9e-09 because a lower bound carried no tolerance.
FLOW_TOLERANCE_MW = 1e-6


class InfeasibleRequest(RuntimeError):
    """The model could not be solved, with the intent that produced it."""

    def __init__(self, message: str, condition: str, intent: OptimizationIntent | None = None):
        super().__init__(message)
        self.condition = condition
        self.intent = intent


@dataclass(frozen=True)
class DispatchResult:
    """A solved day, as data."""

    schedule: pd.DataFrame
    objective_value: float
    objective_name: str
    energy_cost_eur: float
    degradation_cost_eur: float
    self_consumed_mwh: float
    emissions_kg: float
    peak_mw: float
    solve_seconds: float
    termination: str
    meta: dict = field(default_factory=dict)

    @property
    def total_cost_eur(self) -> float:
        return self.energy_cost_eur + self.degradation_cost_eur

    @property
    def battery_throughput_mwh(self) -> float:
        return float(self.schedule["charge_mw"].sum() + self.schedule["discharge_mw"].sum())

    @property
    def grid_import_mwh(self) -> float:
        return float(self.schedule["grid_import_mw"].sum())

    @property
    def grid_export_mwh(self) -> float:
        return float(self.schedule["grid_export_mw"].sum())

    @property
    def pv_curtailed_mwh(self) -> float:
        return float(
            (self.schedule["pv_available_mw"] - self.schedule["pv_used_mw"]).clip(lower=0).sum()
        )


def solve_model(
    model,
    intent: OptimizationIntent | None = None,
    solver_name: str = SOLVER_NAME,
    time_limit_s: float | None = None,
) -> DispatchResult:
    """Solve, check the answer is real, and return it as data."""
    solver = SolverFactory(solver_name)
    started = time.perf_counter()
    kwargs = {"load_solutions": False}
    if time_limit_s is not None:
        # Set on the config rather than passed as a keyword. In project 03 the
        # appsi legacy wrapper overwrote `config.time_limit` with a keyword
        # defaulting to None, so a 15 s limit ran for 221 s.
        try:
            solver.config.time_limit = time_limit_s
        except AttributeError:  # pragma: no cover - solver-dependent
            kwargs["timelimit"] = time_limit_s
    results = solver.solve(model, **kwargs)
    elapsed = time.perf_counter() - started

    condition = results.solver.termination_condition
    if condition not in (TerminationCondition.optimal, TerminationCondition.feasible):
        raise InfeasibleRequest(
            f"the request could not be satisfied: solver reported {condition}. "
            "This is a genuinely impossible combination of limits rather than a "
            "parsing problem — the intent validated before it got here.",
            condition=str(condition),
            intent=intent,
        )

    model.solutions.load_from(results)
    meta = dict(getattr(model, "_copilot", {}))
    hours = meta.get("hours", len(model.T))

    schedule = pd.DataFrame(
        {
            "hour": list(range(hours)),
            "price_eur_mwh": [meta["price"][t] for t in range(hours)],
            "load_mw": [meta["load"][t] for t in range(hours)],
            "pv_available_mw": [meta["pv_available"][t] for t in range(hours)],
            "carbon_g_kwh": [meta["carbon"][t] for t in range(hours)],
            "grid_import_mw": [_read(model.grid_import[t]) for t in range(hours)],
            "grid_export_mw": [_read(model.grid_export[t]) for t in range(hours)],
            "pv_used_mw": [_read(model.pv_used[t]) for t in range(hours)],
            "charge_mw": [_read(model.charge[t]) for t in range(hours)],
            "discharge_mw": [_read(model.discharge[t]) for t in range(hours)],
            "soc_mwh": [_read(model.soc[t]) for t in range(hours)],
        }
    )
    fixed_flex = meta.get("fixed_flex", [0.0] * hours)
    schedule["fixed_flex_mw"] = [fixed_flex[t] for t in range(hours)]

    flex_names = meta.get("flex_loads", ())
    if flex_names:
        for index, name in enumerate(flex_names):
            schedule[f"flex_{name}_mw"] = [
                _read(model.flex[index, t]) for t in range(hours)
            ]
    schedule["flex_total_mw"] = (
        schedule[[f"flex_{name}_mw" for name in flex_names]].sum(axis=1)
        if flex_names
        else 0.0
    )
    # What the site actually consumes, however the block was placed. The
    # before/after comparison is only honest if both sides consume the same
    # energy.
    schedule["total_demand_mw"] = (
        schedule["load_mw"] + schedule["fixed_flex_mw"] + schedule["flex_total_mw"]
    )

    return DispatchResult(
        schedule=schedule,
        objective_value=float(value(model.objective)),
        objective_name=meta.get("objective", "unknown"),
        energy_cost_eur=float(value(model.energy_cost)),
        degradation_cost_eur=float(value(model.degradation_cost)),
        self_consumed_mwh=float(value(model.self_consumed)),
        emissions_kg=float(value(model.emissions_kg)),
        # Read from the schedule, not from `model.peak`. Under any objective
        # that does not minimise it, `peak` is only bounded below, so nothing
        # pushes it down to the actual maximum and its value means nothing.
        peak_mw=float(schedule["grid_import_mw"].max()),
        solve_seconds=elapsed,
        termination=str(condition),
        meta=meta,
    )


def _read(variable) -> float:
    """Read a solved variable, tolerating one the presolve removed.

    A site with no battery gets `charge`, `discharge` and `soc` bounded to
    (0, 0). HiGHS presolves those away entirely, so asking Pyomo for a value
    raises `No value for uninitialized VarData` — which crashed every
    battery-less comparison arm while every battery-having one worked, so the
    test suite never saw it.

    A variable pinned to zero has a perfectly well-defined value. `exception=
    False` returns None for the missing one and zero is the answer.
    """
    return _clean(value(variable, exception=False))


def _clean(raw) -> float:
    """Snap solver noise to zero, in both directions."""
    number = float(raw if raw is not None else 0.0)
    return 0.0 if abs(number) < FLOW_TOLERANCE_MW else number


def check_no_simultaneous_flows(result: DispatchResult) -> list[str]:
    """Is the physically-absurd pair absent?

    One meter cannot import and export at once; one battery cannot charge and
    discharge at once. With `enforce_exclusive_flows=True` the binaries make
    both impossible, and this check confirms it rather than assuming it.

    It exists because the assumption *was* checked and *was* false. Built as a
    pure LP, 5 of 60 solves across 15 days and 4 objectives came back physically
    absurd — see `builder.py`'s module docstring for the two mechanisms. This
    function is what found them, and it still runs on every solved result so a
    future change to an objective cannot quietly reintroduce the problem.
    """
    problems = []
    both = result.schedule[
        (result.schedule["charge_mw"] > FLOW_TOLERANCE_MW)
        & (result.schedule["discharge_mw"] > FLOW_TOLERANCE_MW)
    ]
    if not both.empty:
        problems.append(
            f"battery charges and discharges together in hour(s) "
            f"{both['hour'].tolist()}"
        )
    trading = result.schedule[
        (result.schedule["grid_import_mw"] > FLOW_TOLERANCE_MW)
        & (result.schedule["grid_export_mw"] > FLOW_TOLERANCE_MW)
    ]
    if not trading.empty:
        problems.append(
            f"grid imports and exports together in hour(s) {trading['hour'].tolist()}"
        )
    return problems


def check_power_balance(result: DispatchResult, tolerance: float = 1e-6) -> list[str]:
    """Conservation, re-derived from the returned numbers.

    Deliberately independent of Pyomo: it recomputes the balance from the
    dataframe rather than asking the model whether its own constraint holds.
    """
    schedule = result.schedule
    supply = schedule["grid_import_mw"] + schedule["pv_used_mw"] + schedule["discharge_mw"]
    demand = (
        schedule["load_mw"]
        + schedule["fixed_flex_mw"]
        + schedule["charge_mw"]
        + schedule["grid_export_mw"]
        + schedule["flex_total_mw"]
    )
    residual = (supply - demand).abs()
    bad = schedule.loc[residual > tolerance, "hour"].tolist()
    return [f"power balance violated in hour(s) {bad}"] if bad else []


def solve_intent(
    intent: OptimizationIntent,
    site: SiteConfig,
    profile: pd.DataFrame,
    **kwargs,
) -> DispatchResult:
    """Build and solve in one step — the path the pipeline actually uses."""
    from energy_copilot.optimization.builder import build_model

    return solve_model(build_model(intent, site, profile), intent=intent, **kwargs)
