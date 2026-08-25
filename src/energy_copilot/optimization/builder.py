"""The dispatch model: a linear program built from a *validated* intent.

The signature is the architecture. `build_model` takes an `OptimizationIntent`
and a `SiteConfig` — a schema type and a dataclass — and there is no path by
which text reaches it. The LLM cannot write a constraint here, cannot name a
variable, and cannot choose a solver, because the only thing that crosses this
boundary is a structure that had to validate first.

## The four objectives, and what each actually means

§12 lists four. They are not four weightings of one expression; they optimise
different things and produce visibly different schedules.

**`MINIMIZE_COST`** — energy bought at the day-ahead price, minus energy
exported at `export_price_ratio` of it, plus a degradation charge on throughput
so the battery does not cycle for a hundredth of a cent.

**`MINIMIZE_PEAK`** — the largest hourly grid import over the day. A single
variable bounded below by every hour's import; minimising it flattens the
profile whatever the price does.

**`MAXIMIZE_SELF_CONSUMPTION`** — PV that serves the site rather than the grid,
written as `sum(pv_used) - sum(export)`. Using a kWh of sun on site counts;
exporting it does not. Curtailing it counts for nothing either, so the optimizer
prefers to store surplus over spilling it.

**`MINIMIZE_EMISSIONS`** — carbon on imported energy, **with no credit for
export**. Crediting exports at the same intensity would make import and export
in the same hour exactly neutral, which leaves the solver free to return an
arbitrary pair of large offsetting flows that mean nothing. Refusing the credit
also states a defensible position: this site does not claim someone else's
avoided emissions.

## The binaries are necessary, and that was measured

Simultaneous charge and discharge is physically absurd, and the usual guard is a
binary. The tempting argument is that round-trip losses already make it
wasteful, so the constraint never binds and the binary can be dropped — which is
exactly what `forecast-driven-bess-dispatch` measured and found to be true for
its cost-only model.

It is **false here**, in two separate ways, and both were found by building the
pure LP first and checking the answers rather than reasoning about them:

**Negative prices invert the export discount.** On 2020-06-01 hour 15 the price
is −15.47 EUR/MWh. Importing therefore *earns* 15.47 and exporting *costs* 90%
of it, 13.92 — so importing and exporting the same megawatt-hour nets
**+1.55 EUR/MWh**, through a meter that physically cannot do both. The LP found
it immediately. This is the mirror image of the tariff bug project 01 caught:
there, export priced *above* import gamed the meter at positive prices; here,
export priced *below* import games it at negative ones. Guarding
`export_price_ratio <= 1` in `SiteConfig` closes only the first case.

**Three of the four objectives never mention the battery.**
`MAXIMIZE_SELF_CONSUMPTION` is `sum(pv_used) - sum(export)`; charge and
discharge do not appear in it, so cycling is free, and the solver returned a
schedule charging 0.250 MW and discharging 0.226 MW in the same hour. Not
profitable — merely unpenalised, which is enough. Degeneracy produces absurd
schedules just as reliably as a bad incentive does.

So the model carries two binaries per hour, gating grid direction and battery
direction. `enforce_exclusive_flows=False` rebuilds it as the pure LP, kept so
`tests/test_optimizer.py` can **demonstrate** both failures rather than assert
them — the same reason project 03 kept its broken radiality constraint.
"""

from __future__ import annotations

import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeReals,
    Objective,
    RangeSet,
    Var,
    maximize,
    minimize,
)

from energy_copilot.domain.config import SiteConfig
from energy_copilot.domain.ontology import Objective as ObjectiveKind
from energy_copilot.schemas.intent import OptimizationIntent

# One hour per period. The committed data is hourly and the horizon is a day.
HOURS_PER_PERIOD = 1.0


def build_model(
    intent: OptimizationIntent,
    site: SiteConfig,
    profile: pd.DataFrame,
    enforce_exclusive_flows: bool = True,
) -> ConcreteModel:
    """Assemble the model. Accepts only validated structures — never text.

    `enforce_exclusive_flows=False` drops the direction binaries and yields the
    pure LP. That mode is **wrong on this data** and is kept deliberately so the
    tests can show it failing; see the module docstring.
    """
    if not isinstance(intent, OptimizationIntent):
        # Belt and braces around the boundary this module exists to defend.
        raise TypeError(
            f"build_model takes an OptimizationIntent, got {type(intent).__name__}. "
            "Nothing reaches the solver without validating first."
        )

    hours = min(intent.horizon.hours, len(profile))
    profile = profile.iloc[:hours].reset_index(drop=True)

    price = profile["price_eur_mwh"].to_dict()
    load = profile["load_mw"].to_dict()
    pv_available = profile["pv_mw"].to_dict()
    carbon = profile["carbon_g_kwh"].to_dict()

    battery = site.battery
    constraints = intent.constraints

    model = ConcreteModel(name=f"copilot-{intent.objective.value}")
    model.T = RangeSet(0, hours - 1)

    # --- limits, tightened by the request where it is allowed to tighten ---
    import_limit = site.grid_import_limit_mw
    if constraints.max_grid_import_mw is not None:
        import_limit = min(import_limit, constraints.max_grid_import_mw)

    export_limit = site.grid_export_limit_mw
    if constraints.max_grid_export_mw is not None:
        export_limit = min(export_limit, constraints.max_grid_export_mw)
    if constraints.allow_export is False:
        export_limit = 0.0

    model.grid_import = Var(model.T, within=NonNegativeReals, bounds=(0, import_limit))
    model.grid_export = Var(model.T, within=NonNegativeReals, bounds=(0, export_limit))
    model.pv_used = Var(model.T, within=NonNegativeReals)

    # PV can be spilled unless the request forbids it. Forbidding it is a real
    # operating mode, and it can make an otherwise easy day infeasible, which
    # the caller is entitled to discover.
    allow_curtailment = constraints.allow_pv_curtailment is not False

    def pv_limit(m, t):
        if allow_curtailment:
            return m.pv_used[t] <= pv_available[t]
        return m.pv_used[t] == pv_available[t]

    model.pv_limit = Constraint(model.T, rule=pv_limit)

    # --- battery ---
    if battery is not None:
        soc_floor = battery.soc_min_frac
        soc_ceiling = battery.soc_max_frac
        # A request may tighten the window but never widen it: the equipment
        # governs, and this is where "the LLM cannot bypass physical limits"
        # stops being a claim and becomes an arithmetic fact.
        if constraints.minimum_soc_frac is not None:
            soc_floor = max(soc_floor, min(constraints.minimum_soc_frac, soc_ceiling))
        if constraints.maximum_soc_frac is not None:
            soc_ceiling = min(soc_ceiling, max(constraints.maximum_soc_frac, soc_floor))

        model.charge = Var(model.T, within=NonNegativeReals, bounds=(0, battery.power_mw))
        model.discharge = Var(model.T, within=NonNegativeReals, bounds=(0, battery.power_mw))
        model.soc = Var(
            model.T,
            within=NonNegativeReals,
            bounds=(soc_floor * battery.energy_mwh, soc_ceiling * battery.energy_mwh),
        )

        def soc_dynamics(m, t):
            previous = (
                battery.initial_soc_frac * battery.energy_mwh if t == 0 else m.soc[t - 1]
            )
            return m.soc[t] == previous + HOURS_PER_PERIOD * (
                battery.charge_efficiency * m.charge[t]
                - m.discharge[t] / battery.discharge_efficiency
            )

        model.soc_dynamics = Constraint(model.T, rule=soc_dynamics)

        if constraints.terminal_soc_frac is not None:
            target = constraints.terminal_soc_frac * battery.energy_mwh
            model.terminal_soc = Constraint(
                expr=model.soc[hours - 1] >= target - 1e-9
            )
    else:
        model.charge = Var(model.T, within=NonNegativeReals, bounds=(0, 0))
        model.discharge = Var(model.T, within=NonNegativeReals, bounds=(0, 0))
        model.soc = Var(model.T, within=NonNegativeReals, bounds=(0, 0))

    # --- flexible load ---
    # When shifting is not requested the block still runs -- on its default
    # earliest-first schedule, added to the fixed load. Dropping it would delete
    # real energy from the site and flatter every before/after comparison.
    shifting = intent.shift_flexible_loads and site.has_flexible_load
    flex_loads = site.flexible_loads if shifting else ()
    fixed_flex = [0.0] * hours
    if site.has_flexible_load and not shifting:
        for block in site.flexible_loads:
            for hour, power in enumerate(block.default_schedule(hours)):
                fixed_flex[hour] += power
    model.FLEX = RangeSet(0, max(len(flex_loads) - 1, 0))

    if flex_loads:
        def flex_bounds(m, f, t):
            block = flex_loads[f]
            inside = block.earliest_hour <= t <= block.latest_hour
            return (0, block.max_power_mw if inside else 0)

        model.flex = Var(model.FLEX, model.T, within=NonNegativeReals, bounds=flex_bounds)

        def flex_energy(m, f):
            return sum(m.flex[f, t] for t in m.T) * HOURS_PER_PERIOD == flex_loads[f].energy_mwh

        model.flex_energy = Constraint(model.FLEX, rule=flex_energy)
    else:
        model.flex = Var(model.FLEX, model.T, within=NonNegativeReals, bounds=(0, 0))

    def flex_total(m, t):
        return sum(m.flex[f, t] for f in m.FLEX) if flex_loads else 0.0

    # --- power balance, the constraint everything else hangs from ---
    def power_balance(m, t):
        supply = m.grid_import[t] + m.pv_used[t] + m.discharge[t]
        demand = (
            load[t] + fixed_flex[t] + m.charge[t] + m.grid_export[t] + flex_total(m, t)
        )
        return supply == demand

    model.power_balance = Constraint(model.T, rule=power_balance)

    # --- direction binaries -------------------------------------------------
    # One meter cannot import and export at once, and one battery cannot charge
    # and discharge at once. Both are enforced rather than assumed, because both
    # assumptions were measured and found false. See the module docstring.
    if enforce_exclusive_flows:
        model.grid_mode = Var(model.T, within=Binary)
        model.grid_import_mode = Constraint(
            model.T, rule=lambda m, t: m.grid_import[t] <= import_limit * m.grid_mode[t]
        )
        model.grid_export_mode = Constraint(
            model.T,
            rule=lambda m, t: m.grid_export[t] <= export_limit * (1 - m.grid_mode[t]),
        )
        if battery is not None:
            model.battery_mode = Var(model.T, within=Binary)
            model.charge_mode = Constraint(
                model.T,
                rule=lambda m, t: m.charge[t] <= battery.power_mw * m.battery_mode[t],
            )
            model.discharge_mode = Constraint(
                model.T,
                rule=lambda m, t: m.discharge[t]
                <= battery.power_mw * (1 - m.battery_mode[t]),
            )

    # --- peak ---
    model.peak = Var(within=NonNegativeReals)
    model.peak_definition = Constraint(
        model.T, rule=lambda m, t: m.peak >= m.grid_import[t]
    )
    if constraints.max_peak_mw is not None:
        model.peak_cap = Constraint(expr=model.peak <= constraints.max_peak_mw)

    # --- costed expressions, so every objective reads the same arithmetic ---
    degradation_rate = battery.degradation_cost_eur_mwh if battery is not None else 0.0

    model.energy_cost = Expression(
        expr=sum(
            price[t] * model.grid_import[t] * HOURS_PER_PERIOD
            - price[t] * site.export_price_ratio * model.grid_export[t] * HOURS_PER_PERIOD
            for t in model.T
        )
    )
    model.degradation_cost = Expression(
        expr=sum(
            degradation_rate * (model.charge[t] + model.discharge[t]) * HOURS_PER_PERIOD
            for t in model.T
        )
    )
    model.self_consumed = Expression(
        expr=sum((model.pv_used[t] - model.grid_export[t]) * HOURS_PER_PERIOD for t in model.T)
    )
    # g/kWh x MWh = kg. Stated because a factor of a thousand hides well here.
    model.emissions_kg = Expression(
        expr=sum(carbon[t] * model.grid_import[t] * HOURS_PER_PERIOD for t in model.T)
    )

    _attach_objective(model, intent.objective)

    # Everything downstream needs these and none of it should re-read a file.
    model._copilot = {
        "hours": hours,
        "price": price,
        "load": load,
        "pv_available": pv_available,
        "carbon": carbon,
        "objective": intent.objective.value,
        "import_limit": import_limit,
        "export_limit": export_limit,
        "has_battery": battery is not None,
        "flex_loads": tuple(block.name for block in flex_loads),
        "shifting": shifting,
        "fixed_flex": fixed_flex,
        "allow_curtailment": allow_curtailment,
        "exclusive_flows": enforce_exclusive_flows,
    }
    return model


def _attach_objective(model: ConcreteModel, kind: ObjectiveKind) -> None:
    """One objective, chosen from a closed set. No expression arrives as text."""
    if kind is ObjectiveKind.MINIMIZE_COST:
        model.objective = Objective(
            expr=model.energy_cost + model.degradation_cost, sense=minimize
        )
    elif kind is ObjectiveKind.MINIMIZE_PEAK:
        model.objective = Objective(expr=model.peak, sense=minimize)
    elif kind is ObjectiveKind.MAXIMIZE_SELF_CONSUMPTION:
        model.objective = Objective(expr=model.self_consumed, sense=maximize)
    elif kind is ObjectiveKind.MINIMIZE_EMISSIONS:
        model.objective = Objective(expr=model.emissions_kg, sense=minimize)
    else:  # pragma: no cover - the enum is closed
        raise ValueError(f"no objective implemented for {kind!r}")
