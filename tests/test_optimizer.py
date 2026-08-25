"""The deterministic core: physics, objectives, and the boundary it defends.

The most important test here is `test_the_pure_lp_really_does_produce_absurd_
schedules`. The binaries in `builder.py` are justified by a measurement, and a
justification whose counterexample has been deleted is one nobody can check — so
the broken mode is kept and demonstrated failing, the same way project 03 keeps
its insufficient radiality constraint.
"""

from __future__ import annotations

import pytest

from energy_copilot.domain.config import (
    Battery,
    SiteConfig,
    available_days,
    day_profile,
)
from energy_copilot.optimization.builder import build_model
from energy_copilot.optimization.solve import (
    InfeasibleRequest,
    check_no_simultaneous_flows,
    check_power_balance,
    solve_intent,
    solve_model,
)
from energy_copilot.schemas.intent import OptimizationIntent

OBJECTIVES = (
    "MINIMIZE_COST",
    "MINIMIZE_PEAK",
    "MAXIMIZE_SELF_CONSUMPTION",
    "MINIMIZE_EMISSIONS",
)


@pytest.fixture(scope="module")
def site():
    return SiteConfig()


def intent(objective="MINIMIZE_COST", **kwargs) -> OptimizationIntent:
    kwargs.setdefault("shift_flexible_loads", True)
    return OptimizationIntent(objective=objective, **kwargs)


# --- the boundary -----------------------------------------------------------


def test_the_builder_refuses_anything_that_is_not_a_validated_intent(site):
    """Text cannot reach the solver, and the type check says so out loud."""
    profile = day_profile("2020-09-21")
    with pytest.raises(TypeError, match="without validating first"):
        build_model({"objective": "MINIMIZE_COST"}, site, profile)
    with pytest.raises(TypeError):
        build_model("minimise my costs please", site, profile)


def test_a_request_cannot_widen_the_batterys_own_window(site):
    """The equipment governs. This is where "cannot bypass physical limits"
    stops being a claim and becomes arithmetic."""
    profile = day_profile("2020-09-21")
    # Ask for a floor below, and a ceiling above, what the battery allows.
    result = solve_intent(
        intent(constraints={"minimum_soc_frac": 0.0, "maximum_soc_frac": 1.0}), site, profile
    )
    battery = site.battery
    assert result.schedule["soc_mwh"].min() >= battery.soc_min_frac * battery.energy_mwh - 1e-6
    assert result.schedule["soc_mwh"].max() <= battery.soc_max_frac * battery.energy_mwh + 1e-6


# --- physics ----------------------------------------------------------------


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_power_balances_every_hour(site, objective):
    result = solve_intent(intent(objective), site, day_profile("2020-09-21"))
    assert check_power_balance(result) == []


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_no_simultaneous_flows_under_any_objective(site, objective):
    result = solve_intent(intent(objective), site, day_profile("2020-06-01"))
    assert check_no_simultaneous_flows(result) == []


def test_state_of_charge_follows_its_recursion(site):
    result = solve_intent(intent(), site, day_profile("2020-09-21"))
    battery = site.battery
    schedule = result.schedule
    previous = battery.initial_soc_frac * battery.energy_mwh
    for _, row in schedule.iterrows():
        expected = previous + (
            battery.charge_efficiency * row["charge_mw"]
            - row["discharge_mw"] / battery.discharge_efficiency
        )
        assert row["soc_mwh"] == pytest.approx(expected, abs=1e-6)
        previous = row["soc_mwh"]


def test_flows_respect_their_ratings(site):
    result = solve_intent(intent(), site, day_profile("2020-09-21"))
    schedule = result.schedule
    assert schedule["grid_import_mw"].max() <= site.grid_import_limit_mw + 1e-6
    assert schedule["grid_export_mw"].max() <= site.grid_export_limit_mw + 1e-6
    assert schedule["charge_mw"].max() <= site.battery.power_mw + 1e-6
    assert schedule["discharge_mw"].max() <= site.battery.power_mw + 1e-6
    assert (schedule["pv_used_mw"] <= schedule["pv_available_mw"] + 1e-6).all()


def test_the_flexible_block_consumes_its_energy_wherever_it_lands(site):
    """Shifting moves the block; it does not delete it."""
    profile = day_profile("2020-09-21")
    block = site.flexible_loads[0]
    shifted = solve_intent(intent(shift_flexible_loads=True), site, profile)
    fixed = solve_intent(intent(shift_flexible_loads=False), site, profile)

    assert shifted.schedule["flex_total_mw"].sum() == pytest.approx(block.energy_mwh, abs=1e-6)
    assert fixed.schedule["fixed_flex_mw"].sum() == pytest.approx(block.energy_mwh, abs=1e-6)
    # Same energy served either way, or the comparison between them is a lie.
    assert shifted.schedule["total_demand_mw"].sum() == pytest.approx(
        fixed.schedule["total_demand_mw"].sum(), abs=1e-6
    )


def test_the_flexible_block_stays_inside_its_window():
    from energy_copilot.domain.config import FlexibleLoad

    narrow = SiteConfig(
        flexible_loads=(FlexibleLoad(name="night", energy_mwh=0.3, earliest_hour=0, latest_hour=5),)
    )
    result = solve_intent(intent(shift_flexible_loads=True), narrow, day_profile("2020-09-21"))
    outside = result.schedule.loc[result.schedule["hour"] > 5, "flex_total_mw"]
    assert outside.abs().max() < 1e-6


# --- the objectives are four different things -------------------------------


def test_each_objective_wins_on_its_own_metric(site):
    """Otherwise an objective is decorative.

    This caught a real defect. With PV at 0.30 MWp the array never once exceeded
    demand, so every kWh of sun was self-consumed whatever the optimizer did,
    MAXIMIZE_SELF_CONSUMPTION was a constant, and its "optimum" was an arbitrary
    vertex that happened to post the worst cost of the four.
    """
    profile = day_profile("2020-06-01")
    results = {name: solve_intent(intent(name), site, profile) for name in OBJECTIVES}

    assert min(results, key=lambda k: results[k].total_cost_eur) == "MINIMIZE_COST"
    assert min(results, key=lambda k: results[k].peak_mw) == "MINIMIZE_PEAK"
    assert (
        max(results, key=lambda k: results[k].self_consumed_mwh)
        == "MAXIMIZE_SELF_CONSUMPTION"
    )
    assert min(results, key=lambda k: results[k].emissions_kg) == "MINIMIZE_EMISSIONS"


def test_the_objectives_produce_visibly_different_schedules(site):
    profile = day_profile("2020-06-01")
    schedules = {
        name: solve_intent(intent(name), site, profile).schedule["grid_import_mw"]
        for name in OBJECTIVES
    }
    for first in OBJECTIVES:
        for second in OBJECTIVES:
            if first < second:
                gap = (schedules[first] - schedules[second]).abs().max()
                assert gap > 0.01, f"{first} and {second} produced the same schedule"


# --- the measurement behind the binaries ------------------------------------


def test_the_pure_lp_really_does_produce_absurd_schedules(site):
    """The counterexample, kept so the binaries stay justified.

    If this test ever passes cleanly, the LP is safe after all and the binaries
    can go. It does not pass: on 2020-06-01 the price is negative in hours
    13-15, which makes importing and exporting the same megawatt-hour profitable
    through a meter that cannot do both.
    """
    profile = day_profile("2020-06-01")
    problems: list[str] = []
    for objective in OBJECTIVES:
        relaxed = build_model(
            intent(objective), site, profile, enforce_exclusive_flows=False
        )
        result = solve_model(relaxed, intent=intent(objective))
        problems.extend(check_no_simultaneous_flows(result))

    assert problems, (
        "the pure LP produced no absurd schedule, so the direction binaries are "
        "no longer earning their solve time and builder.py's justification is stale"
    )
    assert any("imports and exports" in problem for problem in problems)
    assert any("charges and discharges" in problem for problem in problems)


def test_the_binaries_fix_exactly_that(site):
    profile = day_profile("2020-06-01")
    for objective in OBJECTIVES:
        result = solve_intent(intent(objective), site, profile)
        assert check_no_simultaneous_flows(result) == []


# --- every committed day solves ---------------------------------------------


@pytest.mark.parametrize("day", available_days())
def test_every_representative_day_solves_and_conserves_power(site, day):
    result = solve_intent(intent(), site, day_profile(day))
    assert check_power_balance(result) == []
    assert check_no_simultaneous_flows(result) == []
    assert result.termination == "optimal"


# --- reproducibility (§25) --------------------------------------------------


def test_the_same_intent_gives_the_same_answer(site):
    """The numerical result must be reproducible with no LLM anywhere near it."""
    profile = day_profile("2020-09-21")
    first = solve_intent(intent(), site, profile)
    second = solve_intent(intent(), site, profile)
    assert first.total_cost_eur == pytest.approx(second.total_cost_eur, abs=1e-9)
    assert first.schedule.equals(second.schedule)


# --- infeasibility carries an explanation -----------------------------------


def test_an_impossible_request_raises_something_a_person_can_read(site):
    """Forbidding curtailment on a sunny day with export capped at zero leaves
    the surplus nowhere to go."""
    profile = day_profile("2020-06-01")
    impossible = intent(
        constraints={"allow_pv_curtailment": False, "max_grid_export_mw": 0.0},
    )
    with pytest.raises(InfeasibleRequest) as caught:
        solve_intent(impossible, site, profile)
    assert "could not be satisfied" in str(caught.value)
    assert caught.value.intent is impossible


def test_a_tiny_battery_still_solves():
    small = SiteConfig(battery=Battery(energy_mwh=0.05, power_mw=0.02))
    result = solve_intent(intent(), small, day_profile("2020-09-21"))
    assert check_power_balance(result) == []
