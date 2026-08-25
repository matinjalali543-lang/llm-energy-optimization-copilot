"""KPIs: arithmetic that the LLM is never allowed to do.

§14 says to calculate every number in Python. These tests check the arithmetic
against values worked out by hand, because the whole point of moving it out of
the model is that it can then be checked.
"""

from __future__ import annotations

import pandas as pd
import pytest

from energy_copilot.domain.config import Battery, FlexibleLoad, SiteConfig, day_profile
from energy_copilot.kpis import (
    KPIs,
    baseline_kpis,
    check_demand_is_conserved,
    comparison_table,
    improvement,
    result_kpis,
)
from energy_copilot.optimization.solve import solve_intent
from energy_copilot.schemas.intent import OptimizationIntent


@pytest.fixture(scope="module")
def site():
    return SiteConfig()


def flat_profile(hours=4, price=50.0, load=0.2, pv=0.0, carbon=400.0) -> pd.DataFrame:
    """A day simple enough to compute on paper."""
    return pd.DataFrame(
        {
            "hour": range(hours),
            "price_eur_mwh": [price] * hours,
            "load_mw": [load] * hours,
            "pv_mw": [pv] * hours,
            "carbon_g_kwh": [carbon] * hours,
        }
    )


# --- the baseline, by hand --------------------------------------------------


def test_the_baseline_is_the_arithmetic_you_would_do_yourself():
    """No PV, no flexible load, flat price: cost is price x energy."""
    bare = SiteConfig(pv_peak_mw=0.0, battery=None, flexible_loads=())
    kpis = baseline_kpis(bare, flat_profile(hours=4, price=50.0, load=0.2))

    assert kpis.grid_import_mwh == pytest.approx(0.8)      # 4 h x 0.2 MW
    assert kpis.cost_eur == pytest.approx(40.0)            # 0.8 MWh x 50
    assert kpis.peak_mw == pytest.approx(0.2)
    assert kpis.emissions_kg == pytest.approx(320.0)       # 0.8 x 400
    assert kpis.battery_throughput_mwh == 0.0


def test_pv_offsets_load_before_anything_is_imported():
    bare = SiteConfig(pv_peak_mw=1.0, battery=None, flexible_loads=())
    kpis = baseline_kpis(bare, flat_profile(hours=4, price=50.0, load=0.2, pv=0.05))

    assert kpis.grid_import_mwh == pytest.approx(0.6)      # (0.2 - 0.05) x 4
    assert kpis.self_consumed_mwh == pytest.approx(0.2)    # 0.05 x 4
    assert kpis.cost_eur == pytest.approx(30.0)


def test_surplus_beyond_the_export_limit_is_curtailed():
    tight = SiteConfig(
        pv_peak_mw=1.0, battery=None, flexible_loads=(), grid_export_limit_mw=0.1
    )
    # 0.5 MW of sun against 0.2 MW of load: 0.3 MW surplus, only 0.1 exportable.
    kpis = baseline_kpis(tight, flat_profile(hours=2, load=0.2, pv=0.5))

    assert kpis.grid_export_mwh == pytest.approx(0.2)      # 0.1 x 2
    assert kpis.pv_curtailed_mwh == pytest.approx(0.4)     # 0.2 x 2
    assert kpis.grid_import_mwh == pytest.approx(0.0)


def test_the_flexible_block_is_in_the_baseline_too():
    """It runs earliest-first when nobody is optimizing it — and it still runs."""
    with_flex = SiteConfig(
        pv_peak_mw=0.0,
        battery=None,
        flexible_loads=(FlexibleLoad(name="b", energy_mwh=0.2, max_power_mw=0.1),),
    )
    kpis = baseline_kpis(with_flex, flat_profile(hours=4, price=50.0, load=0.2))
    # 0.8 MWh of fixed load plus 0.2 MWh of block.
    assert kpis.demand_mwh == pytest.approx(1.0)
    assert kpis.grid_import_mwh == pytest.approx(1.0)


# --- improvement ------------------------------------------------------------


def test_improvement_reports_absolute_and_percentage():
    before = KPIs(100.0, 0.5, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 500.0, 2.0)
    after = KPIs(75.0, 0.4, 1.8, 0.0, 0.0, 1.0, 1.0, 0.5, 450.0, 2.0)
    delta = improvement(before, after)

    assert delta["cost_saving_eur"] == pytest.approx(25.0)
    assert delta["cost_saving_pct"] == pytest.approx(25.0)
    assert delta["peak_reduction_mw"] == pytest.approx(0.1)
    assert delta["peak_reduction_pct"] == pytest.approx(20.0)
    assert delta["emissions_saving_kg"] == pytest.approx(50.0)


def test_no_percentage_is_invented_when_the_baseline_is_zero():
    """Dividing by a zero baseline produces infinity, which reads as a triumph."""
    before = KPIs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    after = KPIs(-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    delta = improvement(before, after)

    assert "cost_saving_pct" not in delta
    assert "peak_reduction_pct" not in delta
    assert delta["cost_saving_eur"] == pytest.approx(5.0)


def test_a_bill_that_becomes_a_credit_is_flagged_rather_than_left_as_131_percent():
    """A real case: 2020-06-01 goes from +63.80 EUR to -19.81 EUR.

    The formula says "131% saving", which is arithmetically right and close to
    meaningless as prose. The flag lets the explanation say what happened.
    """
    before = KPIs(63.80, 0.5, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 500.0, 2.0)
    after = KPIs(-19.81, 0.5, 2.0, 0.0, 0.0, 1.0, 1.0, 0.5, 500.0, 2.0)
    delta = improvement(before, after)

    assert delta["cost_saving_pct"] > 100
    assert delta.get("cost_became_credit") == 1.0


# --- the guard against the most flattering possible bug ---------------------


def test_serving_less_load_is_not_a_saving():
    before = KPIs(100.0, 0.5, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 500.0, 2.0)
    cheating = KPIs(10.0, 0.1, 0.2, 0.0, 0.0, 1.0, 1.0, 0.0, 50.0, 0.2)
    problems = check_demand_is_conserved(before, cheating)

    assert problems
    assert "serving less load is not a saving" in problems[0]


def test_a_real_optimization_conserves_demand(site):
    profile = day_profile("2020-09-21")
    before = baseline_kpis(site, profile)
    after = result_kpis(
        solve_intent(
            OptimizationIntent(objective="MINIMIZE_COST", shift_flexible_loads=True),
            site,
            profile,
        )
    )
    assert check_demand_is_conserved(before, after) == []


# --- the optimizer must beat doing nothing ----------------------------------


def test_optimizing_cost_beats_the_do_nothing_baseline(site):
    """On the widest-spread day, or the optimizer is not earning its keep."""
    profile = day_profile("2020-09-21")
    before = baseline_kpis(site, profile)
    after = result_kpis(
        solve_intent(
            OptimizationIntent(objective="MINIMIZE_COST", shift_flexible_loads=True),
            site,
            profile,
        )
    )
    assert after.cost_eur < before.cost_eur


def test_the_flat_day_saves_much_less_than_the_wide_one(site):
    """The dataset spans regimes on purpose; this is that showing up."""
    savings = {}
    for day in ("2020-09-21", "2019-07-27"):
        profile = day_profile(day)
        before = baseline_kpis(site, profile)
        after = result_kpis(
            solve_intent(
                OptimizationIntent(objective="MINIMIZE_COST", shift_flexible_loads=True),
                site,
                profile,
            )
        )
        savings[day] = improvement(before, after)["cost_saving_eur"]

    assert savings["2020-09-21"] > savings["2019-07-27"] * 3


# --- presentation -----------------------------------------------------------


def test_the_comparison_table_has_a_row_per_metric():
    before = KPIs(100.0, 0.5, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 500.0, 2.0)
    after = KPIs(75.0, 0.4, 1.8, 0.0, 0.0, 1.0, 1.0, 0.5, 450.0, 2.0)
    table = comparison_table(before, after)

    assert list(table.columns) == ["metric", "before", "after", "change"]
    assert len(table) == 8
    row = table[table["metric"] == "Cost (EUR)"].iloc[0]
    assert row["change"] == pytest.approx(-25.0)


def test_battery_throughput_counts_both_directions(site):
    profile = day_profile("2020-09-21")
    result = solve_intent(
        OptimizationIntent(objective="MINIMIZE_COST", shift_flexible_loads=True),
        site,
        profile,
    )
    expected = result.schedule["charge_mw"].sum() + result.schedule["discharge_mw"].sum()
    assert result_kpis(result).battery_throughput_mwh == pytest.approx(expected)


def test_a_battery_that_cannot_move_saves_nothing_extra():
    """Sanity floor: with a negligible battery the answer approaches the baseline."""
    tiny = SiteConfig(battery=Battery(energy_mwh=0.001, power_mw=0.001), flexible_loads=())
    profile = day_profile("2019-07-27")
    before = baseline_kpis(tiny, profile)
    after = result_kpis(
        solve_intent(OptimizationIntent(objective="MINIMIZE_COST"), tiny, profile)
    )
    assert after.cost_eur == pytest.approx(before.cost_eur, abs=1.0)
