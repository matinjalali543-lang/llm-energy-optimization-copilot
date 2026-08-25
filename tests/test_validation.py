"""Validation: is this possible *at this site*, and does the refusal help?

§10 requires the validator to explain why a request was rejected. These tests
assert on the explanation, not just on the refusal, because a system that only
says "rejected" costs the user a turn every time and teaches them nothing.
"""

from __future__ import annotations

import pytest

from energy_copilot.domain.config import Battery, FlexibleLoad, SiteConfig
from energy_copilot.schemas.intent import OptimizationIntent
from energy_copilot.validation.validator import validate_intent


@pytest.fixture
def site():
    return SiteConfig()


def intent(**kwargs) -> OptimizationIntent:
    kwargs.setdefault("objective", "MINIMIZE_COST")
    return OptimizationIntent(**kwargs)


def codes(report) -> set[str]:
    return {rejection.code for rejection in report.rejections}


# --- the structural property behind the security claim ----------------------


def test_the_validator_takes_no_model_and_no_prompt():
    """The signature is the argument.

    `validate_intent` accepts an intent, a site and an optional data path. It
    has no parameter through which a prompt, a model name or a conversation
    could influence it, so its behaviour is identical whether the intent came
    from a careful user or from a model wholly under an attacker's control.
    """
    import inspect

    parameters = set(inspect.signature(validate_intent).parameters)
    assert parameters == {"intent", "site", "data_path"}


def test_identical_intents_validate_identically_regardless_of_origin(site):
    """No hidden state: the same intent twice gives the same verdict."""
    first = validate_intent(intent(constraints={"max_grid_import_mw": 200.0}), site)
    second = validate_intent(intent(constraints={"max_grid_import_mw": 200.0}), site)
    assert codes(first) == codes(second)
    assert [r.reason for r in first.rejections] == [r.reason for r in second.rejections]


# --- a good request passes --------------------------------------------------


def test_a_reasonable_request_is_accepted(site):
    report = validate_intent(
        intent(
            horizon={"label": "specific_day", "day": "2020-09-21"},
            constraints={"minimum_soc_frac": 0.2},
            assets=["battery", "pv", "grid"],
        ),
        site,
    )
    assert report.ok, report.report()
    assert report.report() == "Valid."


# --- the brief's own rejection examples (§10) -------------------------------


def test_an_import_limit_above_the_connection_is_refused_and_suggests_the_fix(site):
    """§10's "run the diesel generator at 200 MW" in the form this site can see.

    200 on a 0.6 MW connection is almost always a unit error, so the message
    says so instead of only reporting the bound.
    """
    report = validate_intent(intent(constraints={"max_grid_import_mw": 200.0}), site)
    assert not report.ok
    assert "above_connection_limit" in codes(report)
    reason = report.rejections[0].reason
    assert "0.6" in reason
    assert "kW" in reason


def test_a_closing_charge_above_the_ceiling_is_refused(site):
    report = validate_intent(intent(constraints={"terminal_soc_frac": 0.99}), site)
    assert "infeasible_soc" in codes(report)
    assert "95%" in report.rejections[0].reason


def test_a_floor_above_the_ceiling_is_refused(site):
    report = validate_intent(intent(constraints={"minimum_soc_frac": 0.99}), site)
    assert "infeasible_soc" in codes(report)


def test_an_unreachable_closing_charge_is_refused_with_the_arithmetic(site):
    """Meaningful, within the window, and still not reachable in the time."""
    small_power = SiteConfig(battery=Battery(energy_mwh=10.0, power_mw=0.05))
    report = validate_intent(
        intent(constraints={"terminal_soc_frac": 0.95}, horizon={"hours": 4}), small_power
    )
    assert "unreachable_soc" in codes(report)
    assert "MWh" in report.rejections[0].reason


def test_an_unknown_day_lists_the_days_that_exist(site):
    """Answering 'no data' without saying what there is wastes the next turn."""
    report = validate_intent(
        intent(horizon={"label": "specific_day", "day": "1999-01-01"}), site
    )
    assert "unknown_day" in codes(report)
    assert "2020-09-21" in report.rejections[0].reason


# --- absent assets ----------------------------------------------------------


def test_constraining_a_battery_that_is_not_there(site):
    no_battery = SiteConfig(battery=None)
    report = validate_intent(intent(constraints={"minimum_soc_frac": 0.2}), no_battery)
    assert "absent_asset" in codes(report)
    assert "no battery" in report.rejections[0].reason


def test_shifting_flexible_load_that_is_not_there(site):
    none_flex = SiteConfig(flexible_loads=())
    report = validate_intent(intent(shift_flexible_loads=True), none_flex)
    assert "absent_asset" in codes(report)


def test_asking_for_pv_at_a_site_without_it(site):
    no_pv = SiteConfig(pv_peak_mw=0.0)
    report = validate_intent(intent(assets=["pv"]), no_pv)
    assert "absent_asset" in codes(report)


# --- objectives that cannot be answered here --------------------------------


def test_self_consumption_without_pv_is_refused_rather_than_answered_with_zero():
    """The objective is well-defined and its answer is trivially zero, which
    would look like a working optimizer producing a useless schedule."""
    no_pv = SiteConfig(pv_peak_mw=0.0)
    report = validate_intent(intent(objective="MAXIMIZE_SELF_CONSUMPTION"), no_pv)
    assert "objective_not_answerable" in codes(report)
    assert "zero" in report.rejections[0].reason


def test_peak_reduction_with_nothing_to_move_it_is_refused():
    bare = SiteConfig(battery=None, flexible_loads=())
    report = validate_intent(intent(objective="MINIMIZE_PEAK"), bare)
    assert "objective_not_answerable" in codes(report)


def test_peak_reduction_with_only_flexible_load_warns_but_proceeds():
    flex_only = SiteConfig(battery=None, flexible_loads=(FlexibleLoad(),))
    report = validate_intent(intent(objective="MINIMIZE_PEAK"), flex_only)
    assert report.ok
    assert any("shifting flexible load" in w for w in report.warnings)


# --- warnings are not rejections --------------------------------------------


def test_a_floor_below_the_batterys_reserve_warns_rather_than_refuses(site):
    """The request is honourable; the equipment simply governs."""
    report = validate_intent(intent(constraints={"minimum_soc_frac": 0.05}), site)
    assert report.ok
    assert any("reserve" in w for w in report.warnings)


def test_a_flat_day_warns_that_the_saving_will_be_small(site):
    """2019-07-27 has a 6.69 EUR/MWh spread. Saying so beats presenting a
    rounding error as a triumph."""
    report = validate_intent(
        intent(horizon={"label": "specific_day", "day": "2019-07-27"}), site
    )
    assert report.ok
    assert any("6.69" in w for w in report.warnings)


def test_a_wide_day_does_not_warn(site):
    report = validate_intent(
        intent(horizon={"label": "specific_day", "day": "2020-09-21"}), site
    )
    assert report.ok
    assert not report.warnings


# --- every problem, not just the first --------------------------------------


def test_several_problems_are_all_reported(site):
    """Failing fast is right for a compiler and wrong for a conversation."""
    report = validate_intent(
        intent(
            horizon={"label": "specific_day", "day": "1999-01-01"},
            constraints={"max_grid_import_mw": 200.0, "terminal_soc_frac": 0.99},
        ),
        site,
    )
    assert len(report.rejections) >= 3
    assert {"unknown_day", "above_connection_limit", "infeasible_soc"} <= codes(report)


def test_every_rejection_carries_a_code_and_a_usable_reason(site):
    report = validate_intent(
        intent(constraints={"max_grid_import_mw": 500.0, "terminal_soc_frac": 0.99}), site
    )
    for rejection in report.rejections:
        assert rejection.code
        assert len(rejection.reason) > 20, rejection.reason
        assert rejection.offending_value is not None
