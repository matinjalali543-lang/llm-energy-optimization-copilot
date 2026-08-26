"""Explanations, and whether their numbers are real.

The check exists because a fabricated figure is formatted exactly like a true
one: "you saved 74 EUR" and "you saved 68.41 EUR" are equally authoritative on
the page, and only one of them came out of the optimizer.
"""

from __future__ import annotations

import pytest

from energy_copilot.domain.config import available_days
from energy_copilot.explain.factuality import (
    check_factuality,
    extract_numbers,
    fact_sheet,
    matches_any,
)
from energy_copilot.explain.render import describe, explain, explain_with_model
from energy_copilot.llm.provider import ScriptedProvider
from energy_copilot.pipeline import run_request

REQUESTS = (
    "Reduce electricity cost.",
    "Reduce the peak demand charge.",
    "Maximize PV self-consumption.",
    "Cut my carbon emissions.",
)


@pytest.fixture(scope="module")
def solved():
    return run_request("Reduce electricity cost tomorrow.", day="2020-09-21")


# --- the extractor ----------------------------------------------------------


def test_numbers_are_pulled_out_of_prose():
    found = extract_numbers("Cost fell from 370.54 EUR to 302.13 EUR, saving 68.41.")
    assert found == [370.54, 302.13, 68.41]


def test_thousands_separators_and_negatives_are_understood():
    assert extract_numbers("1,234.5 kg and -19.81 EUR") == [1234.5, -19.81]


# --- the matcher ------------------------------------------------------------


def test_a_number_matches_the_fact_it_was_rounded_from():
    assert matches_any(68.41, "68.41", [68.407252])
    assert matches_any(0.3, "0.3", [0.29999999])


def test_double_rounding_does_not_produce_a_false_accusation():
    """The bug this test exists for.

    `fact_sheet` stored 41.395; the renderer formatted the unrounded mean and
    printed 41.39; a checker that formatted the *stored* value got 41.40 and
    called a correct explanation a fabrication. Three of sixty explanations were
    wrongly flagged. Distance from the last place is the operation that actually
    asks "is this that number, rounded".
    """
    assert matches_any(41.39, "41.39", [41.395])
    assert matches_any(41.4, "41.4", [41.395])


def test_a_saving_may_be_written_without_its_sign():
    """"a credit of 19.81" for a cost of -19.81 is English, not invention."""
    assert matches_any(19.81, "19.81", [-19.81])


def test_a_genuinely_different_number_does_not_match():
    assert not matches_any(74.0, "74.0", [68.407252, 302.13])
    assert not matches_any(68.5, "68.5", [68.407252])


# --- the deterministic renderer is grounded, and that is pinned -------------


@pytest.mark.parametrize("request_text", REQUESTS)
def test_the_deterministic_renderer_invents_nothing(request_text):
    response = run_request(request_text, day="2020-06-01")
    text, report = explain(response)
    assert report.ok, f"{report.report()} in: {text}"
    assert report.score == 1.0


def test_the_renderer_is_grounded_on_every_committed_day():
    """60 runs. The claim "grounded by construction" was false when first
    written -- the renderer computed average charge prices that were in no fact
    sheet -- so it is checked rather than asserted."""
    for day in available_days():
        for request_text in REQUESTS:
            response = run_request(request_text, day=day)
            text, report = explain(response)
            assert report.ok, f"{day} / {request_text}: {report.report()} in {text}"


def test_the_explanation_says_something_about_the_battery(solved):
    text = describe(solved)
    assert "battery" in text.lower()
    assert "EUR" in text


def test_a_refusal_is_explained_rather_than_dressed_up():
    response = run_request("Run the diesel generator at 200 MW.")
    text = describe(response)
    assert "not run" in text
    assert "diesel" in text.lower() or "fossil" in text.lower()


def test_a_clarification_is_returned_as_a_question():
    response = run_request("Make the battery more aggressive.")
    text = describe(response)
    assert "?" in text
    assert "Options:" in text


# --- fabrication is caught --------------------------------------------------


@pytest.mark.parametrize(
    "fabricated",
    [
        "You saved 74.00 EUR on the day.",
        "The peak fell to 0.123 MW.",
        "That is a 99.5% improvement.",
        "Emissions dropped by 4321.9 kg.",
    ],
)
def test_invented_magnitudes_are_caught(solved, fabricated):
    report = check_factuality(fabricated, fact_sheet(solved))
    assert not report.ok
    assert report.score < 1.0


def test_a_model_that_makes_up_a_saving_is_caught(solved):
    """The whole point, through the real path."""
    provider = ScriptedProvider(
        responses=["Great news: this plan saves you 500.00 EUR against the baseline."]
    )
    text, report = explain_with_model(solved, provider)
    assert "500.00" in text
    assert not report.ok
    assert 500.0 in report.ungrounded


def test_a_model_quoting_real_figures_passes(solved):
    facts = fact_sheet(solved)
    honest = (
        f"The plan moves cost from {facts['before_cost_eur']:.2f} EUR to "
        f"{facts['after_cost_eur']:.2f} EUR."
    )
    provider = ScriptedProvider(responses=[honest])
    _, report = explain_with_model(solved, provider)
    assert report.ok


# --- the stated limits of the check ----------------------------------------


def test_small_integers_pass_because_hours_need_them(solved):
    """A documented hole, asserted so it stays documented.

    "cycled 12 times" is not caught, because 12 is inside the hour allowance.
    The check covers magnitudes and does not police counts below 25. Narrowing
    it with a nearby-word heuristic would trade a stated limitation for one that
    fails silently.
    """
    report = check_factuality("The battery cycled 12 times.", fact_sheet(solved))
    assert report.ok


def test_a_score_of_one_does_not_mean_true(solved):
    """Every figure correct, the causal claim nonsense. Nothing here reads
    meaning, and the docstring says so."""
    facts = fact_sheet(solved)
    misleading = (
        f"Because the moon was full, cost fell to {facts['after_cost_eur']:.2f} EUR."
    )
    assert check_factuality(misleading, facts).ok


def test_an_explanation_with_no_numbers_scores_one(solved):
    report = check_factuality("The battery was used sensibly.", fact_sheet(solved))
    assert report.ok
    assert report.checked == 0
    assert report.score == 1.0


# --- the fact sheet ---------------------------------------------------------


def test_the_fact_sheet_holds_what_the_renderer_needs(solved):
    facts = fact_sheet(solved)
    for key in (
        "before_cost_eur",
        "after_cost_eur",
        "cost_saving_eur",
        "avg_charge_price",
        "avg_discharge_price",
        "charge_hours",
        "price_spread",
    ):
        assert key in facts, key


def test_the_fact_sheet_is_empty_for_a_refusal():
    response = run_request("Run the diesel generator at 200 MW.")
    assert fact_sheet(response) == {}
