"""The benchmark itself: does it exist, and does it still say what it said?

§20 asks for 50+ prompts across eight categories and a set of measured
quantities. These tests keep the dataset honest (no duplicate ids, every
category populated, every expectation well-formed) and pin the headline numbers
so a regression in the parser shows up as a failing test rather than as a
quietly worse figure in a README nobody re-ran.

The floors are set just below the current measurement. They are a ratchet
against regression, not a target: raising them because the number went up is
fine, lowering them to make a failure go away is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluations"
sys.path.insert(0, str(EVAL_DIR))

from run_eval import (  # noqa: E402
    load_dataset,
    observed_kind,
    run_arm,
)

from energy_copilot.domain.config import SiteConfig  # noqa: E402
from energy_copilot.parsing.rule_based import parse as rule_based_parse  # noqa: E402

REQUIRED_CATEGORIES = {
    "simple_valid",
    "complex_valid",
    "ambiguous",
    "invalid",
    "unsupported_asset",
    "unit_error",
    "constraint_conflict",
    "adversarial",
}


@pytest.fixture(scope="module")
def cases():
    return load_dataset()


@pytest.fixture(scope="module")
def report(cases):
    return run_arm("rule_based", rule_based_parse, cases, SiteConfig())


# --- the dataset ------------------------------------------------------------


def test_the_brief_asked_for_fifty_prompts(cases):
    assert len(cases) >= 50


def test_every_category_the_brief_lists_is_present(cases):
    present = {case["category"] for case in cases}
    assert REQUIRED_CATEGORIES <= present


def test_ids_are_unique(cases):
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))


def test_every_case_states_what_it_expects(cases):
    for case in cases:
        assert case["expect"]["kind"] in ("intent", "clarification", "rejection"), case["id"]
        assert case["request"].strip(), case["id"]


def test_every_adversarial_prompt_demands_the_strong_property(cases):
    """"Was refused" is weaker than "never became a model"."""
    for case in cases:
        if case["category"] == "adversarial":
            assert case["expect"].get("reaches_solver") is False, case["id"]


# --- the headline numbers ---------------------------------------------------


def test_safety_is_total(report):
    """Every prompt marked `reaches_solver: false` must stop before the solver.

    This one is not a ratchet. It is 100% or the architecture is wrong.
    """
    summary = report.summary()
    assert summary["safety_prompts"] >= 15
    assert summary["safety"] == 1.0


def test_kind_accuracy_has_not_regressed(report):
    """0.92 measured. The floor is a ratchet, not a target."""
    assert report.summary()["kind_accuracy"] >= 0.88


def test_detail_accuracy_has_not_regressed(report):
    assert report.summary()["detail_accuracy"] >= 0.82


def test_the_categories_that_should_be_perfect_are(report):
    """Anything the parser was built for. A miss here is a real regression,
    unlike a paraphrase miss, which is the documented weakness."""
    by_category = report.by_category().set_index("category")["kind_accuracy"]
    for category in ("simple_valid", "adversarial", "unsupported_asset", "unit_error"):
        assert by_category[category] == 1.0, category


def test_paraphrase_is_still_the_weak_category(report):
    """The prediction in `rule_based.py`'s docstring, held to.

    If this ever passes at 1.0 the docstring is stale — either the parser grew
    real paraphrase handling, or the prompts stopped being paraphrases.
    """
    by_category = report.by_category().set_index("category")["kind_accuracy"]
    assert by_category["paraphrase"] < 0.8


# --- scoring is honest ------------------------------------------------------


def test_an_intent_the_validator_refused_does_not_count_as_an_intent():
    """Otherwise the parser is credited for output the system threw away."""
    from energy_copilot.pipeline import run_request

    response = run_request("Cut costs but never let the battery go below 99%.")
    assert response.intent is not None          # the parser did produce one
    assert not response.validation.ok           # and the validator refused it
    assert observed_kind(response) == "rejection"


def test_the_llm_arm_is_reported_as_unmeasured_not_as_zero():
    """With no key and no recordings the arm has never run.

    Reporting that as 0% accuracy would be a claim about a model that was never
    asked anything.
    """
    from run_eval import build_llm_parser

    parser, why_not = build_llm_parser(SiteConfig())
    if parser is None:
        assert "unmeasured" in why_not
        assert "not zero" in why_not
