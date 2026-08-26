"""Multi-turn state and scenario comparison.

§15 asks for structured state kept separate from conversation text. These tests
check the property that instruction is actually about: a later turn adds to what
was established rather than starting again, and nothing is rediscovered by
re-reading the transcript.
"""

from __future__ import annotations

import pytest

from energy_copilot.comparison import (
    BASE_LABEL,
    apply_variant,
    comparison_frame,
    describe_comparison,
    run_comparison,
)
from energy_copilot.conversation import Conversation, revise
from energy_copilot.domain.config import SiteConfig
from energy_copilot.parsing.rule_based import parse as rule_based_parse
from energy_copilot.schemas.intent import (
    Clarification,
    Objective,
    OptimizationIntent,
    ScenarioVariant,
)

DAY = "2020-09-21"


@pytest.fixture
def chat():
    return Conversation(day=DAY)


# --- the brief's own §15 example --------------------------------------------


def test_a_later_turn_adds_to_what_was_established(chat):
    """"Optimize tomorrow's cost" then "keep 20% in the battery" is ONE intent.

    The second message names no objective. Parsed on its own it produces "which
    objective should govern?" — a question the conversation already knows the
    answer to, and asking it is the failure §15 exists to prevent.
    """
    first = chat.send("Optimize tomorrow's electricity cost.")
    assert first.ok
    assert chat.intent.constraints.minimum_soc_frac is None

    second = chat.send("Keep at least 20% in the battery.")
    assert second.ok, second.summary()
    assert chat.intent.objective is Objective.MINIMIZE_COST      # kept
    assert chat.intent.constraints.minimum_soc_frac == pytest.approx(0.2)  # added


def test_restating_the_goal_replaces_rather_than_merges(chat):
    chat.send("Reduce electricity cost tomorrow.")
    chat.send("Reduce the peak demand charge.")
    assert chat.intent.objective is Objective.MINIMIZE_PEAK


def test_a_refused_turn_does_not_erase_the_conversation(chat):
    chat.send("Reduce electricity cost tomorrow.")
    established = chat.intent
    chat.send("Run the diesel generator at 200 MW.")
    assert chat.intent == established


# --- clarification ----------------------------------------------------------


def test_an_ambiguous_request_asks_and_waits(chat):
    response = chat.send("Make the battery more aggressive.")
    assert isinstance(response.outcome, Clarification)
    assert chat.pending is not None
    assert not response.reached_solver


def test_an_option_answer_changes_a_field_not_a_sentence(chat):
    """The first version re-parsed a synthetic sentence and asked again.

    Round-tripping structured state through prose is precisely what §15 warns
    against, and it had been written into the module that quotes §15.
    """
    chat.send("Reduce electricity cost tomorrow.")
    chat.send("Make the battery more aggressive.")
    response = chat.send("prioritise peak reduction over cost")

    assert response.ok, response.summary()
    assert chat.pending is None
    assert chat.intent.objective is Objective.MINIMIZE_PEAK
    # An enum, not a string: `revise` re-validates rather than copying blindly.
    assert isinstance(chat.intent.objective, Objective)


def test_an_option_can_be_chosen_by_number(chat):
    chat.send("Reduce electricity cost tomorrow.")
    chat.send("Make the battery more aggressive.")
    response = chat.send("1")
    assert response.ok
    assert chat.intent.constraints.minimum_soc_frac is not None


def test_an_unrecognisable_answer_becomes_a_new_request(chat):
    """Re-asking the same question is what makes these systems infuriating."""
    chat.send("Make the battery more aggressive.")
    response = chat.send("Reduce the peak demand charge.")
    assert response.ok
    assert chat.pending is None
    assert chat.intent.objective is Objective.MINIMIZE_PEAK


def test_every_generated_option_is_answerable():
    """The two lists must not drift apart.

    `rule_based.py` generates the options; `conversation.py` maps them to field
    changes. If a new option is added to one and not the other, the assistant
    offers a choice it cannot act on.
    """
    from energy_copilot.conversation import _option_update
    from energy_copilot.parsing.rule_based import AMBIGUOUS_PATTERNS

    site = SiteConfig()
    for _pattern, _question, options in AMBIGUOUS_PATTERNS:
        for option in options:
            assert _option_update(option, site) is not None, option


# --- revise() re-validates --------------------------------------------------


def test_revise_returns_a_validated_intent():
    intent = OptimizationIntent(objective="MINIMIZE_COST")
    revised = revise(intent, objective="MINIMIZE_PEAK")
    assert isinstance(revised.objective, Objective)
    assert revised.objective is Objective.MINIMIZE_PEAK


def test_revise_refuses_an_impossible_update():
    """`model_copy(update=...)` would have accepted this silently."""
    from pydantic import ValidationError

    intent = OptimizationIntent(objective="MINIMIZE_COST")
    with pytest.raises(ValidationError):
        revise(intent, **{"constraint:minimum_soc_frac": 1.5})


def test_revise_leaves_the_original_alone():
    intent = OptimizationIntent(objective="MINIMIZE_COST")
    revise(intent, objective="MINIMIZE_PEAK")
    assert intent.objective is Objective.MINIMIZE_COST


# --- scenario comparison (§17) ---------------------------------------------


def test_comparing_two_battery_sizes_solves_both_and_the_base():
    intent = rule_based_parse("Compare a 1 MWh battery with a 2 MWh battery.")
    assert isinstance(intent, OptimizationIntent)
    outcomes = run_comparison(intent, SiteConfig(), None)

    assert len(outcomes) == 3
    assert outcomes[0].label == BASE_LABEL
    assert all(outcome.ok for outcome in outcomes)


def test_a_bigger_battery_is_worth_more_on_a_wide_spread_day():
    intent = OptimizationIntent(
        objective="MINIMIZE_COST",
        horizon={"label": "specific_day", "day": DAY},
        comparison=[
            {"label": "1 MWh", "battery_energy_mwh": 1.0},
            {"label": "2 MWh", "battery_energy_mwh": 2.0},
        ],
    )
    outcomes = {o.label: o for o in run_comparison(intent)}
    assert outcomes["2 MWh"].kpis.cost_eur < outcomes["1 MWh"].kpis.cost_eur
    assert outcomes["1 MWh"].kpis.cost_eur < outcomes[BASE_LABEL].kpis.cost_eur


def test_the_base_case_is_always_included():
    """Two hypotheticals with no reference point invite the reader to assume the
    better one beats what they have. It might be worse than both."""
    intent = OptimizationIntent(
        objective="MINIMIZE_COST",
        comparison=[{"label": "tiny", "battery_energy_mwh": 0.01}],
    )
    frame = comparison_frame(run_comparison(intent))
    assert BASE_LABEL in set(frame["scenario"])
    base_row = frame[frame["scenario"] == BASE_LABEL].iloc[0]
    tiny_row = frame[frame["scenario"] == "tiny"].iloc[0]
    # And here the variant IS worse, which is the case worth showing.
    assert tiny_row["cost_eur"] > base_row["cost_eur"]


def test_a_variant_never_mutates_the_base_site():
    site = SiteConfig()
    variant = ScenarioVariant(label="big", battery_energy_mwh=9.0)
    altered = apply_variant(site, variant)
    assert altered.battery.energy_mwh == 9.0
    assert site.battery.energy_mwh == 0.5


def test_one_impossible_variant_does_not_discard_the_others():
    intent = OptimizationIntent(
        objective="MINIMIZE_COST",
        comparison=[{"label": "big", "battery_energy_mwh": 4.0}],
    )
    outcomes = run_comparison(intent, SiteConfig(battery=None))
    assert outcomes[0].ok                       # the base still solved
    assert not outcomes[1].ok
    assert "none to change" in outcomes[1].failure


def test_a_negligible_difference_is_reported_as_no_difference():
    """Presenting a 0.004 EUR gap as a winner is how comparisons stop being
    useful."""
    intent = OptimizationIntent(
        objective="MINIMIZE_COST",
        horizon={"label": "specific_day", "day": "2019-07-27"},
        comparison=[
            {"label": "a", "battery_energy_mwh": 0.001, "battery_power_mw": 0.001},
            {"label": "b", "battery_energy_mwh": 0.002, "battery_power_mw": 0.001},
        ],
    )
    outcomes = [o for o in run_comparison(intent) if o.label in ("a", "b")]
    text = describe_comparison(outcomes)
    assert "same" in text or "cheapest" in text


def test_the_comparison_frame_names_the_margin_against_the_base():
    intent = OptimizationIntent(
        objective="MINIMIZE_COST",
        horizon={"label": "specific_day", "day": DAY},
        comparison=[{"label": "2 MWh", "battery_energy_mwh": 2.0}],
    )
    frame = comparison_frame(run_comparison(intent))
    assert "vs_base_eur" in frame.columns
    row = frame[frame["scenario"] == "2 MWh"].iloc[0]
    assert row["vs_base_eur"] > 0


# --- the transcript is for display only -------------------------------------


def test_the_transcript_is_not_where_state_lives(chat):
    chat.send("Optimize tomorrow's electricity cost.")
    chat.send("Keep at least 20% in the battery.")
    assert "user:" in chat.transcript()
    # Wiping the transcript must not change the structured state.
    established = chat.intent
    chat.turns.clear()
    assert chat.intent == established


def test_reset_clears_everything(chat):
    chat.send("Optimize tomorrow's electricity cost.")
    chat.reset()
    assert chat.intent is None
    assert chat.pending is None
    assert chat.turns == []
