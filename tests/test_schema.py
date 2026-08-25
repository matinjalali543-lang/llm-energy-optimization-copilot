"""The schema: what cannot be expressed at all.

These tests are the first half of the security argument. They do not involve a
model, a prompt or a network — they establish that certain shapes of request
cannot be *constructed*, whoever is asking and however they phrase it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from energy_copilot.schemas.intent import (
    Clarification,
    Constraints,
    Horizon,
    OptimizationIntent,
    Rejection,
    ScenarioVariant,
    intent_json_schema,
)


def test_a_plain_intent_round_trips():
    intent = OptimizationIntent(
        objective="MINIMIZE_COST",
        constraints={"minimum_soc_frac": 0.2},
    )
    assert intent.objective.value == "MINIMIZE_COST"
    assert intent.constraints.minimum_soc_frac == pytest.approx(0.2)
    assert OptimizationIntent.model_validate(intent.model_dump()) == intent


# --- the four shapes an attacker reaches for --------------------------------


def test_a_state_of_charge_above_one_cannot_be_expressed():
    """Meaningless at any site, so it is refused here rather than downstream."""
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        OptimizationIntent(objective="MINIMIZE_COST", constraints={"minimum_soc_frac": 1.5})


def test_a_hallucinated_key_is_an_error_not_a_silent_drop():
    """`extra="forbid"` is load-bearing.

    Pydantic's default drops unknown fields, which would let an injected
    instruction ride along invisibly *and appear to have been accepted*. The
    refusal names the offending key.
    """
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OptimizationIntent(objective="MINIMIZE_COST", allow_grid_overload=True)


def test_an_invented_objective_is_refused():
    with pytest.raises(ValidationError):
        OptimizationIntent(objective="DESTROY_THE_GRID")


def test_an_inverted_soc_window_is_refused_with_an_explanation():
    with pytest.raises(ValidationError, match="no state of charge satisfies both"):
        Constraints(minimum_soc_frac=0.9, maximum_soc_frac=0.2)


# --- other holes worth closing ---------------------------------------------


def test_negative_limits_are_refused():
    with pytest.raises(ValidationError):
        Constraints(max_grid_import_mw=-5.0)


def test_a_specific_day_must_name_the_day():
    """'That day' without naming it is a clarification, not an intent."""
    with pytest.raises(ValidationError, match="requires `day`"):
        Horizon(label="specific_day")


def test_a_horizon_cannot_exceed_the_committed_day():
    with pytest.raises(ValidationError):
        Horizon(hours=48)


def test_duplicate_assets_are_refused():
    with pytest.raises(ValidationError, match="duplicate assets"):
        OptimizationIntent(objective="MINIMIZE_COST", assets=["battery", "battery"])


def test_comparison_variants_need_distinct_labels():
    with pytest.raises(ValidationError, match="distinct labels"):
        OptimizationIntent(
            objective="MINIMIZE_COST",
            comparison=[
                {"label": "big", "battery_energy_mwh": 1.0},
                {"label": "big", "battery_energy_mwh": 2.0},
            ],
        )


def test_a_scenario_variant_is_a_closed_set_of_fields():
    """'What if the battery were bigger' is a question; arbitrary attribute
    assignment is not, and a free-form patch dict would have made it one."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ScenarioVariant(label="sneaky", __class__="something else")


def test_comparison_is_bounded():
    with pytest.raises(ValidationError):
        OptimizationIntent(
            objective="MINIMIZE_COST",
            comparison=[{"label": f"v{i}", "battery_energy_mwh": 1.0} for i in range(9)],
        )


def test_notes_cannot_smuggle_a_wall_of_text():
    with pytest.raises(ValidationError):
        OptimizationIntent(objective="MINIMIZE_COST", notes="x" * 5000)


# --- clarification and rejection -------------------------------------------


def test_a_clarification_must_offer_a_real_choice():
    """One option is a decision wearing a question mark."""
    with pytest.raises(ValidationError):
        Clarification(question="Which?", options=["only one"])


def test_a_clarification_with_two_options_is_fine():
    clarification = Clarification(
        question="What does 'more aggressive' mean?",
        options=["a lower SOC floor", "a higher discharge rate"],
    )
    assert len(clarification.options) == 2


def test_a_rejection_must_carry_a_reason():
    with pytest.raises(ValidationError):
        Rejection(reason="", code="empty")


def test_a_rejection_prints_usefully():
    rejection = Rejection(code="absent_asset", reason="this site has no battery")
    assert str(rejection) == "[absent_asset] this site has no battery"


# --- the prompt is generated from the schema -------------------------------


def test_the_json_schema_is_generated_not_hand_written():
    """A prompt describing a drifted schema manufactures model errors."""
    schema = intent_json_schema()
    assert schema["additionalProperties"] is False
    assert "objective" in schema["properties"]
    assert "objective" in schema["required"]


def test_summary_shows_what_was_parsed():
    """§22 asks that the parse be visible to the user."""
    intent = OptimizationIntent(
        objective="MINIMIZE_COST",
        horizon={"label": "specific_day", "day": "2020-09-21"},
        constraints={"minimum_soc_frac": 0.2},
        shift_flexible_loads=True,
    )
    summary = intent.summary()
    assert "MINIMIZE_COST" in summary
    assert "2020-09-21" in summary
    assert "0.2" in summary
    assert "shift flexible loads" in summary
