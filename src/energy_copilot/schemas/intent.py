"""The structured intent, and the contract the model has to satisfy.

This is the narrow gate between prose and arithmetic. Everything upstream is
language; everything downstream is a linear program. Nothing crosses except an
`OptimizationIntent`, and an `OptimizationIntent` cannot be constructed loosely.

## Two layers of bounds, on purpose

There are two different questions to ask about `minimum_soc = 1.5`, and they
belong in different places:

**Is it meaningful at all?** A state of charge outside [0, 1] is not a fact
about this site — it is not a fact about any site. That is enforced *here*, by
the schema, because it is true independently of configuration.

**Is it possible here?** A minimum SOC of 0.05 is perfectly meaningful and is
still impossible on a battery whose floor is 0.10. That is enforced by
`validation/`, which knows the site.

Splitting them means an attacker has to get past both, and neither one has to
know about the other. It also means the error messages are useful: the schema
says "not a fraction", the validator says "below this battery's floor of 10%".

## `extra="forbid"` is load-bearing

A model that hallucinates `"allow_grid_overload": true` must produce an error,
not a silently ignored key. Pydantic's default is to drop unknown fields, which
would let an injected instruction ride along invisibly and give the *appearance*
of having been accepted. Forbidding extras converts that from a silent no-op
into a refusal with the offending key named.

## No units in here

Fields carry their unit in the name — `_mw`, `_mwh`, `_frac` — and hold plain
numbers in canonical units. Unit *strings* never reach this layer; normalising
them is the parser's job, in `domain/units.py`.

That leaves one gap worth naming: a model that means 200 kW and writes `200`
into `max_grid_import_mw` produces a schema-valid intent. It does not survive,
because 200 MW on a 0.6 MW connection is refused by the validator as physically
impossible. Unit confusion of that size becomes a bounds violation, which is
exactly where it should land.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from energy_copilot.domain.ontology import Asset, Objective

# Everything in this module forbids unknown keys and rejects sloppy types.
STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class Horizon(BaseModel):
    """When. A named day from the dataset, or a relative label resolved later."""

    model_config = STRICT

    label: Literal["today", "tomorrow", "specific_day"] = "tomorrow"
    day: str | None = Field(
        default=None,
        description="ISO date from the committed dataset, e.g. '2020-09-21'",
        max_length=10,
    )
    hours: int = Field(default=24, ge=1, le=24)

    @model_validator(mode="after")
    def _specific_day_needs_a_day(self) -> Horizon:
        if self.label == "specific_day" and not self.day:
            raise ValueError(
                "label='specific_day' requires `day`. Saying 'that day' without "
                "naming it is a clarification, not an intent."
            )
        return self


class Constraints(BaseModel):
    """Operating limits the user asked for, in canonical units.

    Every bound here is one a *person* can meaningfully state. Limits that
    belong to the equipment live in `SiteConfig` and are not negotiable from a
    prompt — which is the point: a request cannot raise the battery's power
    rating by asking nicely.
    """

    model_config = STRICT

    minimum_soc_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_soc_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    terminal_soc_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    max_grid_import_mw: float | None = Field(default=None, ge=0.0)
    max_grid_export_mw: float | None = Field(default=None, ge=0.0)
    max_peak_mw: float | None = Field(default=None, gt=0.0)
    required_energy_mwh: float | None = Field(default=None, ge=0.0)
    allow_export: bool | None = None
    allow_pv_curtailment: bool | None = None

    @model_validator(mode="after")
    def _soc_window_is_ordered(self) -> Constraints:
        low, high = self.minimum_soc_frac, self.maximum_soc_frac
        if low is not None and high is not None and low > high:
            raise ValueError(
                f"minimum_soc_frac ({low}) exceeds maximum_soc_frac ({high}); "
                "no state of charge satisfies both"
            )
        return self

    def stated(self) -> dict[str, object]:
        """Only the limits actually asked for — used in explanations."""
        return {
            name: value
            for name, value in self.model_dump().items()
            if value is not None
        }


class ScenarioVariant(BaseModel):
    """One arm of a §17 comparison, expressed as an override of the site.

    Deliberately a **closed set of fields** rather than a free-form patch. "What
    if the battery were twice as big" is a supported question; "what if
    `__class__` were something else" is not a question at all, and a dictionary
    of arbitrary attribute names would have made it one.
    """

    model_config = STRICT

    label: str = Field(max_length=60)
    battery_energy_mwh: float | None = Field(default=None, gt=0.0, le=100.0)
    battery_power_mw: float | None = Field(default=None, gt=0.0, le=100.0)
    pv_peak_mw: float | None = Field(default=None, ge=0.0, le=100.0)


class OptimizationIntent(BaseModel):
    """What the user wants, in a form the builder can act on."""

    model_config = STRICT

    objective: Objective
    horizon: Horizon = Field(default_factory=Horizon)
    constraints: Constraints = Field(default_factory=Constraints)
    assets: list[Asset] = Field(default_factory=list)
    shift_flexible_loads: bool = False
    comparison: list[ScenarioVariant] = Field(default_factory=list, max_length=4)
    notes: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _assets_are_unique(self) -> OptimizationIntent:
        if len(set(self.assets)) != len(self.assets):
            raise ValueError(f"duplicate assets in {self.assets}")
        return self

    @model_validator(mode="after")
    def _comparison_labels_are_unique(self) -> OptimizationIntent:
        labels = [variant.label for variant in self.comparison]
        if len(set(labels)) != len(labels):
            raise ValueError(f"comparison variants need distinct labels, got {labels}")
        return self

    def summary(self) -> str:
        """One line, for the UI and the log. Shown to the user so the parse is
        inspectable — §22 asks for exactly this."""
        parts = [self.objective.value, f"horizon={self.horizon.label}"]
        if self.horizon.day:
            parts.append(self.horizon.day)
        stated = self.constraints.stated()
        if stated:
            parts.append(", ".join(f"{k}={v}" for k, v in stated.items()))
        if self.shift_flexible_loads:
            parts.append("shift flexible loads")
        if self.comparison:
            parts.append(f"compare {[v.label for v in self.comparison]}")
        return " | ".join(parts)


class Clarification(BaseModel):
    """A question back, because guessing would change the answer materially.

    §16's example is "make the battery more aggressive", which could mean a
    lower SOC floor, a higher discharge rate, or simply a stronger weighting on
    cost. Those produce different schedules, so the system asks instead of
    picking.

    `options` is required and must offer at least two. A clarification with one
    option is a decision wearing a question mark.
    """

    model_config = STRICT

    question: str = Field(min_length=1, max_length=300)
    options: list[str] = Field(min_length=2, max_length=6)
    reason: str = Field(default="", max_length=300)
    partial_intent: dict = Field(default_factory=dict)


class Rejection(BaseModel):
    """A refusal, with a reason a person can act on.

    §10 requires the validator to explain *why*. A boolean would satisfy the
    letter and be useless: "rejected" tells a user nothing, while "this site has
    no diesel generator" tells them their question was about the wrong site.
    """

    model_config = STRICT

    reason: str = Field(min_length=1, max_length=500)
    code: str = Field(min_length=1, max_length=60)
    offending_value: str | None = Field(default=None, max_length=200)

    def __str__(self) -> str:
        return f"[{self.code}] {self.reason}"


# What a parser returns. A union rather than an intent-or-None, so that
# "I need to ask something" and "I refuse" stay distinguishable all the way up.
ParseOutcome = OptimizationIntent | Clarification | Rejection


def intent_json_schema() -> dict:
    """The schema the prompt shows the model.

    Generated, never hand-written into the prompt file. A prompt describing a
    schema that has drifted from the code manufactures failures that look like
    model errors and are not.
    """
    return OptimizationIntent.model_json_schema()
