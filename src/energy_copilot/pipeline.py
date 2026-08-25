"""The whole path: request in, answer or refusal out.

    request -> parse -> schema -> validate -> build -> solve -> KPIs -> explain

Each stage is named, and every response records **which stage it stopped at**.
That field is not decoration. It is how the security claim gets measured rather
than asserted: run an attack through the pipeline and the answer to "what saved
us" is a fact in the result, not a story told afterwards.

The stages a request can stop at, in order:

`parse`      the parser refused, or asked a question
`schema`     the structure would not validate — bad type, bad range, unknown key
`validate`   valid, and impossible at this site
`solve`      valid, possible, and infeasible in combination
`complete`   it ran

The distinction between `parse` and everything after it is the one that matters.
A refusal at `parse` depends on the parser's judgement — on a prompt, on a
pattern list, on a model's manners. A refusal at `schema` or `validate` does
not: those layers have never seen the request, only the structure it produced,
and they behave identically whoever wrote it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from energy_copilot.domain.config import SiteConfig, day_profile, scale_to_site
from energy_copilot.kpis import KPIs, baseline_kpis, comparison_table, improvement, result_kpis
from energy_copilot.optimization.solve import (
    DispatchResult,
    InfeasibleRequest,
    check_no_simultaneous_flows,
    check_power_balance,
    solve_intent,
)
from energy_copilot.parsing.rule_based import parse as rule_based_parse
from energy_copilot.schemas.intent import (
    Clarification,
    OptimizationIntent,
    ParseOutcome,
    Rejection,
)
from energy_copilot.validation.validator import ValidationReport, validate_intent

STAGES = ("parse", "schema", "validate", "solve", "complete")

# The parser refuses in schema's name when a model returns something that will
# not validate, so those codes are attributed to the schema rather than to the
# parser's judgement. Getting this wrong would overstate how much the parser is
# doing.
SCHEMA_CODES = frozenset(
    {"schema_rejected", "unparseable_response", "invalid_clarification", "empty_clarification"}
)


@dataclass
class CopilotResponse:
    """Everything one request produced, including where it stopped."""

    request: str
    stopped_at: str
    outcome: ParseOutcome | None = None
    validation: ValidationReport | None = None
    result: DispatchResult | None = None
    before: KPIs | None = None
    after: KPIs | None = None
    delta: dict[str, float] = field(default_factory=dict)
    physics_problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    prompt_version: str | None = None
    parser_name: str = "rule_based"

    @property
    def ok(self) -> bool:
        return self.stopped_at == "complete"

    @property
    def reached_solver(self) -> bool:
        """Did anything actually get built and solved?

        The security tests assert this is False for every attack. It is a
        stronger statement than "the answer was refused", because it says the
        attack never became a model at all.
        """
        return self.result is not None

    @property
    def intent(self) -> OptimizationIntent | None:
        return self.outcome if isinstance(self.outcome, OptimizationIntent) else None

    def comparison(self) -> pd.DataFrame | None:
        if self.before is None or self.after is None:
            return None
        return comparison_table(self.before, self.after)

    def summary(self) -> str:
        if isinstance(self.outcome, Clarification):
            return f"question: {self.outcome.question}"
        if isinstance(self.outcome, Rejection):
            return f"refused at {self.stopped_at}: {self.outcome}"
        if self.validation is not None and not self.validation.ok:
            reasons = "; ".join(str(r) for r in self.validation.rejections)
            return f"refused at validate: {reasons}"
        if self.result is not None:
            return (
                f"{self.result.objective_name}: cost "
                f"{self.after.cost_eur:.2f} EUR against a baseline of "
                f"{self.before.cost_eur:.2f} EUR"
            )
        return f"stopped at {self.stopped_at}"


def run_request(
    request: str,
    site: SiteConfig | None = None,
    parser: Callable[[str, SiteConfig], ParseOutcome] | None = None,
    profile: pd.DataFrame | None = None,
    day: str | None = None,
) -> CopilotResponse:
    """Run one request through every stage, stopping at the first refusal.

    `parser` is injectable so the same pipeline can be driven by the rule-based
    arm, an LLM arm, or an attacker's payload. Nothing downstream of it knows or
    cares which was used, which is the property the security tests check.
    """
    site = site or SiteConfig()
    parse_fn = parser or rule_based_parse
    response = CopilotResponse(request=request, stopped_at="parse")

    outcome = parse_fn(request, site)
    response.outcome = outcome

    if isinstance(outcome, Clarification):
        return response
    if isinstance(outcome, Rejection):
        response.stopped_at = "schema" if outcome.code in SCHEMA_CODES else "parse"
        return response
    if not isinstance(outcome, OptimizationIntent):
        # A parser that returned something outside the union. Refuse rather than
        # trust it: this is exactly the shape a compromised component takes.
        response.stopped_at = "schema"
        response.outcome = Rejection(
            code="unexpected_parser_output",
            reason=(
                f"the parser returned {type(outcome).__name__}, which is not an "
                "intent, a clarification or a rejection"
            ),
        )
        return response

    response.stopped_at = "validate"
    report = validate_intent(outcome, site)
    response.validation = report
    response.notes.extend(report.warnings)
    if not report.ok:
        return response

    if profile is None:
        chosen = day or outcome.horizon.day
        profile = day_profile(chosen) if chosen else _default_profile()
    profile = scale_to_site(profile, site)

    response.stopped_at = "solve"
    try:
        result = solve_intent(outcome, site, profile)
    except InfeasibleRequest as exc:
        response.outcome = Rejection(
            code="infeasible",
            reason=str(exc),
            offending_value=outcome.summary()[:200],
        )
        return response

    response.result = result
    response.physics_problems = check_power_balance(result) + check_no_simultaneous_flows(result)
    response.before = baseline_kpis(site, profile, hours=len(result.schedule))
    response.after = result_kpis(result)
    response.delta = improvement(response.before, response.after)
    response.stopped_at = "complete"
    return response


def _default_profile() -> pd.DataFrame:
    """"Tomorrow", when nobody said which day.

    The widest-spread day, chosen because it makes the difference between
    objectives visible. Stated here rather than left as an accident of file
    order, since a demo that silently picks the flattest day looks broken.
    """
    return day_profile("2020-09-21")
