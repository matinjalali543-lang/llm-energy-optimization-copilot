"""A deterministic parser, and the control arm the LLM is measured against.

This is not a mock. It is a genuine natural-language parser built from patterns
and the domain ontology, and on the phrasings it was designed for it is exactly
as correct as a language model — faster, free, and reproducible to the character.

## Why it exists

Without it, an accuracy figure for the LLM answers only "did it run". With it,
the evaluation answers **what the LLM adds over pattern matching**, which is the
question worth asking before putting a model in a control path. It is the same
question `forecast-driven-bess-dispatch` asked about forecasts: not "does the
model work" but "is it worth what it costs".

It also means the evaluation reports a real, reproducible number in CI with no
API key, no network and no billing account.

## What it is expected to be bad at

Written down in advance, so the comparison is a prediction rather than a
rationalisation after the fact. This parser should lose on:

* **paraphrase it has no pattern for** — "trim the bill", "go easy on the grid"
* **implicit units** — "keep it under 300" with the unit only inferable from
  context
* **compound requests** — two objectives in one sentence, where choosing which
  governs needs judgement
* **novel phrasings of ambiguity** — it can only flag the ambiguities it was
  told about

And win on: anything it has a pattern for, plus every adversarial prompt, since
it has no instructions to override.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from energy_copilot.domain.config import SiteConfig, available_days
from energy_copilot.domain.ontology import Objective, explain_absent_asset
from energy_copilot.domain.units import UnitError, convert
from energy_copilot.schemas.intent import (
    Clarification,
    OptimizationIntent,
    ParseOutcome,
    Rejection,
)

# --- objective cues, most specific first -----------------------------------
# Order matters: "reduce my carbon cost" contains "cost", so emissions cues are
# tested first. A parser that scanned in dictionary order would answer
# MINIMIZE_COST to a request about carbon and look plausible doing it.
OBJECTIVE_PATTERNS: tuple[tuple[str, Objective], ...] = (
    (
        r"\b(carbon|emission|emissions|co2|greener|greenest|green)\b",
        Objective.MINIMIZE_EMISSIONS,
    ),
    (
        r"\b(self[- ]consum\w*|use (my|our) own (solar|pv)|on[- ]site solar)\b",
        Objective.MAXIMIZE_SELF_CONSUMPTION,
    ),
    # "peak-price hours" is about PRICE, and it must be caught before the
    # peak-demand pattern below. Scanning in dictionary order answered
    # MINIMIZE_PEAK to "shift flexible loads away from peak-price hours" and
    # looked entirely plausible doing it: the intent was well-formed, passed the
    # schema, passed the validator, solved -- and optimised the wrong thing.
    # Nothing downstream can catch that, which is what makes it worth a comment.
    (r"\bpeak[- ]price", Objective.MINIMIZE_COST),
    (r"\b(peak|demand charge|capacity charge|flatten)\b", Objective.MINIMIZE_PEAK),
    (
        # Plurals matter and are easy to miss. `\bcost\b` does not match
        # "costs", so "cut costs but keep import under 300 kW" fell through to
        # "which objective should govern?" -- a clarification for a request that
        # names its objective in the first two words.
        r"\b(costs?|cheap\w*|bills?|spend\w*|prices?|expensive|save money|savings?)\b",
        Objective.MINIMIZE_COST,
    ),
)

# Requests that are not about operating the site at all. This arm refuses them
# outright, which is easy for it: it has no instructions that could be
# overridden in the first place.
INJECTION_PATTERNS: tuple[str, ...] = (
    r"\bignore\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}"
    r"\b(instruction|prompt|rule)",
    r"\bdisregard\b.{0,30}\b(instruction|prompt|rule|constraint|limit)",
    r"\b(execute|run|eval)\b.{0,20}\b(this )?(python|code|script|command|shell)\b",
    r"(os\.system|subprocess|import os|__import__|exec\()",
    r"\boverride\b.{0,30}\b(physical|safety|limit|constraint)",
    r"\byou are now\b|\bnew instructions\b|\bsystem prompt\b",
)

# --- ambiguity: phrasings that change the answer without saying how ---------
AMBIGUOUS_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        r"\b(more )?aggressive\w*\b",
        "What should 'more aggressive' change?",
        (
            "discharge deeper by lowering the state-of-charge floor",
            "cycle more often to chase price spreads",
            "prioritise peak reduction over cost",
        ),
    ),
    (
        r"\b(optimi[sz]e|improve|make it better|do better|sort out)\b"
        r"(?!.*\b(cost|peak|carbon|emission|self))",
        "Which objective should govern?",
        (
            "minimise cost",
            "minimise peak import",
            "maximise self-consumption",
            "minimise emissions",
        ),
    ),
    (
        r"\bbetter\b(?!.*\bthan\b)",
        "Better in which respect?",
        ("cheaper", "lower peak", "lower emissions"),
    ),
)

PERCENT = r"(\d+(?:\.\d+)?)\s*%"
NUMBER = r"(\d+(?:\.\d+)?)"
POWER_UNIT = r"(kw|mw|kilowatts?|megawatts?)"
ENERGY_UNIT = r"(kwh|mwh|kilowatt[- ]hours?|megawatt[- ]hours?)"

# A gap that may not cross a clause boundary or another constraint's noun.
# "Lower the peak and hold export under 100 kW" let the peak pattern reach
# across "and hold export" and claim the export figure, capping total import at
# 100 kW on a site whose load alone exceeds that. The request was well formed
# and the resulting model was infeasible, which is the least debuggable pair of
# properties a parse can have.
GAP = r"(?:(?!\b(?:and|export|import|peak|battery|soc)\b)[^,;.])"


@dataclass(frozen=True)
class ParseTrace:
    """What the parser matched, so a wrong answer can be diagnosed."""

    matched: tuple[str, ...] = ()
    unmatched_terms: tuple[str, ...] = ()


def parse(request: str, site: SiteConfig | None = None) -> ParseOutcome:
    """Turn a request into an intent, a clarification, or a refusal."""
    site = site or SiteConfig()
    text = request.lower().strip()
    if not text:
        return Rejection(code="empty_request", reason="the request is empty")

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return Rejection(
                code="out_of_scope",
                reason=(
                    "this request is not about operating the site: it asks to "
                    "override the system's own instructions or execute code. "
                    "Note that refusing here is a convenience, not the "
                    "safeguard -- the schema and validator would reject the "
                    "result regardless of what any parser returned."
                ),
                offending_value=request[:200],
            )

    # Unsupported equipment next. A request about a diesel generator is not a
    # cost request that happens to mention diesel.
    absent = _absent_asset(text)
    if absent is not None:
        return Rejection(
            code="absent_asset",
            reason=absent,
            offending_value=request[:200],
        )

    ambiguity = _ambiguity(text)
    if ambiguity is not None:
        return ambiguity

    # Impossible values are refused BEFORE asking which objective governs.
    # "Set the state of charge to 150%" used to be answered with "which
    # objective should govern?" — a question whose answer cannot rescue the
    # request, and which makes the system look like it did not read it. A
    # fraction outside [0, 1] is meaningless whatever the goal is.
    impossible = _impossible_percentage(text)
    if impossible is not None:
        return impossible

    objective = _objective(text)
    comparison = _comparison(text)
    if objective is None and comparison:
        # "Compare a 1 MWh battery with a 2 MWh battery" names no goal, and
        # asking which one would be pedantic: cost is the only objective under
        # which battery *size* is the interesting variable. §17 lists this
        # exact request as one the system should answer.
        objective = Objective.MINIMIZE_COST
    if objective is None:
        return Clarification(
            question="Which objective should govern this request?",
            options=[
                "minimise cost",
                "minimise peak import",
                "maximise self-consumption",
                "minimise emissions",
            ],
            reason="the request does not name a goal this system can optimise",
        )

    constraints: dict[str, object] = {}
    unit_problem = _soc_constraints(text, constraints)
    if unit_problem is not None:
        return unit_problem
    unit_problem = _power_constraints(text, constraints, site)
    if unit_problem is not None:
        return unit_problem

    if re.search(r"\b(no|without|don'?t|do not|not|never)\s+export\w*\b", text):
        constraints["allow_export"] = False
    if re.search(r"\b(no|without|don'?t|do not|not|never)\s+curtail\w*\b", text):
        constraints["allow_pv_curtailment"] = False

    # Comparison variants consume quantities too — "compare a 1 MWh battery
    # with a 2 MWh battery" states two numbers and uses both. Checking only
    # `constraints` made the guard reject every comparison request it was meant
    # to protect.
    consumed = dict(constraints)
    for index, variant in enumerate(comparison):
        for key, value in variant.items():
            if isinstance(value, int | float):
                consumed[f"variant_{index}_{key}"] = value

    leftover = _unconsumed_quantity(text, consumed)
    if leftover is not None:
        return leftover

    try:
        return OptimizationIntent(
            objective=objective,
            horizon=_horizon(text),
            constraints=constraints,
            assets=_assets(text, site),
            shift_flexible_loads=_shifting(text),
            comparison=comparison,
        )
    except ValueError as exc:
        # The schema refused what the patterns produced. That is the schema
        # doing its job on this parser exactly as it would on a model.
        return Rejection(
            code="schema_rejected",
            reason=str(exc).split("\n")[0][:400],
            offending_value=request[:200],
        )


# Numbers that are part of a date, not a quantity.
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
QUANTITY = re.compile(
    r"(\d+(?:\.\d+)?)\s*(%|kwh|mwh|kw|mw|hours?|hrs?)\b", re.IGNORECASE
)


SOC_CONTEXT = re.compile(r"\b(charge|charged|soc|state of charge|battery)\b")


def _impossible_percentage(text: str) -> Rejection | None:
    """A state of charge outside 0-100% is meaningless whatever the goal is."""
    if not SOC_CONTEXT.search(text):
        return None
    for match in re.finditer(PERCENT, text):
        value = float(match.group(1))
        if value > 100.0:
            return Rejection(
                code="impossible_value",
                reason=(
                    f"{value:g}% is not a state of charge a battery can reach: a "
                    "fraction of full charge cannot exceed 100%. This is refused "
                    "regardless of what the request is trying to achieve."
                ),
                offending_value=match.group(0),
            )
    return None


def _unconsumed_quantity(text: str, constraints: dict) -> Rejection | None:
    """Refuse when the request states a quantity the parser did not use.

    This is the difference between failing and failing *quietly*. Before this
    guard, "cap import at 200 MW" produced a confident, well-formed, entirely
    valid intent with no import cap in it — so the operator was told their
    request had been carried out, and the number they gave had simply
    evaporated. Seven of the eight failing evaluation prompts were that shape.

    A parser that cannot use a number the user supplied has not understood the
    request, and saying so is strictly better than proceeding as though the
    number were never there.
    """
    without_dates = ISO_DATE.sub(" ", text)
    used = [float(v) for v in constraints.values() if isinstance(v, int | float)]

    # A limit written as a word is still a limit. "grid import capped at zero"
    # carries no digits, so the numeric scan below never saw it and the cap was
    # dropped exactly as silently as "cap import at 200 MW" once was.
    worded = re.search(
        r"\b(?:import|draw|export|peak)\w*\b[^,;.]{0,30}?"
        r"(?:capped at|cap\w* at|at|to|under|below)\s+(zero|none|nothing)\b",
        without_dates,
    )
    if worded and not any(abs(value) <= 1e-12 for value in used):
        return Rejection(
            code="unused_quantity",
            reason=(
                f"'{worded.group(0).strip()}' states a limit of zero, which this "
                "parser did not convert into a constraint. Proceeding would mean "
                "ignoring it; write it as '0 MW' to be sure it is applied."
            ),
            offending_value=worded.group(0).strip()[:60],
        )

    for match in QUANTITY.finditer(without_dates):
        raw, unit = float(match.group(1)), match.group(2).lower()
        if unit == "%":
            candidates = [raw / 100.0, raw]
        elif unit in ("kw", "kwh"):
            candidates = [raw / 1000.0, raw]
        elif unit.startswith("h"):
            # An hour count is a horizon, and 24 is the only one supported.
            if abs(raw - 24) < 1e-9:
                continue
            return Rejection(
                code="unsupported_horizon",
                reason=(
                    f"this system optimises one day at a time, so a horizon of "
                    f"{raw:g} hours cannot be honoured. Ask for a single day."
                ),
                offending_value=match.group(0),
            )
        else:
            candidates = [raw]

        if not any(
            any(abs(candidate - value) <= 1e-9 for value in used)
            for candidate in candidates
        ):
            return Rejection(
                code="unused_quantity",
                reason=(
                    f"'{match.group(0)}' was not understood as a limit this "
                    "system can apply, and proceeding would mean silently "
                    "ignoring a number you gave. Try phrasing it as 'keep grid "
                    "import under 400 kW' or 'keep the battery above 20%'."
                ),
                offending_value=match.group(0),
            )
    return None


def _absent_asset(text: str) -> str | None:
    """Check single words and adjacent pairs against the ontology.

    The first version scanned `[a-z][a-z ]{2,20}` runs, which produced strings
    like "run the diesel generator at " -- none of which the ontology has ever
    heard of, so "run the diesel generator at 200 MW" sailed through as an
    ordinary request with no objective. Word n-grams find the term.
    """
    words = re.findall(r"[a-z]+", text)
    for size in (2, 1):
        for start in range(len(words) - size + 1):
            explanation = explain_absent_asset(" ".join(words[start : start + size]))
            if explanation:
                return explanation
    return None


def _ambiguity(text: str) -> Clarification | None:
    for pattern, question, options in AMBIGUOUS_PATTERNS:
        if re.search(pattern, text):
            return Clarification(
                question=question,
                options=list(options),
                reason="this could mean materially different things and the "
                "choice changes the schedule",
            )
    return None


def _objective(text: str) -> Objective | None:
    for pattern, objective in OBJECTIVE_PATTERNS:
        if re.search(pattern, text):
            return objective
    return None


def _horizon(text: str) -> dict:
    explicit = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if explicit:
        return {"label": "specific_day", "day": explicit.group(1), "hours": 24}
    if re.search(r"\btoday\b", text):
        return {"label": "today", "hours": 24}
    return {"label": "tomorrow", "hours": 24}


def _assets(text: str, site: SiteConfig) -> list[str]:
    assets = []
    if site.has_battery and re.search(r"\b(battery|storage|soc|state of charge)\b", text):
        assets.append("battery")
    if site.pv_peak_mw > 0 and re.search(r"\b(pv|solar|panels?)\b", text):
        assets.append("pv")
    if re.search(r"\b(grid|import|export|connection)\b", text):
        assets.append("grid")
    if site.has_flexible_load and re.search(r"\b(flexible|shiftable|deferrable)\b", text):
        assets.append("flexible_load")
    return assets


def _shifting(text: str) -> bool:
    return bool(
        re.search(r"\b(shift|move|reschedul\w+|defer)\b.{0,30}\b(load|demand|block)\b", text)
        or re.search(r"\bflexible loads?\b", text)
    )


def _soc_constraints(text: str, constraints: dict) -> Rejection | None:
    """Floor, ceiling and closing charge, as fractions."""
    # "between 20% and 80%" sets both ends at once, and has to be read before
    # either single-ended pattern gets hold of one of the numbers.
    window = re.search(rf"between\s*{PERCENT}\s*and\s*{PERCENT}", text)
    if window:
        low, high = float(window.group(1)) / 100.0, float(window.group(2)) / 100.0
        constraints["minimum_soc_frac"] = min(low, high)
        constraints["maximum_soc_frac"] = max(low, high)
        return None

    # A NEGATED "below" is a floor, not a ceiling. "never let the battery go
    # below 99%" was read as a maximum of 99% -- the exact opposite of the
    # request, expressed as a perfectly valid intent that then solved. Checked
    # first, because the plain ceiling pattern would otherwise claim the number.
    negated_floor = re.search(
        # Up to five intervening words: "never LET THE BATTERY GO below 99%" is
        # four, and a limit of three silently fell through to the ceiling
        # pattern and inverted the constraint.
        rf"(?:never|not|don'?t|do not|no lower than|avoid)\s+(?:\w+\s+){{0,5}}"
        rf"(?:below|under)\D{{0,10}}{PERCENT}",
        text,
    )
    if negated_floor:
        constraints["minimum_soc_frac"] = float(negated_floor.group(1)) / 100.0
        return None

    floor = re.search(
        rf"(?:above|over|at least|no lower than|not below|minimum(?: of)?"
        rf"|keep\w*(?: soc)?(?: above)?)\D{{0,20}}{PERCENT}",
        text,
    )
    if floor:
        constraints["minimum_soc_frac"] = float(floor.group(1)) / 100.0

    ceiling = re.search(
        rf"(?:below|under|at most|no higher than|maximum(?: of)?)\D{{0,20}}{PERCENT}",
        text,
    )
    if ceiling and "minimum_soc_frac" not in constraints:
        constraints["maximum_soc_frac"] = float(ceiling.group(1)) / 100.0

    terminal = re.search(
        rf"(?:end|finish|close|leave|by (?:the )?end of (?:the )?day)\D{{0,30}}{PERCENT}", text
    )
    if terminal:
        constraints["terminal_soc_frac"] = float(terminal.group(1)) / 100.0
    return None


def _power_constraints(text: str, constraints: dict, site: SiteConfig) -> Rejection | None:
    """Grid limits, normalised into MW.

    A bare number with no unit is refused rather than guessed. "Keep import
    under 300" on a 0.6 MW connection means 300 kW to a person and 300 MW to a
    parser, and the difference is three orders of magnitude.
    """
    limit = re.search(
        rf"\b(?:import|draw|take|pull)\w*\b{GAP}{{0,30}}?"
        rf"(?:under|below|less than|at most|no more than|more than|capped at"
        rf"|cap\w* at|at|to)\s*{NUMBER}\s*{POWER_UNIT}?",
        text,
    )
    if limit:
        value, unit = limit.group(1), limit.group(2)
        if unit is None:
            return Rejection(
                code="missing_unit",
                reason=(
                    f"'{value}' has no unit. On a {site.grid_import_limit_mw} MW "
                    f"connection, {value} kW and {value} MW differ by a factor of a "
                    "thousand, so this is not something to guess at."
                ),
                offending_value=value,
            )
        try:
            constraints["max_grid_import_mw"] = convert(
                float(value), _canonical_power_unit(unit), "MW"
            )
        except UnitError as exc:
            return Rejection(code="bad_unit", reason=str(exc), offending_value=unit)

    export_cap = re.search(
        rf"\bexport\w*\b{GAP}{{0,30}}?"
        rf"(?:under|below|less than|at most|no more than|more than|capped at"
        rf"|cap\w* at|at|to)\s*{NUMBER}\s*{POWER_UNIT}",
        text,
    )
    if export_cap:
        try:
            constraints["max_grid_export_mw"] = convert(
                float(export_cap.group(1)), _canonical_power_unit(export_cap.group(2)), "MW"
            )
        except UnitError as exc:
            return Rejection(code="bad_unit", reason=str(exc))

    energy = re.search(
        rf"\b(?:deliver|supply|provide|need|require)\w*\b\D{{0,20}}{NUMBER}\s*{ENERGY_UNIT}",
        text,
    )
    if energy:
        try:
            constraints["required_energy_mwh"] = convert(
                float(energy.group(1)),
                "kWh" if energy.group(2).startswith("k") else "MWh",
                "MWh",
            )
        except UnitError as exc:
            return Rejection(code="bad_unit", reason=str(exc))

    peak_cap = re.search(
        rf"\bpeak\b{GAP}{{0,30}}?(?:under|below|at most|no more than|to)"
        rf"\s*{NUMBER}\s*{POWER_UNIT}",
        text,
    )
    if peak_cap:
        try:
            constraints["max_peak_mw"] = convert(
                float(peak_cap.group(1)), _canonical_power_unit(peak_cap.group(2)), "MW"
            )
        except UnitError as exc:
            return Rejection(code="bad_unit", reason=str(exc))
    return None


def _canonical_power_unit(raw: str) -> str:
    raw = raw.lower().rstrip("s")
    return {"kw": "kW", "kilowatt": "kW", "mw": "MW", "megawatt": "MW"}[raw]


def _comparison(text: str) -> list[dict]:
    """`compare a 1 MWh battery with a 2 MWh battery`."""
    if not re.search(r"\b(compare|versus|vs\.?|against|what if)\b", text):
        return []
    sizes = re.findall(rf"{NUMBER}\s*{ENERGY_UNIT}", text)
    variants = []
    for value, unit in sizes[:4]:
        try:
            mwh = convert(float(value), "kWh" if unit.startswith("k") else "MWh", "MWh")
        except UnitError:
            continue
        label = f"{value} {unit}"
        if any(variant["label"] == label for variant in variants):
            continue
        variants.append({"label": label, "battery_energy_mwh": mwh})
    return variants if len(variants) >= 2 else []


def extract_constraints(text: str, site: SiteConfig | None = None) -> dict[str, object]:
    """Pull only the limits out of a fragment, ignoring objective and horizon.

    Exists for follow-up turns. "Keep at least 20% in the battery" is not a
    request — it names no goal — but it does carry a fact, and a conversation
    that discarded it would be re-asking for something it had just been told.
    `Conversation` merges the result into the intent already established.
    """
    site = site or SiteConfig()
    lowered = text.lower().strip()
    constraints: dict[str, object] = {}
    if _soc_constraints(lowered, constraints) is not None:
        return {}
    if _power_constraints(lowered, constraints, site) is not None:
        return {}
    if re.search(r"\b(no|without|don'?t|do not|not|never)\s+export\w*\b", lowered):
        constraints["allow_export"] = False
    if re.search(r"\b(no|without|don'?t|do not|not|never)\s+curtail\w*\b", lowered):
        constraints["allow_pv_curtailment"] = False
    return constraints


def known_days() -> list[str]:
    return available_days()
