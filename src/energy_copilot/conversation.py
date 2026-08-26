"""Multi-turn state, kept separate from the conversation text (§15).

The instruction in §15 is one line — "Keep structured state separate from
conversation text" — and it is the difference between a system that accumulates
understanding and one that re-reads its own transcript hoping to find it again.

Concretely: this class holds an `OptimizationIntent`. It does not hold a
paragraph describing an intent, and it never re-parses previous turns to work
out what was agreed. When a clarification is answered, the answer updates
*fields*, and everything already established stays established.

## Answering a question is not the same as making a request

    "Optimise tomorrow's cost."          -> a request, parsed in full
    "20%."                               -> an answer, only meaningful in context

A fragment like "20%" parsed as a fresh request is nonsense — no objective, no
verb, nothing to optimise. So when a clarification is outstanding, the next
message is read as an answer to it, against the options that were offered. That
is why `pending` exists.

## Clarifications carry the partial intent forward

A question asked halfway through does not discard what came before it. The
partial intent is kept on the conversation, and answering merges into it, so
"cut my bill tomorrow" followed by "keep 20% in the battery" produces one intent
with both facts rather than a second intent with only the last.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from energy_copilot.domain.config import SiteConfig
from energy_copilot.parsing.rule_based import extract_constraints
from energy_copilot.parsing.rule_based import parse as rule_based_parse
from energy_copilot.pipeline import SCHEMA_CODES, CopilotResponse, run_request
from energy_copilot.schemas.intent import (
    Clarification,
    OptimizationIntent,
    ParseOutcome,
    Rejection,
)

PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
BARE_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")

# Which field a clarification is asking about, keyed by words in its question.
# Small and explicit: a clarification whose subject cannot be identified is
# answered by re-parsing rather than by guessing which field to fill.
QUESTION_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("state-of-charge floor", "minimum_soc_frac"),
    ("state of charge", "minimum_soc_frac"),
    ("soc", "minimum_soc_frac"),
    ("minimum", "minimum_soc_frac"),
)


@dataclass
class Turn:
    """One exchange, kept for display only."""

    speaker: str
    text: str


@dataclass
class Conversation:
    """A session: structured state, plus a transcript nobody reasons from."""

    site: SiteConfig = field(default_factory=SiteConfig)
    parser: Callable[[str, SiteConfig], ParseOutcome] = rule_based_parse
    turns: list[Turn] = field(default_factory=list)
    intent: OptimizationIntent | None = None
    pending: Clarification | None = None
    day: str | None = None

    def send(self, message: str) -> CopilotResponse:
        """Handle one message and return the response."""
        self.turns.append(Turn("user", message))

        if self.pending is not None:
            response = self._answer_pending(message)
        elif self.intent is not None:
            response = self._followup(message)
        else:
            response = run_request(
                message, site=self.site, parser=self.parser, day=self.day
            )

        self._absorb(response)
        self.turns.append(Turn("assistant", response.summary()))
        return response

    # --- a follow-up that adds to what is already established ---------------

    def _followup(self, message: str) -> CopilotResponse:
        """A later turn may add a limit without restating the goal.

        "Keep at least 20% in the battery" names no objective, so parsed on its
        own it produces "which objective should govern?" — a question the
        conversation already knows the answer to. Merging the *fields* is the
        point of holding structured state at all; re-parsing the transcript to
        rediscover the objective would be the thing §15 warns against.
        """
        fresh = self.parser(message, self.site)
        if isinstance(fresh, OptimizationIntent):
            # A complete request in its own right. It replaces, rather than
            # merging, because the user restated the goal.
            return self._run_with(fresh)

        extra = extract_constraints(message, self.site)
        if extra:
            merged = revise(self.intent, **{f"constraint:{k}": v for k, v in extra.items()})
            return self._run_with(merged)

        # Nothing to merge: report whatever the parser made of it.
        response = CopilotResponse(request=message, stopped_at="parse", outcome=fresh)
        if isinstance(fresh, Rejection) and fresh.code in SCHEMA_CODES:
            response.stopped_at = "schema"
        return response

    def _run_with(self, intent: OptimizationIntent) -> CopilotResponse:
        return run_request(
            intent.summary(),
            site=self.site,
            parser=lambda _request, _site: intent,
            day=self.day,
        )

    # --- answering an outstanding question ---------------------------------

    def _answer_pending(self, message: str) -> CopilotResponse:
        question = self.pending
        assert question is not None
        field_name = _subject_of(question.question)
        value = _numeric_answer(message)

        if field_name is not None and value is not None:
            base = self.intent or _partial_to_intent(question)
            if base is not None:
                merged = revise(base, **{f"constraint:{field_name}": value})
                self.pending = None
                return run_request(
                    message,
                    site=self.site,
                    parser=lambda _request, _site: merged,
                    day=self.day,
                )

        # The answer chose one of the offered options by name or by position.
        # Each option maps to a FIELD CHANGE, not to a sentence: the first
        # version re-parsed "discharge deeper ... — MINIMIZE_COST | horizon..."
        # as though it were a fresh request, which produced another question.
        # Round-tripping structured state through prose is exactly what §15 is
        # warning about, and I had written it into the module that quotes §15.
        chosen = _chosen_option(message, question.options)
        if chosen is not None:
            base = self.intent or _partial_to_intent(question)
            update = _option_update(chosen, self.site)
            if base is not None and update is not None:
                self.pending = None
                return self._run_with(_apply_update(base, update))

        # Not recognisable as an answer. Treat it as a new request rather than
        # asking the same question again, which is the behaviour that makes
        # these systems infuriating.
        self.pending = None
        return run_request(message, site=self.site, parser=self.parser, day=self.day)

    # --- state ---------------------------------------------------------------

    def _absorb(self, response: CopilotResponse) -> None:
        if isinstance(response.outcome, Clarification):
            self.pending = response.outcome
            return
        self.pending = None
        if isinstance(response.outcome, OptimizationIntent):
            self.intent = response.outcome
        # A Rejection leaves `intent` alone: a refused turn should not erase
        # what the previous turns established.

    def transcript(self) -> str:
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in self.turns)

    def reset(self) -> None:
        self.turns.clear()
        self.intent = None
        self.pending = None


# What each offered option actually changes. Keyed by a distinctive phrase from
# the option text, because the options are generated in `rule_based.py` and the
# two lists must agree — a test asserts every generated option is answerable
# here, so they cannot drift apart silently.
OPTION_ACTIONS: tuple[tuple[str, str, object], ...] = (
    ("lowering the state-of-charge floor", "constraint:minimum_soc_frac", "battery_floor"),
    ("cycle more often", "objective", "MINIMIZE_COST"),
    ("prioritise peak reduction", "objective", "MINIMIZE_PEAK"),
    ("minimise cost", "objective", "MINIMIZE_COST"),
    ("minimise peak", "objective", "MINIMIZE_PEAK"),
    ("maximise self-consumption", "objective", "MAXIMIZE_SELF_CONSUMPTION"),
    ("minimise emissions", "objective", "MINIMIZE_EMISSIONS"),
    ("cheaper", "objective", "MINIMIZE_COST"),
    ("lower peak", "objective", "MINIMIZE_PEAK"),
    ("lower emissions", "objective", "MINIMIZE_EMISSIONS"),
)


def _option_update(chosen: str, site: SiteConfig) -> tuple[str, object] | None:
    lowered = chosen.lower()
    for phrase, target, value in OPTION_ACTIONS:
        if phrase in lowered:
            if value == "battery_floor":
                floor = site.battery.soc_min_frac if site.battery else 0.0
                return (target, floor)
            return (target, value)
    return None


def revise(intent: OptimizationIntent, **updates) -> OptimizationIntent:
    """Return a new intent with fields changed, **re-validated**.

    Not `model_copy(update=...)`. That skips validation, and it silently left a
    raw string in `objective` where the enum belonged, which then blew up three
    frames away in `summary()`. In a system whose entire claim is that nothing
    reaches the optimizer without validating, an update path that bypasses the
    schema is the one place it must not be allowed to happen.

    Round-tripping through `model_validate` costs microseconds and means a
    revised intent is exactly as trustworthy as a parsed one.
    """
    payload = intent.model_dump()
    for key, value in updates.items():
        if key.startswith("constraint:"):
            payload["constraints"][key.split(":", 1)[1]] = value
        else:
            payload[key] = value
    return OptimizationIntent.model_validate(payload)


def _apply_update(intent: OptimizationIntent, update: tuple[str, object]):
    target, value = update
    return revise(intent, **{target: value})


def _subject_of(question: str) -> str | None:
    lowered = question.lower()
    for phrase, field_name in QUESTION_SUBJECTS:
        if phrase in lowered:
            return field_name
    return None


def _numeric_answer(message: str) -> float | None:
    """"20%" or "20" -> 0.2. Anything else is not a number."""
    percent = PERCENT.search(message)
    if percent:
        return float(percent.group(1)) / 100.0
    bare = BARE_NUMBER.match(message)
    if bare:
        number = float(bare.group(1))
        # A bare number answering an SOC question is a percentage if it looks
        # like one, and a fraction if it looks like that. 0.2 and 20 are the
        # same answer; 200 is neither and is refused by the schema.
        return number / 100.0 if number > 1.0 else number
    return None


def _chosen_option(message: str, options: list[str]) -> str | None:
    lowered = message.lower().strip()
    for index, option in enumerate(options, start=1):
        if lowered in (str(index), f"{index}.", f"option {index}"):
            return option
        if option.lower() in lowered or lowered in option.lower():
            return option
    return None


def _partial_to_intent(question: Clarification) -> OptimizationIntent | None:
    if not question.partial_intent:
        return None
    try:
        return OptimizationIntent.model_validate(question.partial_intent)
    except Exception:
        return None


def ask(message: str, site: SiteConfig | None = None, **kwargs) -> CopilotResponse:
    """One-shot convenience for a single request."""
    return run_request(message, site=site, **kwargs)


__all__ = ["Conversation", "Rejection", "Turn", "ask"]
