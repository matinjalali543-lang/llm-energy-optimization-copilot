"""The LLM arm: same interface as the rule-based one, so both can be scored.

Both parsers take a request and return `OptimizationIntent | Clarification |
Rejection`. That shared signature is what makes the evaluation a comparison
rather than two separate reports.

## Everything the model returns is treated as hostile input

Not because the model is expected to be hostile, but because the pipeline must
behave identically whether it is or not. Concretely:

* The response is parsed as JSON, and a failure is a `Rejection` — never a
  retry with a more insistent prompt, and never a partial salvage.
* The JSON is fed to Pydantic, which forbids unknown keys and enforces every
  bound. A hallucinated field is an error naming the field.
* Nothing is executed. There is no `eval`, no `exec`, no attribute lookup driven
  by model output, and no code path where a returned string becomes a name.
* The `clarification` and `refusal` envelopes are the only alternative shapes
  accepted, and both are themselves validated.

## What happens when the model returns prose

Models wrap JSON in code fences, add a sentence of preamble, or explain
themselves afterwards. `_extract_json` handles the common cases, because
refusing a correct answer over a code fence would measure formatting rather than
comprehension — and this repository is trying to measure comprehension.

It does **not** try to repair malformed JSON. There is a real line between
"finding the object in a response that contains one" and "guessing what the
model meant", and repair is on the wrong side of it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from energy_copilot.domain.config import SiteConfig
from energy_copilot.llm.prompts import SYSTEM_PROMPT, render_intent_prompt
from energy_copilot.llm.provider import LLMProvider, MissingRecording, default_provider
from energy_copilot.schemas.intent import (
    Clarification,
    OptimizationIntent,
    ParseOutcome,
    Rejection,
)

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class LLMParser:
    """Parse with a model, and refuse anything that does not validate."""

    provider: LLMProvider | None = None
    site: SiteConfig | None = None
    prompt_version: str = "intent_parser_v1"

    def __post_init__(self) -> None:
        self.provider = self.provider or default_provider()
        self.site = self.site or SiteConfig()

    @property
    def name(self) -> str:
        return f"llm:{getattr(self.provider, 'name', 'unknown')}"

    def parse(self, request: str) -> ParseOutcome:
        if not request.strip():
            return Rejection(code="empty_request", reason="the request is empty")

        prompt = render_intent_prompt(request, self.site, self.prompt_version)
        try:
            raw = self.provider.complete(SYSTEM_PROMPT, prompt.text)
        except MissingRecording as exc:
            # Deliberately not swallowed into a rejection. A missing recording is
            # a gap in the measurement setup, not a property of the request, and
            # quietly scoring it as "the model refused" would be a lie about
            # what happened.
            raise exc
        except Exception as exc:  # pragma: no cover - provider-specific
            return Rejection(
                code="provider_error",
                reason=f"the model could not be reached: {type(exc).__name__}",
            )

        return interpret_response(raw, request=request)


def interpret_response(raw: str, request: str = "") -> ParseOutcome:
    """Turn a model's raw text into a validated outcome, or refuse it.

    Separated from the provider call so the security tests can drive it directly
    with an attacker's payload and no model at all.
    """
    payload = _extract_json(raw)
    if payload is None:
        return Rejection(
            code="unparseable_response",
            reason=(
                "the model did not return JSON. Refusing rather than guessing: "
                "a repaired response measures the repair, not the model."
            ),
            offending_value=raw[:200],
        )

    if not isinstance(payload, dict):
        return Rejection(
            code="unparseable_response",
            reason=f"expected a JSON object, got {type(payload).__name__}",
            offending_value=str(payload)[:200],
        )

    # The two alternative envelopes, each validated in its own right.
    if "clarification" in payload:
        body = payload["clarification"]
        if body is None:
            return Rejection(
                code="empty_clarification",
                reason="the model signalled no clarification was needed but "
                "returned no intent either",
            )
        try:
            return Clarification.model_validate(body)
        except ValidationError as exc:
            return Rejection(
                code="invalid_clarification",
                reason=_first_error(exc),
                offending_value=str(body)[:200],
            )

    if "refusal" in payload:
        body = payload["refusal"] or {}
        return Rejection(
            code="model_refused",
            reason=str(body.get("reason", "the model declined the request"))[:500],
            offending_value=request[:200] or None,
        )

    try:
        return OptimizationIntent.model_validate(payload)
    except ValidationError as exc:
        return Rejection(
            code="schema_rejected",
            reason=_first_error(exc),
            offending_value=json.dumps(payload)[:200],
        )


def _extract_json(raw: str) -> object | None:
    """Find the JSON object in a response that may be wrapped in prose."""
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    fenced = FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # A model that wrote a sentence before the object. Take the outermost
    # brace-delimited span and try that -- once, without repair.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _first_error(exc: ValidationError) -> str:
    """The first validation failure, phrased for a person."""
    errors = exc.errors()
    if not errors:  # pragma: no cover - pydantic always reports at least one
        return str(exc)[:400]
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "(root)"
    return f"{location}: {first.get('msg', 'invalid')}"[:400]
