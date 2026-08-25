"""Providers: everything that can produce text, including one that lies.

The protocol is deliberately tiny — a provider takes a system prompt and a user
prompt and returns a string. Nothing in this package can call a tool, execute
code, or reach the network on the model's behalf, because there is no method
through which it could.

## The four implementations, and why the hostile one is the important one

**`AnthropicProvider`** calls a real model. It is optional: the `anthropic`
package is an extra, and nothing in the test suite requires it.

**`RecordedProvider`** replays committed fixtures keyed by a hash of the prompt.
This is how a real model's answers stay auditable and reproducible without a key
and without cost — the same move as project 03 committing the network CSVs so
pandapower was not needed at runtime. It raises on an unknown prompt rather than
returning something plausible, because a silent miss would turn a measurement
into a guess.

**`ScriptedProvider`** returns queued strings. For tests that need a specific
malformed output.

**`CompromisedProvider`** returns whatever an attacker wants, on every request,
regardless of the prompt. It exists because the interesting security question is
not "does a well-behaved model decline?" — that tests the model's manners on the
day it was asked. The question worth answering is what happens when the model is
**entirely under the attacker's control**, and this is how that gets tested.

If the pipeline holds against `CompromisedProvider`, the safety of the system
does not depend on the model behaving. If it does not hold, no amount of prompt
wording fixes it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

RECORDINGS_DIR = Path(__file__).resolve().parents[3] / "evaluations" / "recordings"


@runtime_checkable
class LLMProvider(Protocol):
    """Text in, text out. There is deliberately nothing else."""

    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response."""
        ...


def prompt_key(system: str, user: str) -> str:
    """Stable id for a prompt pair, used to name recordings."""
    digest = hashlib.sha256()
    digest.update(system.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(user.encode("utf-8"))
    return digest.hexdigest()[:16]


class MissingRecording(KeyError):
    """A prompt was asked for that nobody recorded."""


@dataclass
class RecordedProvider:
    """Replay committed responses. No key, no network, no cost.

    `strict=True` is the default and is the point: an unknown prompt raises. A
    provider that quietly returned an empty string on a miss would let the
    evaluation report accuracy for prompts that were never actually run, which
    is worse than having no number at all.
    """

    directory: Path = RECORDINGS_DIR
    strict: bool = True
    name: str = "recorded"
    _cache: dict[str, dict] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self._load()

    def _load(self) -> None:
        self._cache = {}
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for record in payload.get("records", []):
                self._cache[record["key"]] = record

    @property
    def available(self) -> int:
        return len(self._cache)

    def complete(self, system: str, user: str) -> str:
        key = prompt_key(system, user)
        record = self._cache.get(key)
        if record is None:
            if self.strict:
                raise MissingRecording(
                    f"no recording for prompt {key}. Recordings live in "
                    f"{self.directory} and are produced by "
                    "`python evaluations/record.py` with an API key set. "
                    "Refusing to invent a response: an evaluation scored "
                    "against fabricated output measures nothing."
                )
            return ""
        return record["response"]

    def describe(self) -> dict[str, object]:
        models = sorted({record.get("model", "unknown") for record in self._cache.values()})
        return {"recordings": len(self._cache), "models": models}


@dataclass
class ScriptedProvider:
    """Returns queued strings in order. For tests with a specific output in mind."""

    responses: list[str] = field(default_factory=list)
    name: str = "scripted"
    calls: list[tuple[str, str]] = field(default_factory=list, repr=False)

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise RuntimeError("ScriptedProvider ran out of queued responses")
        return self.responses.pop(0)


@dataclass
class CompromisedProvider:
    """A model wholly under an attacker's control.

    It ignores the system prompt entirely. Whatever the prompt says about units,
    schemas, refusals or good behaviour is irrelevant — this provider returns the
    attacker's payload every time.

    Used by `tests/test_security.py` to establish that the deterministic layers
    hold on their own. Never exported for real use, and never constructed
    anywhere outside tests.
    """

    payload: str
    name: str = "compromised"
    calls: int = 0

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002 - ignoring is the point
        self.calls += 1
        return self.payload


@dataclass
class AnthropicProvider:
    """A real model. Optional, and never required by the test suite.

    The key is read from the environment and never from an argument, a file in
    the repository, or a prompt. It is not logged, not echoed, and not included
    in any recording.
    """

    model: str = "claude-sonnet-5"
    max_tokens: int = 1024
    temperature: float = 0.0
    name: str = "anthropic"

    def __post_init__(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. This provider is optional — the "
                "rule-based parser and the recorded provider both run without "
                "it, which is why the whole test suite passes with no key."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                'the anthropic package is an optional extra: pip install -e ".[llm]"'
            ) from exc
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - needs a key
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


def default_provider(strict_recordings: bool = True) -> LLMProvider:
    """What to use when nobody said.

    Prefers a real model when a key is present, falls back to replaying
    recordings. Never silently produces text from nowhere.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicProvider()
        except (ImportError, RuntimeError):
            pass
    return RecordedProvider(strict=strict_recordings)
