"""One record per request: what was asked, what was understood, what was done.

§23 lists the fields. The one that matters most is not on its list and is
recorded anyway: **which stage the request stopped at**. A log that says
"rejected" tells an operator nothing; a log that says the validator refused it
because the requested import exceeded the connection tells them what to change.

## What is deliberately not logged

§23 also says "do not log sensitive personal data", which in this system means
two concrete things rather than a policy sentence:

**No credentials, ever.** The API key is read from the environment by the
provider and never passes through a record. `sanitise` strips anything that
looks like one from free text as a second line of defence, because the first
line is only as good as every future edit to it.

**The request text is optional.** `include_request=False` keeps the shape of
what happened — stage, codes, timings, objective — without storing what anyone
typed. An operator's phrasing is not needed to debug a validator rejection, and
a log that keeps it forever is a liability the system does not need to carry.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

# Anything shaped like a key. Deliberately broad: a false positive costs a
# redacted string in a log, a false negative costs a leaked credential.
SECRET_PATTERN = re.compile(
    r"\b(sk-[A-Za-z0-9_-]{8,}|gho_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{12,}"
    r"|ANTHROPIC_API_KEY\s*=\s*\S+|api[_-]?key\s*[:=]\s*\S+)",
    re.IGNORECASE,
)
REDACTED = "[redacted]"


def sanitise(text: str | None) -> str | None:
    if text is None:
        return None
    return SECRET_PATTERN.sub(REDACTED, text)


@dataclass
class RequestRecord:
    """§23's fields, plus the stage that actually explains the outcome."""

    request_id: str
    stopped_at: str
    ok: bool
    reached_solver: bool
    parser: str
    request: str | None = None
    intent_summary: str | None = None
    objective: str | None = None
    horizon_day: str | None = None
    rejection_code: str | None = None
    rejection_reason: str | None = None
    validation_codes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    solver_termination: str | None = None
    objective_value: float | None = None
    solve_seconds: float | None = None
    total_seconds: float | None = None
    cost_eur: float | None = None
    baseline_cost_eur: float | None = None
    physics_problems: list[str] = field(default_factory=list)
    prompt_version: str | None = None
    llm_model: str | None = None
    factuality_score: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), default=str)


def record_response(
    response,
    *,
    parser: str = "rule_based",
    include_request: bool = True,
    total_seconds: float | None = None,
    prompt_version: str | None = None,
    llm_model: str | None = None,
    factuality_score: float | None = None,
) -> RequestRecord:
    """Turn a `CopilotResponse` into one loggable record."""
    from energy_copilot.schemas.intent import Rejection

    intent = response.intent
    rejection = response.outcome if isinstance(response.outcome, Rejection) else None
    validation = response.validation

    return RequestRecord(
        request_id=uuid.uuid4().hex[:12],
        stopped_at=response.stopped_at,
        ok=response.ok,
        reached_solver=response.reached_solver,
        parser=parser,
        request=sanitise(response.request) if include_request else None,
        intent_summary=intent.summary() if intent else None,
        objective=intent.objective.value if intent else None,
        horizon_day=intent.horizon.day if intent else None,
        rejection_code=rejection.code if rejection else None,
        rejection_reason=sanitise(rejection.reason) if rejection else None,
        validation_codes=[r.code for r in validation.rejections] if validation else [],
        warnings=list(response.notes),
        solver_termination=response.result.termination if response.result else None,
        objective_value=response.result.objective_value if response.result else None,
        solve_seconds=response.result.solve_seconds if response.result else None,
        total_seconds=total_seconds,
        cost_eur=response.after.cost_eur if response.after else None,
        baseline_cost_eur=response.before.cost_eur if response.before else None,
        physics_problems=list(response.physics_problems),
        prompt_version=prompt_version,
        llm_model=llm_model,
        factuality_score=factuality_score,
    )


@dataclass
class RequestLog:
    """An in-memory log, appendable to disk as JSON lines."""

    records: list[RequestRecord] = field(default_factory=list)
    path: Path | None = None
    include_request: bool = True

    def add(self, response, **kwargs) -> RequestRecord:
        kwargs.setdefault("include_request", self.include_request)
        record = record_response(response, **kwargs)
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.as_json() + "\n")
        return record

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.as_dict() for record in self.records])

    def stage_counts(self) -> dict[str, int]:
        """Where requests are ending up — the operational health question."""
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.stopped_at] = counts.get(record.stopped_at, 0) + 1
        return counts

    def summary(self) -> str:
        if not self.records:
            return "no requests recorded"
        counts = self.stage_counts()
        solved = [r for r in self.records if r.solve_seconds is not None]
        parts = [f"{len(self.records)} request(s)", f"stages {counts}"]
        if solved:
            mean = sum(r.solve_seconds for r in solved) / len(solved)
            parts.append(f"mean solve {mean * 1000:.0f} ms over {len(solved)}")
        return "; ".join(parts)


class timed:
    """Context manager giving wall-clock seconds for a request."""

    def __enter__(self):
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.seconds = time.perf_counter() - self._started
        return False
