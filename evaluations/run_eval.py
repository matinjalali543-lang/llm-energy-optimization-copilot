"""Score both parser arms on the same prompts.

    python evaluations/run_eval.py                 # rule-based only (no key needed)
    python evaluations/run_eval.py --arms both     # adds the LLM arm

The point of running both is that an accuracy figure for one arm alone answers
only "did it run". Scored against the same prompts and the same expectations,
the comparison answers **what the LLM adds over pattern matching** — which is
the question worth settling before putting a model anywhere near a control path.

## What is scored

`kind`     did it produce the right *shape* — intent, clarification, rejection
`detail`   when an intent was expected, is the objective (and any pinned
           constraint) right
`safety`   for prompts marked `reaches_solver: false`, did the request stop
           before anything was built

`safety` is reported separately and never averaged into the headline accuracy.
A system that refuses everything would score 100% on safety and be useless, so
mixing the two would let one hide the other.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_copilot.domain.config import SiteConfig  # noqa: E402
from energy_copilot.parsing.rule_based import parse as rule_based_parse  # noqa: E402
from energy_copilot.pipeline import run_request  # noqa: E402
from energy_copilot.schemas.intent import (  # noqa: E402
    Clarification,
    OptimizationIntent,
    Rejection,
)

HERE = Path(__file__).parent
DATASET = HERE / "dataset.yaml"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_dataset(path: Path = DATASET) -> list[dict]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate prompt id(s) in {path.name}: {duplicates}")
    return cases


@dataclass
class CaseScore:
    """How one arm did on one prompt."""

    id: str
    category: str
    request: str
    expected_kind: str
    got_kind: str
    kind_ok: bool
    detail_ok: bool | None
    safety_ok: bool | None
    stopped_at: str
    note: str = ""


@dataclass
class ArmReport:
    arm: str
    scores: list[CaseScore] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([score.__dict__ for score in self.scores])

    def summary(self) -> dict[str, float]:
        frame = self.frame()
        detail = frame["detail_ok"].dropna()
        safety = frame["safety_ok"].dropna()
        return {
            "arm": self.arm,
            "prompts": len(frame),
            "kind_accuracy": round(frame["kind_ok"].mean(), 4),
            "detail_accuracy": round(detail.mean(), 4) if len(detail) else float("nan"),
            "safety": round(safety.mean(), 4) if len(safety) else float("nan"),
            "safety_prompts": len(safety),
        }

    def by_category(self) -> pd.DataFrame:
        frame = self.frame()
        grouped = frame.groupby("category").agg(
            prompts=("id", "count"),
            kind_accuracy=("kind_ok", "mean"),
        )
        return grouped.round(4).reset_index()


def observed_kind(response) -> str:
    outcome = response.outcome
    if isinstance(outcome, Clarification):
        return "clarification"
    if isinstance(outcome, Rejection):
        return "rejection"
    if isinstance(outcome, OptimizationIntent):
        # An intent the validator refused is not a usable intent, whatever the
        # parser produced. Scoring it as one would credit the parser for output
        # the system then threw away.
        if response.validation is not None and not response.validation.ok:
            return "rejection"
        return "intent"
    return "none"


def check_detail(case: dict, response) -> tuple[bool | None, str]:
    """Only for prompts that expected an intent."""
    expect = case["expect"]
    if expect.get("kind") != "intent":
        return None, ""
    intent = response.intent
    if intent is None:
        return False, "no intent produced"

    problems = []
    if "objective" in expect and intent.objective.value != expect["objective"]:
        problems.append(f"objective {intent.objective.value} != {expect['objective']}")
    if "horizon_day" in expect and intent.horizon.day != expect["horizon_day"]:
        problems.append(f"day {intent.horizon.day} != {expect['horizon_day']}")
    if "shift_flexible_loads" in expect and (
        intent.shift_flexible_loads != expect["shift_flexible_loads"]
    ):
        problems.append("shift_flexible_loads mismatch")
    if "comparison_count" in expect and len(intent.comparison) != expect["comparison_count"]:
        problems.append(f"{len(intent.comparison)} variants != {expect['comparison_count']}")
    for name, wanted in (expect.get("constraints") or {}).items():
        actual = getattr(intent.constraints, name, None)
        if actual is None or abs(float(actual) - float(wanted)) > 1e-9:
            problems.append(f"{name} {actual} != {wanted}")
    return not problems, "; ".join(problems)


def score_case(case: dict, parser, site: SiteConfig) -> CaseScore:
    response = run_request(case["request"], site=site, parser=parser)
    expect = case["expect"]
    got = observed_kind(response)
    detail_ok, note = check_detail(case, response)

    safety_ok = None
    if expect.get("reaches_solver") is False:
        safety_ok = not response.reached_solver

    return CaseScore(
        id=case["id"],
        category=case["category"],
        request=case["request"],
        expected_kind=expect["kind"],
        got_kind=got,
        kind_ok=got == expect["kind"],
        detail_ok=detail_ok,
        safety_ok=safety_ok,
        stopped_at=response.stopped_at,
        note=note,
    )


def run_arm(name: str, parser, cases: list[dict], site: SiteConfig) -> ArmReport:
    report = ArmReport(arm=name)
    for case in cases:
        report.scores.append(score_case(case, parser, site))
    return report


def build_llm_parser(site: SiteConfig):
    """The LLM arm, if a provider can be had. Returns None when it cannot."""
    from energy_copilot.llm.provider import MissingRecording, RecordedProvider, default_provider
    from energy_copilot.parsing.llm_parser import LLMParser

    provider = default_provider(strict_recordings=True)
    if isinstance(provider, RecordedProvider) and provider.available == 0:
        return None, (
            "no API key and no recordings, so the LLM arm has never been run. "
            "Its accuracy is unmeasured -- not zero, unmeasured. Record with "
            "`python evaluations/record.py` once a key is set."
        )
    parser = LLMParser(provider=provider, site=site)

    def call(request: str, _site: SiteConfig):
        try:
            return parser.parse(request)
        except MissingRecording as exc:
            return Rejection(code="no_recording", reason=str(exc)[:400])

    return call, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", choices=("rule", "both"), default="rule")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    cases = load_dataset()
    site = SiteConfig()
    reports = [run_arm("rule_based", rule_based_parse, cases, site)]

    if args.arms == "both":
        llm_parser, why_not = build_llm_parser(site)
        if llm_parser is None:
            print(f"LLM arm skipped: {why_not}\n")
        else:
            reports.append(run_arm("llm", llm_parser, cases, site))

    args.out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([report.summary() for report in reports])
    print(f"{len(cases)} prompts, {len({c['category'] for c in cases})} categories\n")
    print(summary.to_string(index=False))

    for report in reports:
        print(f"\n--- {report.arm} by category ---")
        print(report.by_category().to_string(index=False))
        frame = report.frame()
        frame.to_csv(args.out / f"eval_{report.arm}.csv", index=False)

        wrong = frame[~frame["kind_ok"]]
        if not wrong.empty:
            print(f"\n{report.arm}: {len(wrong)} wrong shape")
            for _, row in wrong.iterrows():
                print(
                    f"  {row['id']:<16} expected {row['expected_kind']:<13} "
                    f"got {row['got_kind']:<13} {row['request'][:44]}"
                )
        detail_wrong = frame[frame["detail_ok"] == False]  # noqa: E712
        if not detail_wrong.empty:
            print(f"\n{report.arm}: {len(detail_wrong)} right shape, wrong detail")
            for _, row in detail_wrong.iterrows():
                print(f"  {row['id']:<16} {row['note'][:70]}")

    summary.to_csv(args.out / "eval_summary.csv", index=False)
    print(f"\nwrote {args.out / 'eval_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
