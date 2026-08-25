"""Versioned prompts, loaded from files and never written inline.

§19 asks for versioned prompt files and says: "Never rely on an undocumented
system prompt." Two consequences that are easy to skip and are implemented here:

**The version travels with the result.** Every parse and every explanation
records which prompt file produced it, so a change in behaviour between two runs
can be attributed rather than guessed at. A prompt is as much a part of the
system's configuration as a solver tolerance.

**The schema in the prompt is generated.** `intent_parser_v1.txt` has a
`{schema}` placeholder filled from `OptimizationIntent.model_json_schema()` at
render time. A prompt that described the schema in prose would drift from the
code the moment a field changed, and the resulting failures would look like
model errors while being documentation errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from energy_copilot.domain.config import SiteConfig
from energy_copilot.domain.ontology import Objective
from energy_copilot.schemas.intent import intent_json_schema

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

INTENT_PARSER = "intent_parser_v1"
RESULT_EXPLAINER = "result_explainer_v1"
CLARIFICATION = "clarification_v1"


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt with its identity attached."""

    version: str
    text: str

    def __str__(self) -> str:
        return self.text


@cache
def load_template(version: str) -> str:
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))
        raise FileNotFoundError(
            f"no prompt '{version}' in {PROMPTS_DIR}; available: {available}"
        )
    return path.read_text(encoding="utf-8")


def objective_list() -> str:
    return "\n".join(f"- {objective.value}" for objective in Objective)


def render_intent_prompt(
    request: str, site: SiteConfig, version: str = INTENT_PARSER
) -> RenderedPrompt:
    """Build the parser prompt for one request.

    The site description is generated from the configuration, so the model is
    never told about an asset the site does not have — which removes the most
    obvious source of confident hallucination at the root rather than asking the
    model not to do it.
    """
    template = load_template(version)
    text = template.format(
        site_description=site.describe(),
        objectives=objective_list(),
        import_limit=site.grid_import_limit_mw,
        schema=json.dumps(intent_json_schema(), indent=2),
        request=request,
    )
    return RenderedPrompt(version=version, text=text)


def render_explanation_prompt(
    intent_summary: str,
    result_facts: str,
    comparison: str,
    version: str = RESULT_EXPLAINER,
) -> RenderedPrompt:
    template = load_template(version)
    text = template.format(
        intent_summary=intent_summary,
        result_facts=result_facts,
        comparison=comparison,
    )
    return RenderedPrompt(version=version, text=text)


def render_clarification_prompt(
    request: str, site: SiteConfig, version: str = CLARIFICATION
) -> RenderedPrompt:
    template = load_template(version)
    text = template.format(site_description=site.describe(), request=request)
    return RenderedPrompt(version=version, text=text)


SYSTEM_PROMPT = (
    "You are a component inside an energy optimization system. You interpret "
    "requests and describe results. You do not calculate, schedule, or decide "
    "what is physically possible; other parts of the system do that and will "
    "reject anything you emit that they cannot honour."
)


def available_versions() -> list[str]:
    return sorted(path.stem for path in PROMPTS_DIR.glob("*.txt"))
