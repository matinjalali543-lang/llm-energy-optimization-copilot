"""Turning a solved result into something a person reads.

Two renderers, and the deterministic one is not a fallback — it is the control
arm, in the same way `rule_based.py` is the control arm for parsing.

**`describe(response)`** builds the explanation from the numbers directly. It is
grounded by construction: every figure it prints came out of a KPI dictionary a
line earlier, so it cannot invent one. It needs no key, no network and no model,
which means the whole pipeline produces a readable answer offline.

**`explain_with_model(response, provider)`** asks a model to write the same thing
more fluently, from the same facts. It is checked afterwards by
`factuality.check_factuality`, because fluency is exactly the quality that makes
a fabricated number hard to spot.

Both are given *numbers*, never the schedule. §14 is explicit — "Do not ask the
LLM to calculate financial KPIs" — and the reason is that a model asked to sum a
column returns something shaped like a sum.
"""

from __future__ import annotations

import pandas as pd

from energy_copilot.explain.factuality import FactualityReport, check_factuality, fact_sheet
from energy_copilot.llm.prompts import SYSTEM_PROMPT, render_explanation_prompt
from energy_copilot.llm.provider import LLMProvider


def describe(response) -> str:
    """A grounded explanation, assembled from the numbers themselves."""
    if response.outcome is None:
        return "Nothing was parsed from the request."

    from energy_copilot.schemas.intent import Clarification, Rejection

    if isinstance(response.outcome, Clarification):
        options = "; ".join(response.outcome.options)
        return f"{response.outcome.question} Options: {options}."
    if isinstance(response.outcome, Rejection):
        return f"That request was not run. {response.outcome.reason}"
    if response.validation is not None and not response.validation.ok:
        reasons = " ".join(rejection.reason for rejection in response.validation.rejections)
        return f"That request was not run. {reasons}"
    if response.result is None:
        return f"The request stopped at the {response.stopped_at} stage."

    result = response.result
    before, after, delta = response.before, response.after, response.delta
    schedule = result.schedule

    sentences: list[str] = []

    objective_phrase = {
        "MINIMIZE_COST": "minimising cost",
        "MINIMIZE_PEAK": "minimising peak grid import",
        "MAXIMIZE_SELF_CONSUMPTION": "maximising on-site use of the PV",
        "MINIMIZE_EMISSIONS": "minimising imported carbon",
    }.get(result.objective_name, result.objective_name)

    if delta.get("cost_became_credit"):
        sentences.append(
            f"Optimising for {objective_phrase} turns a bill of "
            f"{before.cost_eur:.2f} EUR into a credit of {abs(after.cost_eur):.2f} EUR."
        )
    else:
        sentences.append(
            f"Optimising for {objective_phrase} moves the day's cost from "
            f"{before.cost_eur:.2f} EUR to {after.cost_eur:.2f} EUR, a change of "
            f"{delta.get('cost_saving_eur', 0.0):.2f} EUR."
        )

    # Why the schedule looks the way it does, in physical terms.
    charged = schedule.loc[schedule["charge_mw"] > 1e-6]
    discharged = schedule.loc[schedule["discharge_mw"] > 1e-6]
    if not charged.empty and not discharged.empty:
        sentences.append(
            f"The battery charges in {len(charged)} hour(s) at an average price of "
            f"{charged['price_eur_mwh'].mean():.2f} EUR/MWh and discharges in "
            f"{len(discharged)} hour(s) at an average of "
            f"{discharged['price_eur_mwh'].mean():.2f} EUR/MWh."
        )
    elif result.battery_throughput_mwh <= 1e-6:
        sentences.append("The battery stays idle: nothing on this day pays for a cycle.")

    if result.objective_name == "MINIMIZE_PEAK":
        sentences.append(
            f"Peak grid import falls from {before.peak_mw:.3f} MW to "
            f"{after.peak_mw:.3f} MW."
        )
    elif after.peak_mw > before.peak_mw + 1e-6:
        sentences.append(
            f"Peak import rises from {before.peak_mw:.3f} MW to {after.peak_mw:.3f} MW, "
            "which is the cost of buying hard in the cheap hours."
        )

    if after.pv_curtailed_mwh > 1e-6 or before.pv_curtailed_mwh > 1e-6:
        sentences.append(
            f"PV curtailment goes from {before.pv_curtailed_mwh:.3f} MWh to "
            f"{after.pv_curtailed_mwh:.3f} MWh."
        )

    if result.objective_name == "MINIMIZE_EMISSIONS":
        sentences.append(
            f"Imported carbon falls from {before.emissions_kg:.1f} kg to "
            f"{after.emissions_kg:.1f} kg on the proxy intensity described in the docs."
        )

    for note in response.notes[:2]:
        sentences.append(note[0].upper() + note[1:] if note else note)

    return " ".join(sentences)


def result_facts_block(response) -> str:
    """The numbers handed to a model, one per line."""
    facts = fact_sheet(response)
    return "\n".join(f"{name}: {value}" for name, value in sorted(facts.items()))


def comparison_block(response) -> str:
    table: pd.DataFrame | None = response.comparison()
    if table is None:
        return "(no comparison available)"
    return table.to_string(index=False)


def explain_with_model(
    response, provider: LLMProvider, prompt_version: str = "result_explainer_v1"
) -> tuple[str, FactualityReport]:
    """Ask a model to write it, then check every number it used.

    The check is not advisory. A caller that ignores it has an explanation whose
    figures nobody verified, which is the situation this module exists to
    prevent — so the report comes back alongside the text rather than being
    logged somewhere.
    """
    intent = response.intent
    prompt = render_explanation_prompt(
        intent_summary=intent.summary() if intent else response.request,
        result_facts=result_facts_block(response),
        comparison=comparison_block(response),
        version=prompt_version,
    )
    text = provider.complete(SYSTEM_PROMPT, prompt.text)
    return text, check_factuality(text, fact_sheet(response))


def explain(response, provider: LLMProvider | None = None) -> tuple[str, FactualityReport]:
    """The explanation a caller should use: model when available, else grounded.

    The deterministic renderer scores 1.0 by construction, which is worth
    stating plainly rather than presenting as an achievement: it cannot invent a
    number because it only ever prints numbers it was handed.
    """
    if provider is None:
        text = describe(response)
        return text, check_factuality(text, fact_sheet(response))
    return explain_with_model(response, provider)
