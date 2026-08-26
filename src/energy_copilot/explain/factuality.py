"""Is every number in this explanation one the system actually computed?

§20 asks for "explanation factuality" as a measured quantity. The obvious way to
measure it is to ask a second model whether the first one was honest, which
replaces a question about arithmetic with a question about a second model's
opinion. This does something narrower and checkable instead:

    extract every number from the explanation
    require each one to appear in the structured result

That is not a full account of truthfulness — an explanation can be entirely
grounded and still misleading, and this check would pass it. What it does catch
is the failure that matters most here and is otherwise invisible: **a fabricated
figure formatted exactly like a real one.** "You saved 74 EUR" and "you saved
68.41 EUR" look equally authoritative on the page.

## Matching at the precision the text uses

A figure written as "68.41" must match a fact of 68.407252, because that is the
same number rounded for a reader. So a written number matches a fact when they
agree at the number of decimals the *text* used. Comparing exactly would fail
every well-formatted explanation; comparing loosely would let real errors past.

## What is allowed without being a fact

Hours (0-23) and small ordinals appear naturally — "charging in hour 3", "the
first two hours" — and demanding they be present in the KPI dictionary would
make the check unusable. They are allowed explicitly, and the allowance is
narrow enough to state: integers 0-24, and 100 when it is part of "100%".

## What this check does not catch, stated plainly

**Fabricated small integers.** "The battery cycled 12 times" passes, because 12
is inside the hour allowance. The check covers *magnitudes* — euros, megawatts,
kilograms, percentages — and does not police counts below 25. Narrowing the
allowance by looking for the word "hour" nearby was considered and rejected: it
would trade a stated limitation for a fragile heuristic that fails silently.

**Correct numbers assembled into a false claim.** An explanation can quote every
figure accurately and still imply the wrong cause. Nothing here reads meaning.

So a factuality score of 1.0 means "no invented magnitudes", not "true". That is
a narrower guarantee than the phrase suggests, which is exactly why it is
written down here instead of being left to the reader's assumption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numbers as a reader writes them: 1 234.5, -19.81, 68,41 is not attempted --
# this project writes English with a decimal point.
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Hour indices, small counts and percentage anchors. Deliberately short.
ALWAYS_ALLOWED = frozenset(float(n) for n in range(0, 25)) | {100.0}


@dataclass
class FactualityReport:
    """Which numbers were grounded, and which were invented."""

    grounded: list[float] = field(default_factory=list)
    ungrounded: list[float] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.ungrounded

    @property
    def score(self) -> float:
        """Share of numbers that were grounded. 1.0 when there are none."""
        return 1.0 if self.checked == 0 else len(self.grounded) / self.checked

    def report(self) -> str:
        if self.ok:
            return f"all {self.checked} number(s) grounded"
        return (
            f"{len(self.ungrounded)} of {self.checked} number(s) not found in the "
            f"result: {self.ungrounded}"
        )


def extract_numbers(text: str) -> list[float]:
    """Every number a reader would see, in order."""
    found = []
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group().replace(",", "")
        try:
            found.append(float(raw))
        except ValueError:  # pragma: no cover - the pattern guarantees a float
            continue
    return found


def _decimals(value: float, text_form: str) -> int:
    return len(text_form.split(".")[1]) if "." in text_form else 0


def matches_any(written: float, written_text: str, facts: list[float]) -> bool:
    """Does this written number correspond to a fact, at the text's precision?

    The test is **distance**, not equality of rounded forms, and getting this
    wrong marked three of sixty correct explanations as fabrications.

    The trap is double rounding. `fact_sheet` stores an average charge price of
    41.395; the renderer formats the unrounded mean and prints `41.39`; the
    checker formatted the stored 41.395 and got `41.40`. Both are faithful
    roundings of the same underlying number and they disagree, because a value
    was rounded twice at different precisions and a tie broke each way.

    Asking instead whether the written figure lies within half a unit of its
    last place sidesteps every tie-breaking convention, because that is the
    actual definition of "this is that number, rounded".
    """
    places = _decimals(written, written_text)
    # Half a unit in the last place, plus a hair for binary representation.
    tolerance = 0.5 * (10.0 ** -places) * (1.0 + 1e-9) + 1e-12
    for fact in facts:
        if abs(fact - written) <= tolerance:
            return True
        # The magnitude with the sign dropped: an explanation may write "a
        # saving of 19.81" for a cost of -19.81, which is correct English
        # rather than a fabrication.
        if abs(abs(fact) - abs(written)) <= tolerance:
            return True
    return False


def check_factuality(
    text: str,
    facts: dict[str, float] | list[float],
    extra_allowed: set[float] | None = None,
) -> FactualityReport:
    """Check an explanation against the numbers the system computed."""
    values = list(facts.values()) if isinstance(facts, dict) else list(facts)
    numeric_facts = [float(v) for v in values if isinstance(v, int | float)]
    allowed = ALWAYS_ALLOWED | (extra_allowed or set())

    report = FactualityReport()
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group().replace(",", "")
        try:
            written = float(raw)
        except ValueError:  # pragma: no cover
            continue
        report.checked += 1
        if written in allowed or matches_any(written, raw, numeric_facts):
            report.grounded.append(written)
        else:
            report.ungrounded.append(written)
    return report


def fact_sheet(response) -> dict[str, float]:
    """Every number an explanation of this response is allowed to use.

    Assembled from the structured result only. If a figure is not in here, an
    explanation that quotes it invented it — which is exactly what the check is
    for.
    """
    facts: dict[str, float] = {}
    if response.before is not None:
        facts.update({f"before_{k}": v for k, v in response.before.as_dict().items()})
    if response.after is not None:
        facts.update({f"after_{k}": v for k, v in response.after.as_dict().items()})
    facts.update(response.delta)

    result = response.result
    if result is not None:
        facts["objective_value"] = round(result.objective_value, 6)
        facts["solve_seconds"] = round(result.solve_seconds, 6)
        schedule = result.schedule
        facts["min_price"] = round(float(schedule["price_eur_mwh"].min()), 6)
        facts["max_price"] = round(float(schedule["price_eur_mwh"].max()), 6)
        facts["price_spread"] = round(facts["max_price"] - facts["min_price"], 6)
        facts["cheapest_hour"] = float(schedule["price_eur_mwh"].idxmin())
        facts["priciest_hour"] = float(schedule["price_eur_mwh"].idxmax())
        facts["negative_price_hours"] = float((schedule["price_eur_mwh"] < 0).sum())
        facts["max_soc_mwh"] = round(float(schedule["soc_mwh"].max()), 6)
        facts["min_soc_mwh"] = round(float(schedule["soc_mwh"].min()), 6)

        # Battery behaviour, which is what an operator actually asks about.
        #
        # These were missing at first, and the omission was caught by this
        # module checking `render.describe` -- whose docstring claimed to be
        # "grounded by construction" while quoting average charge and discharge
        # prices it had computed itself and put in no fact sheet. It scored
        # 0.78. The claim was wrong, not the checker, and a test now pins the
        # deterministic renderer at 1.0 so the claim cannot rot again.
        charged = schedule.loc[schedule["charge_mw"] > 1e-6]
        discharged = schedule.loc[schedule["discharge_mw"] > 1e-6]
        facts["charge_hours"] = float(len(charged))
        facts["discharge_hours"] = float(len(discharged))
        if not charged.empty:
            facts["avg_charge_price"] = round(float(charged["price_eur_mwh"].mean()), 6)
        if not discharged.empty:
            facts["avg_discharge_price"] = round(
                float(discharged["price_eur_mwh"].mean()), 6
            )
    return facts
