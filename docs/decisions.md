# Decisions made after the brief

`PROJECT_BRIEF.md` holds the brief as written, unchanged. This file records every
decision taken after it — including the places where the brief was overridden,
and the numbers that drove each choice. Nothing here was decided silently.

## A. The prompt is not the security boundary (§3, §21)

§21 asks for prompt-injection tests. The weak reading of that is to show a
well-behaved model declining a nasty request, and it proves close to nothing: it
tests the model's manners on the day it was asked, with the prompt that was in
front of it, at the temperature it happened to run at.

This repository takes the strong reading. `CompromisedProvider` is a test double
that returns **exactly what an attacker wants** — a state of charge of 150%, a
diesel generator that does not exist in the configuration, a negative price, an
embedded `os.system` call, a horizon of ten thousand hours. It models an LLM
that is entirely under the attacker's control, which is the only assumption
worth designing against.

Every one of those must be refused by the deterministic layers, with a reason.
If any survives to the solver, the architecture is wrong and no amount of prompt
wording fixes it.

The consequence, stated plainly: **the safety of this system does not depend on
the model behaving.** It depends on a strict schema and a validator that has
never heard of the model.

## B. Offline-first, so that "it works" is checkable (§4)

The brief allows OpenAI, Anthropic or a local model. The environment this was
built in has **no API key and no local model**, and CI has neither — nor should
CI ever hold a billing credential to run a test suite.

So the LLM is optional at every level:

- A provider abstraction with a real Anthropic implementation, a **recorded**
  provider that replays committed fixtures, and the hostile double above.
- A **rule-based parser** that is a genuine parser, not a mock.
- Every test passes, and the whole pipeline runs end to end, with no provider
  installed and no key set.

This is the same move as project 03 committing the IEEE 33-bus CSVs so that
pandapower was not needed at runtime: the expensive, unreliable dependency is
pushed to a one-off step whose output is committed and auditable.

**What this costs, stated rather than hidden:** until a key exists, the LLM
arm's accuracy against a real model is *unmeasured*. The README says so in those
words. A number will not be quoted for something that was never run.

## C. A rule-based parser as a scored control, not a stub

The evaluation runs both arms over the same 50+ prompts. Without the control
arm, an accuracy figure for the LLM answers only "did it run"; with it, the
question becomes **what does the LLM add over pattern matching** — the same
shape of question `forecast-driven-bess-dispatch` asked about forecasts.

It also means CI reports a real, reproducible accuracy number with no key.

## D. Carbon intensity is a proxy, and is labelled one (§12)

The emissions objective needs a carbon signal. The pinned OPSD snapshot carries
German load, solar and wind — and **no fossil generation by fuel**, so a true
fuel-mix carbon intensity is not derivable from it. That was checked before the
objective was designed, not after.

Rather than invent a plausible-looking series, carbon intensity is computed as a
documented function of **residual load** (load minus wind and solar) and is
labelled an assumption of this project everywhere it appears — the same
treatment project 03 gave its line ratings, which the source also did not
provide.

It is not presented as measured data, and the emissions objective's results are
therefore about the *mechanism*, not about grams of CO₂ in Germany.

## E. Data derived once from a pinned snapshot (§6)

Price, load and PV come from Open Power System Data, snapshot `2020-10-06`, the
same pinned release project 01 documented. A script derives a small slice; the
processed CSV is committed; nothing touches the network at runtime.

## F. Build sequencing

Built phase by phase with a commit each, verified locally, then pushed as a
complete history — matching the seven sibling repos.
