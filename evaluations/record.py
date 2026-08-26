"""Record real model responses once, so CI can replay them for free.

    ANTHROPIC_API_KEY=... python evaluations/record.py

Every prompt in `dataset.yaml` is sent to the model once and the raw response is
written to `evaluations/recordings/`, keyed by a hash of the prompt. After that
the LLM arm runs anywhere — in CI, on a clone, with no key and no billing
account — and returns exactly what the model actually said.

This is the same move as project 03 committing the IEEE 33-bus CSVs: the
expensive, unreliable, credential-requiring step happens once and its output is
committed and auditable. A reviewer can read the recordings and see for
themselves what the model returned, which is not true of a number in a README.

The key is read from the environment, never written to a recording, and never
logged. Recordings contain the prompt hash, the model id, the prompt version and
the response text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_copilot.domain.config import SiteConfig  # noqa: E402
from energy_copilot.llm.prompts import SYSTEM_PROMPT, render_intent_prompt  # noqa: E402
from energy_copilot.llm.provider import AnthropicProvider, prompt_key  # noqa: E402

HERE = Path(__file__).parent
RECORDINGS = HERE / "recordings"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--out", type=Path, default=RECORDINGS)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Nothing was recorded, and nothing was "
            "invented in its place: the LLM arm stays unmeasured until a real "
            "model has actually answered these prompts."
        )
        return 1

    from run_eval import load_dataset

    cases = load_dataset()[: args.limit]
    provider = AnthropicProvider(model=args.model)
    site = SiteConfig()

    records = []
    for index, case in enumerate(cases, start=1):
        prompt = render_intent_prompt(case["request"], site)
        started = time.perf_counter()
        response = provider.complete(SYSTEM_PROMPT, prompt.text)
        records.append(
            {
                "key": prompt_key(SYSTEM_PROMPT, prompt.text),
                "id": case["id"],
                "category": case["category"],
                "request": case["request"],
                "model": args.model,
                "prompt_version": prompt.version,
                "response": response,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        print(f"  [{index}/{len(cases)}] {case['id']}")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"intent_{args.model}.json"
    path.write_text(
        json.dumps({"model": args.model, "records": records}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {path} ({len(records)} records)")
    print("Commit it: a recording that is not committed is a measurement nobody can check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
