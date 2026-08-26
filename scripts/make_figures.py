"""Render every committed figure.

    python scripts/make_figures.py

Every figure in the README comes from here, so a number in the text and a number
in a picture cannot drift apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # a script may pin the backend; a library may not.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluations"))

from energy_copilot.domain.config import SiteConfig  # noqa: E402
from energy_copilot.parsing.rule_based import parse as rule_based_parse  # noqa: E402
from energy_copilot.pipeline import run_request  # noqa: E402
from energy_copilot.visualization import (  # noqa: E402
    plot_before_after,
    plot_day,
    plot_evaluation,
    plot_objective_comparison,
    plot_security,
)

FIGURES = ROOT / "figures"
WIDE_DAY = "2020-09-21"
SUNNY_DAY = "2020-06-01"


def save(fig, name: str, out: Path) -> None:
    path = out / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    site = SiteConfig()

    print("dispatch")
    cost = run_request("Reduce electricity cost tomorrow.", site=site, day=WIDE_DAY)
    save(plot_day(cost), "dispatch_cost.png", args.out)
    save(plot_before_after(cost), "before_after.png", args.out)

    sunny = run_request("Reduce electricity cost.", site=site, day=SUNNY_DAY)
    save(plot_day(sunny), "dispatch_negative_prices.png", args.out)

    print("objectives")
    requests = {
        "MINIMIZE_COST": "Reduce electricity cost.",
        "MINIMIZE_PEAK": "Reduce the peak demand charge.",
        "MAXIMIZE_SELF_CONSUMPTION": "Maximise PV self-consumption.",
        "MINIMIZE_EMISSIONS": "Cut my carbon emissions.",
    }
    results = {
        name: run_request(text, site=site, day=SUNNY_DAY)
        for name, text in requests.items()
    }
    save(plot_objective_comparison(results), "objectives.png", args.out)

    print("evaluation")
    from run_eval import load_dataset, run_arm

    report = run_arm("rule_based", rule_based_parse, load_dataset(), site)
    save(plot_evaluation(report.by_category()), "evaluation.png", args.out)

    print("security")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_security import HOSTILE_PAYLOADS, compromised

    stages: dict[str, int] = {}
    for payload in HOSTILE_PAYLOADS.values():
        response = run_request(
            "reduce my electricity cost tomorrow", site=site,
            parser=compromised(payload, site),
        )
        stages[response.stopped_at] = stages.get(response.stopped_at, 0) + 1
    save(plot_security(stages), "security.png", args.out)
    print(f"  stage counts: {stages}")

    print("\nfigures written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
