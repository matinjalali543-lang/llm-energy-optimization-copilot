"""Figures.

**No backend is pinned here.** An earlier repo in this series called
`matplotlib.use("Agg")` inside a library module and every notebook figure came
out as the string `<Figure size 640x480>`. The backend belongs to whatever is
running the code; `tests/conftest.py` sets it for the test suite.

The figures answer three questions, and each is drawn only if it has something
to say:

* **What did the optimizer do all day** — dispatch and state of charge, against
  the price that drove it.
* **What did it change** — the same day with and without the optimizer, which is
  the §14 before/after in a form a reader can take in at a glance.
* **What did the benchmark find** — accuracy by category, with the weak category
  visible rather than averaged away.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

PRICE_COLOUR = "#4a4e69"
IMPORT_COLOUR = "#c1121f"
EXPORT_COLOUR = "#1b9e4b"
PV_COLOUR = "#e09f3e"
BATTERY_COLOUR = "#2a9d8f"
BASELINE_COLOUR = "#9aa3ad"


def plot_day(response, ax=None, figsize=(11, 7)):
    """Dispatch, state of charge and price for one solved day."""
    if response.result is None:
        raise ValueError(
            "nothing to plot: this request stopped at the "
            f"{response.stopped_at} stage and produced no schedule"
        )
    schedule = response.result.schedule
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True, height_ratios=[3, 2, 2]
    ) if ax is None else (ax.figure, [ax, ax, ax])

    hours = schedule["hour"]

    # --- power ---
    top = axes[0]
    top.step(hours, schedule["grid_import_mw"], where="mid", color=IMPORT_COLOUR,
             label="grid import", lw=1.8)
    if schedule["grid_export_mw"].max() > 1e-6:
        top.step(hours, -schedule["grid_export_mw"], where="mid", color=EXPORT_COLOUR,
                 label="grid export", lw=1.8)
    top.fill_between(hours, 0, schedule["pv_used_mw"], step="mid", color=PV_COLOUR,
                     alpha=0.45, label="PV used")
    curtailed = schedule["pv_available_mw"] - schedule["pv_used_mw"]
    if curtailed.max() > 1e-6:
        top.fill_between(hours, schedule["pv_used_mw"], schedule["pv_available_mw"],
                         step="mid", color=PV_COLOUR, alpha=0.15, hatch="//",
                         label="PV curtailed")
    top.step(hours, schedule["total_demand_mw"], where="mid", color="#22223b",
             ls="--", lw=1.3, label="demand")
    top.axhline(0, color="#adb5bd", lw=0.8)
    top.set_ylabel("MW")
    top.legend(ncol=3, fontsize=8, loc="upper left", framealpha=0.9)
    top.set_title(
        f"{response.result.objective_name} — cost {response.after.cost_eur:.2f} EUR "
        f"against a baseline of {response.before.cost_eur:.2f} EUR",
        loc="left", fontsize=11,
    )

    # --- state of charge ---
    middle = axes[1]
    middle.fill_between(hours, 0, schedule["soc_mwh"], step="mid",
                        color=BATTERY_COLOUR, alpha=0.35)
    middle.step(hours, schedule["soc_mwh"], where="mid", color=BATTERY_COLOUR, lw=1.6)
    middle.set_ylabel("SOC (MWh)")
    # Zero-based, so a battery that barely moves looks like a battery that
    # barely moves rather than filling the panel.
    middle.set_ylim(0, max(schedule["soc_mwh"].max() * 1.15, 1e-3))

    # --- price, which explains the rest ---
    bottom = axes[2]
    bottom.step(hours, schedule["price_eur_mwh"], where="mid", color=PRICE_COLOUR, lw=1.6)
    bottom.fill_between(hours, 0, schedule["price_eur_mwh"], step="mid",
                        color=PRICE_COLOUR, alpha=0.15)
    if (schedule["price_eur_mwh"] < 0).any():
        bottom.axhline(0, color=IMPORT_COLOUR, lw=1.0, ls=":")
        negative = int((schedule["price_eur_mwh"] < 0).sum())
        bottom.annotate(
            f"{negative} negative-price hour(s)",
            xy=(0.99, 0.06), xycoords="axes fraction", ha="right", fontsize=8,
            color=IMPORT_COLOUR,
        )
    bottom.set_ylabel("EUR/MWh")
    bottom.set_xlabel("hour")
    bottom.set_xlim(hours.min(), hours.max())

    fig.tight_layout()
    return fig


def plot_before_after(response, figsize=(9, 5)):
    """The §14 comparison as paired bars.

    Metrics are on separate axes because they are in different units and
    different orders of magnitude: emissions in kilograms would flatten cost in
    euros to a line, and a reader would conclude nothing changed.
    """
    if response.before is None or response.after is None:
        raise ValueError("no before/after available for this response")

    metrics = [
        ("Cost (EUR)", response.before.cost_eur, response.after.cost_eur),
        ("Peak (MW)", response.before.peak_mw, response.after.peak_mw),
        ("Import (MWh)", response.before.grid_import_mwh, response.after.grid_import_mwh),
        ("Emissions (kg)", response.before.emissions_kg, response.after.emissions_kg),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    improved_any = False
    worsened_any = False
    for axis, (label, before, after) in zip(axes, metrics, strict=True):
        # Colour the "after" bar by DIRECTION, not by being the after bar.
        # Every metric here is lower-is-better, and optimising cost genuinely
        # raises the peak — buying hard in the cheap hours is what that costs.
        # Painting every "after" bar the same colour implied the second bar was
        # always the good one and hid the trade-off the figure exists to show.
        improved = after <= before + 1e-12
        improved_any |= improved
        worsened_any |= not improved
        bars = axis.bar(
            ["before", "after"], [before, after],
            color=[BASELINE_COLOUR, EXPORT_COLOUR if improved else IMPORT_COLOUR],
            width=0.6,
        )
        axis.set_title(label, fontsize=9, loc="left")
        axis.tick_params(labelsize=8)
        span = max(abs(before), abs(after), 1e-9)
        for bar, value in zip(bars, (before, after), strict=True):
            axis.annotate(
                f"{value:,.2f}" if span < 100 else f"{value:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3 if value >= 0 else -11),
                textcoords="offset points", ha="center", fontsize=8,
            )
        if min(before, after) < 0:
            axis.axhline(0, color="#22223b", lw=0.8)
        axis.margins(y=0.22)
    note = "every metric is lower-is-better"
    if improved_any and worsened_any:
        note += "; green improved, red got worse"
    fig.suptitle(
        f"before and after — {response.result.objective_name}  ({note})",
        fontsize=10, x=0.02, ha="left",
    )
    fig.tight_layout()
    return fig


def plot_objective_comparison(results: dict[str, object], figsize=(9, 5)):
    """The four objectives on the same day, each on the metric it optimises.

    Drawn as four panels rather than one grouped chart because the quantities
    share no unit. The point the figure has to make is that the winner of each
    panel is a different arm, and a grouped chart in mixed units buries it.
    """
    metrics = [
        ("cost (EUR)", lambda r: r.after.cost_eur, True),
        ("peak (MW)", lambda r: r.after.peak_mw, True),
        ("self-consumed (MWh)", lambda r: r.after.self_consumed_mwh, False),
        ("emissions (kg)", lambda r: r.after.emissions_kg, True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    names = list(results)
    short = [
        n.replace("MINIMIZE_", "min ").replace("MAXIMIZE_", "max ").replace("_", " ").lower()
        for n in names
    ]

    for axis, (label, getter, lower_is_better) in zip(axes, metrics, strict=True):
        values = [getter(results[name]) for name in names]
        best = min(values) if lower_is_better else max(values)
        colours = [IMPORT_COLOUR if v == best else BASELINE_COLOUR for v in values]
        axis.barh(range(len(names)), values, color=colours, height=0.6)
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(short, fontsize=7)
        axis.set_title(label, fontsize=9, loc="left")
        axis.tick_params(labelsize=7)

        # A zero line whenever anything is negative. Without it the cost panel
        # showed the winning arm as a thin sliver that read as a small positive
        # number -- when it was -19.81 EUR, a credit, and the single most
        # interesting result on the chart.
        if min(values) < 0:
            axis.axvline(0, color="#22223b", lw=0.9)

        # Values on the bars. The panels have different units and different
        # magnitudes, so the shapes alone tell a reader almost nothing.
        span = max(abs(v) for v in values) or 1.0
        for index, value in enumerate(values):
            # Anchored at max(value, 0), never at the bar's left end. Placing a
            # negative bar's label to its left ran it straight into the y-tick
            # label, so "min cost" and "-7.29" printed on top of each other.
            axis.annotate(
                f"{value:,.2f}" if span < 100 else f"{value:,.0f}",
                xy=(max(value, 0.0), index),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=7,
            )
        axis.margins(x=0.30)
        axis.invert_yaxis()
    fig.suptitle(
        "each objective wins on its own metric (winner highlighted)",
        fontsize=11, x=0.02, ha="left",
    )
    fig.tight_layout()
    return fig


def plot_evaluation(by_category: pd.DataFrame, figsize=(8, 4.5)):
    """Benchmark accuracy per category, weakest first."""
    frame = by_category.sort_values("kind_accuracy")
    fig, axis = plt.subplots(figsize=figsize)
    colours = [
        IMPORT_COLOUR if value < 0.8 else BASELINE_COLOUR
        for value in frame["kind_accuracy"]
    ]
    bars = axis.barh(frame["category"], frame["kind_accuracy"], color=colours, height=0.62)
    for bar, (_, row) in zip(bars, frame.iterrows(), strict=True):
        axis.annotate(
            f"{row['kind_accuracy']:.0%}  (n={row['prompts']})",
            xy=(row["kind_accuracy"], bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
        )
    axis.set_xlim(0, 1.28)
    axis.set_xlabel("share parsed to the right shape")
    axis.set_title(
        "rule-based arm by category — paraphrase is the documented weakness",
        fontsize=11, loc="left",
    )
    axis.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_security(stage_counts: dict[str, int], figsize=(8, 3.2)):
    """Where hostile payloads died. The bar that matters is the empty one."""
    order = ["parse", "schema", "validate", "solve", "complete"]
    values = [stage_counts.get(stage, 0) for stage in order]
    fig, axis = plt.subplots(figsize=figsize)
    colours = ["#adb5bd", BATTERY_COLOUR, BATTERY_COLOUR, IMPORT_COLOUR, IMPORT_COLOUR]
    bars = axis.bar(order, values, color=colours, width=0.6)
    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            str(value), xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9,
        )
    axis.set_ylabel("hostile payloads")
    axis.set_title(
        "where a compromised model's payloads stopped — none reached the solver",
        fontsize=11, loc="left",
    )
    axis.margins(y=0.25)
    axis.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
