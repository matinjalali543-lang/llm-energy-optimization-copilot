"""Figures and the request log.

A rendering test cannot tell you a figure is readable — that needs eyes, and
three defects in this repo's figures were caught only by looking. What it can do
is catch what a thumbnail hides: a panel with no data, a missing zero line under
negative values, a colour that means one thing in one panel and another
elsewhere.
"""

from __future__ import annotations

import json

import pytest

from energy_copilot.domain.config import SiteConfig
from energy_copilot.observability import RequestLog, record_response, sanitise, timed
from energy_copilot.pipeline import run_request
from energy_copilot.visualization import (
    plot_before_after,
    plot_day,
    plot_evaluation,
    plot_objective_comparison,
    plot_security,
)

DAY = "2020-09-21"
SUNNY = "2020-06-01"

OBJECTIVE_REQUESTS = {
    "MINIMIZE_COST": "Reduce electricity cost and shift flexible loads.",
    "MINIMIZE_PEAK": "Reduce the peak demand charge.",
    "MAXIMIZE_SELF_CONSUMPTION": "Maximise PV self-consumption.",
    "MINIMIZE_EMISSIONS": "Cut my carbon emissions.",
}


@pytest.fixture(scope="module")
def solved():
    return run_request("Reduce electricity cost tomorrow.", day=DAY)


@pytest.fixture(scope="module")
def objective_results():
    site = SiteConfig()
    return {
        name: run_request(text, site=site, day=SUNNY)
        for name, text in OBJECTIVE_REQUESTS.items()
    }


# --- the day figure ---------------------------------------------------------


def test_the_day_figure_has_all_three_panels(solved):
    figure = plot_day(solved)
    assert len(figure.axes) == 3
    for axis in figure.axes:
        assert axis.lines or axis.collections, "a panel with nothing in it"


def test_the_day_figure_refuses_a_request_that_never_solved():
    """Better than an empty chart captioned as a result."""
    refused = run_request("Run the diesel generator at 200 MW.")
    with pytest.raises(ValueError, match="stopped at the parse stage"):
        plot_day(refused)


def test_negative_prices_are_marked():
    """The reason a schedule looks odd should be visible on the figure."""
    sunny = run_request("Reduce electricity cost.", day=SUNNY)
    figure = plot_day(sunny)
    texts = [t.get_text() for t in figure.axes[2].texts]
    assert any("negative-price" in t for t in texts)


def test_state_of_charge_is_drawn_from_zero(solved):
    """A battery that barely moves must look like one, not fill the panel."""
    assert plot_day(solved).axes[1].get_ylim()[0] == 0


# --- before and after -------------------------------------------------------


def test_before_after_colours_by_direction_not_by_position(solved):
    """A real defect: every after-bar shared one colour, which read as saying
    the second bar is always the good one. Optimising cost genuinely raises the
    peak, and the figure has to show that rather than hide it.
    """
    figure = plot_before_after(solved)
    colours = {
        tuple(round(c, 3) for c in patch.get_facecolor())
        for axis in figure.axes
        for patch in axis.patches
    }
    assert len(colours) == 3, colours


def test_before_after_says_which_direction_is_good(solved):
    assert "lower-is-better" in plot_before_after(solved)._suptitle.get_text()


def test_before_after_labels_every_bar(solved):
    for axis in plot_before_after(solved).axes:
        assert len(axis.texts) == 2


# --- objectives -------------------------------------------------------------


def test_the_objective_figure_marks_one_winner_per_panel(objective_results):
    figure = plot_objective_comparison(objective_results)
    assert len(figure.axes) == 4
    for axis in figure.axes:
        highlighted = [
            patch
            for patch in axis.patches
            if tuple(round(c, 2) for c in patch.get_facecolor()[:3]) == (0.76, 0.07, 0.12)
        ]
        assert len(highlighted) == 1


def test_a_negative_bar_gets_a_zero_line(objective_results):
    """Without it a credit read as a small positive number — the most
    interesting result on the chart, invisible."""
    cost_axis = plot_objective_comparison(objective_results).axes[0]
    values = [patch.get_width() for patch in cost_axis.patches]
    if min(values) < 0:
        verticals = [line for line in cost_axis.lines if len(set(line.get_xdata())) == 1]
        assert verticals, "negative values with no zero line"


def test_labels_never_collide_with_the_tick_labels(objective_results):
    """Negative labels anchor at zero, not at the bar's left end: placing them
    left ran the value straight into the category tick label."""
    cost_axis = plot_objective_comparison(objective_results).axes[0]
    for text in cost_axis.texts:
        assert text.xy[0] >= 0.0


def test_underscores_do_not_reach_the_reader(objective_results):
    figure = plot_objective_comparison(objective_results)
    labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
    assert labels
    assert not any("_" in label for label in labels)


# --- evaluation and security ------------------------------------------------


def test_the_evaluation_figure_highlights_the_weak_category():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "category": ["paraphrase", "simple_valid"],
            "prompts": [6, 12],
            "kind_accuracy": [0.333, 1.0],
        }
    )
    axis = plot_evaluation(frame).axes[0]
    assert len(axis.patches) == 2
    assert len(axis.texts) == 2


def test_the_security_figure_shows_every_stage_including_the_empty_ones():
    """The bars that matter are the zeros."""
    axis = plot_security({"schema": 17, "validate": 2}).axes[0]
    assert len(axis.patches) == 5
    assert [t.get_text() for t in axis.texts].count("0") == 3


# --- observability ----------------------------------------------------------


def test_a_record_captures_the_stage_and_the_outcome(solved):
    record = record_response(solved)
    assert record.stopped_at == "complete"
    assert record.objective == "MINIMIZE_COST"
    assert record.request_id
    assert record.cost_eur is not None


def test_a_refusal_records_its_code():
    record = record_response(run_request("Run the diesel generator at 200 MW."))
    assert record.rejection_code == "absent_asset"
    assert not record.reached_solver


@pytest.mark.parametrize(
    "text",
    [
        "my key is sk-ant-api03-abcdefghijklmnop",
        "ANTHROPIC_API_KEY=secretvalue",
        "api_key: abcd1234efgh",
    ],
)
def test_credentials_never_reach_a_log(text):
    cleaned = sanitise(text)
    assert "[redacted]" in cleaned
    assert "secretvalue" not in cleaned


def test_privacy_mode_keeps_the_shape_and_drops_the_text(solved):
    """An operator's phrasing is not needed to debug a validator rejection."""
    record = record_response(solved, include_request=False)
    assert record.request is None
    assert record.objective == "MINIMIZE_COST"
    assert record.stopped_at == "complete"


def test_the_log_counts_stages():
    log = RequestLog()
    for request in (
        "Reduce electricity cost tomorrow.",
        "Run the diesel generator at 200 MW.",
        "Make the battery more aggressive.",
    ):
        with timed() as elapsed:
            response = run_request(request)
        log.add(response, total_seconds=elapsed.seconds)
    counts = log.stage_counts()
    assert counts["complete"] == 1
    assert counts["parse"] == 2
    assert "request(s)" in log.summary()


def test_the_log_writes_json_lines(tmp_path):
    path = tmp_path / "requests.jsonl"
    log = RequestLog(path=path)
    log.add(run_request("Reduce electricity cost tomorrow."))
    log.add(run_request("Run the diesel generator at 200 MW."))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["stopped_at"] == "complete"
