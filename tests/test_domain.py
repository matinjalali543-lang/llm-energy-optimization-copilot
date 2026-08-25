"""The site and its vocabulary.

Two properties matter here. A `SiteConfig` that exists must be physically
coherent, and the vocabulary must be a **closed set** — the copilot's refusal to
invent assets rests on this file, not on the model's restraint.
"""

from __future__ import annotations

import pytest

from energy_copilot.domain.config import (
    Battery,
    FlexibleLoad,
    SiteConfig,
    available_days,
    day_profile,
    load_days,
    scale_to_site,
)
from energy_copilot.domain.ontology import (
    Asset,
    Objective,
    explain_absent_asset,
    resolve_asset,
    resolve_objective,
    resolve_quantity,
    vocabulary,
)

# --- the committed dataset -------------------------------------------------


def test_every_committed_day_has_24_hours():
    frame = load_days()
    counts = frame.groupby("day").size()
    assert (counts == 24).all(), counts[counts != 24].to_dict()


def test_the_dataset_spans_regimes_that_change_the_answer():
    """A set of comfortable days would flatter the optimizer."""
    frame = load_days()
    spreads = frame.groupby("day")["price_eur_mwh"].agg(lambda s: s.max() - s.min())
    assert spreads.max() > 100, "no day with a wide spread to arbitrage"
    assert spreads.min() < 10, "no flat day to show the optimizer earning little"
    assert (frame["price_eur_mwh"] < 0).any(), "no negative prices anywhere"


def test_asking_for_a_day_that_is_not_there_says_what_is():
    with pytest.raises(KeyError, match="Available days"):
        day_profile("1999-01-01")


def test_day_profile_is_indexed_zero_to_23():
    profile = day_profile(available_days()[0])
    assert list(profile["hour"]) == list(range(24))


# --- the site validates at construction ------------------------------------


def test_a_battery_cannot_start_outside_its_own_window():
    with pytest.raises(ValueError, match="outside its own window"):
        Battery(soc_min_frac=0.2, soc_max_frac=0.9, initial_soc_frac=0.05)


def test_a_battery_window_must_be_ordered():
    with pytest.raises(ValueError, match="0 <= min < max <= 1"):
        Battery(soc_min_frac=0.9, soc_max_frac=0.1)


def test_export_priced_above_import_is_refused_as_a_broken_tariff():
    """Not a clever dispatch: the optimum imports and exports in the same hour.

    Catching it in the config means the model needs no binary to forbid it.
    """
    with pytest.raises(ValueError, match="meter can be gamed"):
        SiteConfig(export_price_ratio=1.4)


def test_a_flexible_load_that_cannot_fit_its_own_window_is_refused():
    """Infeasible by construction, and far clearer here than as a solver code."""
    with pytest.raises(ValueError, match="infeasible by construction"):
        FlexibleLoad(energy_mwh=5.0, max_power_mw=0.1, earliest_hour=8, latest_hour=17)


def test_duplicate_flexible_load_names_are_refused():
    with pytest.raises(ValueError, match="duplicate flexible load"):
        SiteConfig(flexible_loads=(FlexibleLoad(name="a"), FlexibleLoad(name="a")))


def test_usable_energy_excludes_the_reserved_window():
    battery = Battery(energy_mwh=1.0, soc_min_frac=0.1, soc_max_frac=0.9)
    assert battery.usable_mwh == pytest.approx(0.8)


def test_describe_mentions_only_assets_that_exist():
    """The prompt is built from this. It must not advertise absent equipment."""
    described = SiteConfig(pv_peak_mw=0.0, battery=None, flexible_loads=()).describe()
    assert "PV" not in described
    assert "battery" not in described
    assert "grid" in described


# --- the vocabulary is closed ----------------------------------------------


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("bess", Asset.BATTERY), ("storage", Asset.BATTERY), ("batteries", Asset.BATTERY),
        ("solar", Asset.PV), ("photovoltaic", Asset.PV), ("panels", Asset.PV),
        ("utility", Asset.GRID), ("mains", Asset.GRID),
        ("shiftable load", Asset.FLEXIBLE_LOAD),
    ],
)
def test_synonyms_collapse_to_one_asset(term, expected):
    assert resolve_asset(term) is expected


@pytest.mark.parametrize(
    "term", ["battery charge", "charging", "charge battery", "charge power"]
)
def test_the_briefs_worked_example_resolves(term):
    """§8 names these four explicitly."""
    assert resolve_quantity(term).value == "battery_charge_power"


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("cheapest", Objective.MINIMIZE_COST),
        ("save money", Objective.MINIMIZE_COST),
        ("peak shaving", Objective.MINIMIZE_PEAK),
        ("self-consumption", Objective.MAXIMIZE_SELF_CONSUMPTION),
        ("co2", Objective.MINIMIZE_EMISSIONS),
        ("MINIMIZE_COST", Objective.MINIMIZE_COST),
    ],
)
def test_objectives_resolve_from_prose_and_from_the_enum(term, expected):
    assert resolve_objective(term) is expected


def test_an_unknown_term_resolves_to_nothing():
    """The refusal to invent assets rests on this returning None."""
    assert resolve_asset("flux capacitor") is None
    assert resolve_objective("maximise shareholder value") is None
    assert resolve_quantity("vibes") is None


@pytest.mark.parametrize(
    "term", ["diesel generator", "chp", "wind turbine", "ev charger", "heat pump"]
)
def test_plausible_but_absent_assets_get_a_reason_not_a_shrug(term):
    """'This site has no diesel generator' beats 'unknown term'."""
    assert resolve_asset(term) is None
    reason = explain_absent_asset(term)
    assert reason and len(reason) > 10


def test_vocabulary_is_generated_from_the_enums():
    """The prompt is built from this, so it cannot drift from the validator."""
    vocab = vocabulary()
    assert set(vocab["assets"]) == {asset.value for asset in Asset}
    assert set(vocab["objectives"]) == {objective.value for objective in Objective}


# --- scaling ---------------------------------------------------------------


def test_scaling_changes_pv_without_touching_price():
    """Scenario comparison depends on this: a bigger array is not a cheaper day.

    The committed data is derived at 0.60 MWp, so doubling means 1.20.
    """
    profile = day_profile(available_days()[0])
    scaled = scale_to_site(profile, SiteConfig(pv_peak_mw=1.20))
    assert scaled["pv_mw"].max() == pytest.approx(profile["pv_mw"].max() * 2, rel=1e-9)
    assert list(scaled["price_eur_mwh"]) == list(profile["price_eur_mwh"])
    assert scaled["load_mw"].to_list() == pytest.approx(profile["load_mw"].to_list())


def test_the_reference_site_needs_no_rescaling():
    profile = day_profile(available_days()[0])
    scaled = scale_to_site(profile, SiteConfig())
    assert scaled["load_mw"].to_list() == pytest.approx(profile["load_mw"].to_list())
