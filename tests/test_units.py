"""Units: does the layer refuse to guess?

These are short because the property under test is short. Every one of them is
really the same assertion — that an ambiguous input produces an error rather
than a plausible number — and that assertion is the only thing standing between
a misread "200" and a schedule that is wrong by a factor of a thousand.
"""

from __future__ import annotations

import pytest

from energy_copilot.domain.units import (
    Dimension,
    UnitError,
    base_unit,
    convert,
    normalise_fraction,
    parse_unit,
    to_base,
)


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("kW", "kW"), ("kw", "kW"), ("KW", "kW"), ("kilowatt", "kW"),
        ("MWh", "MWh"), ("megawatt hour", "MWh"), ("mwh", "MWh"),
        ("%", "percent"), ("pct", "percent"),
        ("€/kWh", "EUR/kWh"), ("ct/kWh", "ct/kWh"),
        ("gCO2/kWh", "g/kWh"),
    ],
)
def test_spellings_resolve_to_one_canonical_unit(spelling, canonical):
    assert parse_unit(spelling).name == canonical


def test_an_unknown_unit_raises_rather_than_defaulting():
    with pytest.raises(UnitError, match="unknown unit"):
        parse_unit("furlongs")


def test_a_missing_unit_raises_because_there_is_no_default():
    with pytest.raises(UnitError, match="no default unit"):
        parse_unit("")


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (250.0, "kW", "MW", 0.25),
        (0.25, "MW", "kW", 250.0),
        (1.0, "MWh", "kWh", 1000.0),
        (50.0, "EUR/MWh", "ct/kWh", 5.0),
        (20.0, "percent", "fraction", 0.2),
        (90.0, "minute", "hour", 1.5),
    ],
)
def test_conversions_are_exact(value, source, target, expected):
    assert convert(value, source, target) == pytest.approx(expected)


def test_conversion_round_trips():
    assert convert(convert(137.0, "kW", "MW"), "MW", "kW") == pytest.approx(137.0)


def test_power_cannot_become_energy():
    """The error that actually bites: a state of charge quoted in kW.

    Dividing by an assumed hour would make this 'work' and would put a
    factor-of-time error into everything downstream.
    """
    with pytest.raises(UnitError, match="different physical quantities"):
        convert(100.0, "kW", "kWh")


def test_price_cannot_become_power():
    with pytest.raises(UnitError, match="different physical quantities"):
        convert(50.0, "EUR/MWh", "MW")


def test_kg_per_mwh_equals_g_per_kwh():
    """Not a coincidence worth glossing: they are the same number."""
    assert convert(430.0, "kg/MWh", "g/kWh") == pytest.approx(430.0)


def test_to_base_uses_the_dataset_units():
    assert to_base(250.0, "kW") == pytest.approx(0.25)          # MW
    assert base_unit(Dimension.POWER) == "MW"
    assert base_unit(Dimension.ENERGY) == "MWh"
    assert base_unit(Dimension.PRICE) == "EUR/MWh"


# --- the fraction rule, which is where the silent corruption would have been ---


def test_a_fraction_with_a_unit_is_converted():
    assert normalise_fraction(20.0, "percent") == pytest.approx(0.2)
    assert normalise_fraction(0.2, "fraction") == pytest.approx(0.2)


def test_a_bare_fraction_in_range_is_taken_as_given():
    assert normalise_fraction(0.2) == pytest.approx(0.2)


def test_a_bare_value_above_one_raises_instead_of_being_rescaled():
    """The regression guard for a bug this module nearly shipped.

    The first version divided bare values above 1 by 100. That reads `20` as
    20% and looks helpful, until an impossible "SOC 150%" arrives as `1.5`: it
    would have been rescaled to `0.015`, passed the validator whose whole job
    was to reject it, and produced a schedule pinning the battery at one and a
    half percent. An out-of-range number with no unit has to be refused, not
    repaired.
    """
    with pytest.raises(UnitError, match="not a fraction"):
        normalise_fraction(1.5)
    with pytest.raises(UnitError, match="not a fraction"):
        normalise_fraction(20.0)


def test_an_impossible_percentage_survives_conversion_as_impossible():
    """150% must still be out of range afterwards, for the validator to catch."""
    assert normalise_fraction(150.0, "percent") == pytest.approx(1.5)
