"""Units, and the refusal to guess between them.

§8 of the brief says "Never silently mix units." That sounds like a formatting
rule and is really a safety rule: a copilot that reads "keep 200 below the peak"
and assumes kW when the user meant MW is off by a factor of a thousand, and
every downstream number — the schedule, the savings, the explanation — is
confidently wrong in a way no constraint will catch.

So conversion here is **total or it raises**. There is no default unit, no
"probably kW", and no silent pass-through of an unrecognised string.

## Dimensions, not just names

`kWh` and `kW` differ by more than a letter: one is energy and one is power, and
converting between them requires a duration that a parser does not have. They
are therefore separate dimensions, and asking for the conversion is an error
rather than a division by an assumed hour.

That distinction is the one that actually bites. "Charge the battery to 200 kW"
is a category error — a state of charge is energy — and catching it here means
the validator never has to guess what the user meant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Dimension(StrEnum):
    POWER = "power"
    ENERGY = "energy"
    PRICE = "price"            # currency per unit energy
    FRACTION = "fraction"      # 0..1, or a percentage on the way in
    TEMPERATURE = "temperature"
    DURATION = "duration"
    EMISSION_RATE = "emission_rate"   # mass per unit energy


@dataclass(frozen=True)
class Unit:
    """A unit, its dimension, and its size relative to that dimension's base."""

    name: str
    dimension: Dimension
    to_base: float

    def __post_init__(self) -> None:
        if self.to_base <= 0:
            raise ValueError(f"{self.name}: to_base must be > 0, got {self.to_base}")


# Base units, chosen to match the committed dataset so that the common path
# involves no conversion at all: MW, MWh, EUR/MWh, fraction, degC, hour, g/kWh.
_UNITS: tuple[Unit, ...] = (
    Unit("W", Dimension.POWER, 1e-6),
    Unit("kW", Dimension.POWER, 1e-3),
    Unit("MW", Dimension.POWER, 1.0),
    Unit("GW", Dimension.POWER, 1e3),
    Unit("Wh", Dimension.ENERGY, 1e-6),
    Unit("kWh", Dimension.ENERGY, 1e-3),
    Unit("MWh", Dimension.ENERGY, 1.0),
    Unit("GWh", Dimension.ENERGY, 1e3),
    Unit("EUR/kWh", Dimension.PRICE, 1e3),
    Unit("EUR/MWh", Dimension.PRICE, 1.0),
    Unit("ct/kWh", Dimension.PRICE, 10.0),
    Unit("fraction", Dimension.FRACTION, 1.0),
    Unit("percent", Dimension.FRACTION, 0.01),
    Unit("degC", Dimension.TEMPERATURE, 1.0),
    Unit("hour", Dimension.DURATION, 1.0),
    Unit("minute", Dimension.DURATION, 1.0 / 60.0),
    Unit("day", Dimension.DURATION, 24.0),
    Unit("g/kWh", Dimension.EMISSION_RATE, 1.0),
    Unit("kg/MWh", Dimension.EMISSION_RATE, 1.0),
)

# Spellings a person or a model might produce, mapped to the canonical name.
# Deliberately explicit: an unknown spelling must fail loudly rather than be
# pattern-matched into something plausible.
_ALIASES: dict[str, str] = {
    "w": "W", "watt": "W", "watts": "W",
    "kw": "kW", "kilowatt": "kW", "kilowatts": "kW",
    "mw": "MW", "megawatt": "MW", "megawatts": "MW",
    "gw": "GW", "gigawatt": "GW",
    "wh": "Wh", "watthour": "Wh", "watt-hour": "Wh",
    "kwh": "kWh", "kilowatthour": "kWh", "kilowatt-hour": "kWh", "kilowatt hour": "kWh",
    "mwh": "MWh", "megawatthour": "MWh", "megawatt-hour": "MWh", "megawatt hour": "MWh",
    "gwh": "GWh",
    "eur/kwh": "EUR/kWh", "€/kwh": "EUR/kWh", "euro/kwh": "EUR/kWh",
    "$/kwh": "EUR/kWh",  # see note in `parse_unit`
    "eur/mwh": "EUR/MWh", "€/mwh": "EUR/MWh", "euro/mwh": "EUR/MWh",
    "$/mwh": "EUR/MWh",
    "ct/kwh": "ct/kWh", "cent/kwh": "ct/kWh", "cents/kwh": "ct/kWh",
    "fraction": "fraction", "pu": "fraction", "p.u.": "fraction",
    "percent": "percent", "%": "percent", "pct": "percent",
    "degc": "degC", "c": "degC", "°c": "degC", "celsius": "degC",
    "hour": "hour", "hours": "hour", "h": "hour", "hr": "hour",
    "minute": "minute", "minutes": "minute", "min": "minute",
    "day": "day", "days": "day", "d": "day",
    "g/kwh": "g/kWh", "gco2/kwh": "g/kWh", "g co2/kwh": "g/kWh",
    "kg/mwh": "kg/MWh",
}

BY_NAME: dict[str, Unit] = {unit.name: unit for unit in _UNITS}

# The currency is a labelling question, not a conversion one. The dataset is in
# EUR and this project does no FX, so "$/kWh" is accepted as a *spelling* of the
# price dimension and the number is NOT converted. Anything else would invent an
# exchange rate. Callers who care should read `CURRENCY_NOTE`.
CURRENCY_NOTE = (
    "Prices are EUR. '$/kWh' is accepted as a spelling of the price dimension "
    "and its magnitude is used unchanged -- no FX conversion is performed, "
    "because inventing an exchange rate would corrupt every cost figure "
    "downstream."
)


class UnitError(ValueError):
    """Raised when a unit is unknown, or a conversion crosses dimensions."""


def parse_unit(text: str) -> Unit:
    """Canonical `Unit` for a spelling, or `UnitError`."""
    if not isinstance(text, str) or not text.strip():
        raise UnitError("no unit given; this project has no default unit by design")
    key = text.strip().lower()
    name = _ALIASES.get(key) or (text.strip() if text.strip() in BY_NAME else None)
    if name is None:
        raise UnitError(
            f"unknown unit {text!r}. Known: {sorted(BY_NAME)}. Refusing to guess, "
            "because a wrong guess between kW and MW is a factor of a thousand."
        )
    return BY_NAME[name]


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert, or raise. Never returns a value it is unsure about."""
    source, target = parse_unit(from_unit), parse_unit(to_unit)
    if source.dimension is not target.dimension:
        raise UnitError(
            f"cannot convert {source.name} ({source.dimension}) to {target.name} "
            f"({target.dimension}). These are different physical quantities; "
            "converting power to energy would need a duration this layer does "
            "not have, and guessing one hour is how a factor-of-time error gets "
            "in."
        )
    return value * source.to_base / target.to_base


def to_base(value: float, unit: str) -> float:
    """Convert to the dimension's base unit (MW, MWh, EUR/MWh, fraction, ...)."""
    parsed = parse_unit(unit)
    return value * parsed.to_base


def base_unit(dimension: Dimension) -> str:
    for unit in _UNITS:
        if unit.dimension is dimension and unit.to_base == 1.0:
            return unit.name
    raise UnitError(f"no base unit registered for {dimension}")


def normalise_fraction(value: float, unit: str | None = None) -> float:
    """A state of charge as a fraction in [0, 1] — converted, never guessed.

    With a unit this is an ordinary conversion: `20` + `percent` is `0.2`.

    **Without a unit, a value above 1 raises.** The first version of this
    function divided such values by 100, which reads `20` as 20% and looks
    helpful until you follow it through: a request for an impossible
    "SOC 150%" arriving as `1.5` would have been quietly rescaled to `0.015`,
    sailed through a validator whose job was to reject it, and produced a
    schedule holding the battery at one and a half percent.

    An out-of-range number with no unit is a request that has to be refused, not
    repaired. The module exists to refuse to guess; guessing here would have
    been the one place it mattered most.
    """
    if unit is not None:
        return convert(value, unit, "fraction")
    if value > 1.0:
        raise UnitError(
            f"{value} is not a fraction and carries no unit. Say "
            f"'{value} percent' if that is what you mean. Silently reading it "
            "as percent would turn an impossible request into a valid one "
            "instead of rejecting it."
        )
    return value
