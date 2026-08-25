"""Canonical names, and the vocabulary the copilot is allowed to have.

§8 asks that "battery charge", "charging", "charge battery" and "charge power"
all resolve to one canonical name. That is the easy half. The half that matters
is the other direction: **a term that is not in here does not exist**, and the
parser must say so rather than inventing an asset.

That is what stops "run the diesel generator at 200 MW" from becoming a
plausible-looking intent about an asset the site does not have. The vocabulary
is a closed set, declared in one file, and the validator checks membership
rather than trusting the model to have restricted itself.

## Why a dictionary and not embeddings

A synonym table is auditable: you can read it, diff it, and say exactly what the
system will and will not accept. A nearest-neighbour lookup over embeddings
would map "diesel generator" to whichever asset happened to be closest in vector
space, with a confidence score, and would therefore *always* return something.
For a component whose job is to refuse unknown terms, always returning something
is the wrong failure mode.
"""

from __future__ import annotations

from enum import StrEnum


class Asset(StrEnum):
    """Every asset this copilot knows how to talk about. §6 keeps it small."""

    GRID = "grid"
    PV = "pv"
    BATTERY = "battery"
    LOAD = "load"
    FLEXIBLE_LOAD = "flexible_load"


class Objective(StrEnum):
    """§7's objective enum. Anything else is refused, not approximated."""

    MINIMIZE_COST = "MINIMIZE_COST"
    MINIMIZE_PEAK = "MINIMIZE_PEAK"
    MAXIMIZE_SELF_CONSUMPTION = "MAXIMIZE_SELF_CONSUMPTION"
    MINIMIZE_EMISSIONS = "MINIMIZE_EMISSIONS"


class Quantity(StrEnum):
    """Canonical variable names, per §8's worked example."""

    BATTERY_CHARGE_POWER = "battery_charge_power"
    BATTERY_DISCHARGE_POWER = "battery_discharge_power"
    BATTERY_STATE_OF_CHARGE = "battery_state_of_charge"
    GRID_IMPORT_POWER = "grid_import_power"
    GRID_EXPORT_POWER = "grid_export_power"
    PV_GENERATION = "pv_generation"
    PV_CURTAILMENT = "pv_curtailment"
    SITE_LOAD = "site_load"
    ELECTRICITY_PRICE = "electricity_price"
    CARBON_INTENSITY = "carbon_intensity"


ASSET_SYNONYMS: dict[str, Asset] = {
    "grid": Asset.GRID, "utility": Asset.GRID, "network": Asset.GRID,
    "mains": Asset.GRID, "import": Asset.GRID, "supply": Asset.GRID,
    "pv": Asset.PV, "solar": Asset.PV, "solar pv": Asset.PV, "panels": Asset.PV,
    "solar panels": Asset.PV, "array": Asset.PV, "photovoltaic": Asset.PV,
    "photovoltaics": Asset.PV, "rooftop solar": Asset.PV,
    "battery": Asset.BATTERY, "batteries": Asset.BATTERY, "bess": Asset.BATTERY,
    "storage": Asset.BATTERY, "energy storage": Asset.BATTERY,
    "battery storage": Asset.BATTERY, "accumulator": Asset.BATTERY,
    "load": Asset.LOAD, "demand": Asset.LOAD, "consumption": Asset.LOAD,
    "site load": Asset.LOAD, "base load": Asset.LOAD,
    "flexible load": Asset.FLEXIBLE_LOAD, "flexible loads": Asset.FLEXIBLE_LOAD,
    "shiftable load": Asset.FLEXIBLE_LOAD, "deferrable load": Asset.FLEXIBLE_LOAD,
    "flex": Asset.FLEXIBLE_LOAD, "shiftable": Asset.FLEXIBLE_LOAD,
}

QUANTITY_SYNONYMS: dict[str, Quantity] = {
    "battery charge": Quantity.BATTERY_CHARGE_POWER,
    "charging": Quantity.BATTERY_CHARGE_POWER,
    "charge battery": Quantity.BATTERY_CHARGE_POWER,
    "charge power": Quantity.BATTERY_CHARGE_POWER,
    "charge rate": Quantity.BATTERY_CHARGE_POWER,
    "battery discharge": Quantity.BATTERY_DISCHARGE_POWER,
    "discharging": Quantity.BATTERY_DISCHARGE_POWER,
    "discharge power": Quantity.BATTERY_DISCHARGE_POWER,
    "discharge rate": Quantity.BATTERY_DISCHARGE_POWER,
    "soc": Quantity.BATTERY_STATE_OF_CHARGE,
    "state of charge": Quantity.BATTERY_STATE_OF_CHARGE,
    "battery level": Quantity.BATTERY_STATE_OF_CHARGE,
    "charge level": Quantity.BATTERY_STATE_OF_CHARGE,
    "battery soc": Quantity.BATTERY_STATE_OF_CHARGE,
    "grid import": Quantity.GRID_IMPORT_POWER,
    "import power": Quantity.GRID_IMPORT_POWER,
    "drawn from the grid": Quantity.GRID_IMPORT_POWER,
    "grid export": Quantity.GRID_EXPORT_POWER,
    "export power": Quantity.GRID_EXPORT_POWER,
    "feed-in": Quantity.GRID_EXPORT_POWER,
    "pv generation": Quantity.PV_GENERATION,
    "solar generation": Quantity.PV_GENERATION,
    "solar output": Quantity.PV_GENERATION,
    "curtailment": Quantity.PV_CURTAILMENT,
    "pv curtailment": Quantity.PV_CURTAILMENT,
    "spill": Quantity.PV_CURTAILMENT,
    "load": Quantity.SITE_LOAD,
    "site load": Quantity.SITE_LOAD,
    "price": Quantity.ELECTRICITY_PRICE,
    "electricity price": Quantity.ELECTRICITY_PRICE,
    "tariff": Quantity.ELECTRICITY_PRICE,
    "carbon": Quantity.CARBON_INTENSITY,
    "carbon intensity": Quantity.CARBON_INTENSITY,
    "emissions": Quantity.CARBON_INTENSITY,
    "co2": Quantity.CARBON_INTENSITY,
}

OBJECTIVE_SYNONYMS: dict[str, Objective] = {
    "cost": Objective.MINIMIZE_COST,
    "minimize cost": Objective.MINIMIZE_COST,
    "minimise cost": Objective.MINIMIZE_COST,
    "reduce cost": Objective.MINIMIZE_COST,
    "cheapest": Objective.MINIMIZE_COST,
    "save money": Objective.MINIMIZE_COST,
    "bill": Objective.MINIMIZE_COST,
    "peak": Objective.MINIMIZE_PEAK,
    "minimize peak": Objective.MINIMIZE_PEAK,
    "peak demand": Objective.MINIMIZE_PEAK,
    "demand charge": Objective.MINIMIZE_PEAK,
    "peak shaving": Objective.MINIMIZE_PEAK,
    "self consumption": Objective.MAXIMIZE_SELF_CONSUMPTION,
    "self-consumption": Objective.MAXIMIZE_SELF_CONSUMPTION,
    "use my own solar": Objective.MAXIMIZE_SELF_CONSUMPTION,
    "autarky": Objective.MAXIMIZE_SELF_CONSUMPTION,
    "emissions": Objective.MINIMIZE_EMISSIONS,
    "carbon": Objective.MINIMIZE_EMISSIONS,
    "co2": Objective.MINIMIZE_EMISSIONS,
    "greenest": Objective.MINIMIZE_EMISSIONS,
}

# Assets a user may plausibly ask about that this site does not have. Naming
# them lets the validator say "this site has no diesel generator" instead of
# "unknown term", which is the difference between a useful refusal and a shrug.
KNOWN_ABSENT_ASSETS: dict[str, str] = {
    "diesel generator": "no dispatchable fossil generation is configured",
    "diesel": "no dispatchable fossil generation is configured",
    "generator": "no dispatchable fossil generation is configured",
    "genset": "no dispatchable fossil generation is configured",
    "chp": "no combined heat and power unit is configured",
    "wind": "no wind turbine is configured at this site",
    "wind turbine": "no wind turbine is configured at this site",
    "ev": "no EV charger is modelled; see the stretch goals in the brief",
    "ev charger": "no EV charger is modelled; see the stretch goals in the brief",
    "electric vehicle": "no EV charger is modelled",
    "hvac": "no thermal model is included in this project",
    "heat pump": "no thermal model is included in this project",
    "boiler": "no thermal model is included in this project",
    "electrolyser": "no hydrogen equipment is configured",
    "fuel cell": "no hydrogen equipment is configured",
}


def _lookup(term: str, table: dict):
    if not isinstance(term, str):
        return None
    return table.get(term.strip().lower())


def resolve_asset(term: str) -> Asset | None:
    return _lookup(term, ASSET_SYNONYMS)


def resolve_quantity(term: str) -> Quantity | None:
    return _lookup(term, QUANTITY_SYNONYMS)


def resolve_objective(term: str) -> Objective | None:
    """Resolve an objective, accepting the enum's own spelling as well."""
    if isinstance(term, str):
        upper = term.strip().upper()
        if upper in Objective.__members__:
            return Objective[upper]
    return _lookup(term, OBJECTIVE_SYNONYMS)


def explain_absent_asset(term: str) -> str | None:
    """Why a plausible-but-absent asset is not available here."""
    return _lookup(term, KNOWN_ABSENT_ASSETS)


def vocabulary() -> dict[str, list[str]]:
    """Everything the copilot accepts — for the prompt, and for the README.

    Generated from the tables rather than written out by hand, so the prompt
    given to the model cannot drift away from what the validator enforces. A
    prompt that advertises a capability the validator rejects is a prompt that
    manufactures failures.
    """
    return {
        "assets": [asset.value for asset in Asset],
        "objectives": [objective.value for objective in Objective],
        "quantities": [quantity.value for quantity in Quantity],
    }
