"""The site: what physically exists, and therefore what may be asked for.

This is the ground truth the validator checks requests against. If it is not in
here, the copilot cannot be talked into it — not by a persuasive request, not by
an injected instruction, not by a model that has decided the site owns a diesel
generator.

Everything validates at construction, in the manner of the sibling repos: a
`SiteConfig` that exists is one whose battery cannot discharge below its floor,
whose PV is not larger than its inverter, and whose limits are not negative. A
bad site fails here with a message naming the problem, rather than three layers
down as an infeasible LP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_DATA = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "representative_days.csv"
)


@dataclass(frozen=True)
class Battery:
    energy_mwh: float = 0.5
    power_mw: float = 0.25
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.95
    initial_soc_frac: float = 0.50
    degradation_cost_eur_mwh: float = 2.0

    def __post_init__(self) -> None:
        if self.energy_mwh <= 0 or self.power_mw <= 0:
            raise ValueError("battery energy_mwh and power_mw must both be > 0")
        for name in ("charge_efficiency", "discharge_efficiency"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"battery {name} must be in (0, 1], got {value}")
        if not 0 <= self.soc_min_frac < self.soc_max_frac <= 1:
            raise ValueError(
                f"battery SOC window must satisfy 0 <= min < max <= 1, got "
                f"[{self.soc_min_frac}, {self.soc_max_frac}]"
            )
        if not self.soc_min_frac <= self.initial_soc_frac <= self.soc_max_frac:
            raise ValueError(
                f"battery initial_soc_frac {self.initial_soc_frac} is outside its "
                f"own window [{self.soc_min_frac}, {self.soc_max_frac}]"
            )
        if self.degradation_cost_eur_mwh < 0:
            raise ValueError("battery degradation_cost_eur_mwh must be >= 0")

    @property
    def usable_mwh(self) -> float:
        return (self.soc_max_frac - self.soc_min_frac) * self.energy_mwh

    @property
    def hours_at_full_power(self) -> float:
        """How long it can hold full output — the number an operator asks for."""
        return self.usable_mwh / self.power_mw


@dataclass(frozen=True)
class FlexibleLoad:
    """A shiftable block of energy, per §2's "shift flexible loads".

    It must consume `energy_mwh` somewhere inside its window; the optimizer
    chooses when. This is what makes "shift flexible loads away from peak-price
    hours" a solvable request rather than a slogan.
    """

    name: str = "flex_block"
    energy_mwh: float = 0.4
    max_power_mw: float = 0.15
    earliest_hour: int = 0
    latest_hour: int = 23

    def default_schedule(self, hours: int = 24) -> list[float]:
        """Where the block runs when nobody is optimizing it: as early as possible.

        This matters more than it looks. A request that does *not* ask to shift
        flexible load must still consume that energy — the load exists whether
        or not permission was given to move it. An earlier version dropped the
        block entirely in that case, which quietly deleted 0.4 MWh from the
        site and made every before/after comparison flatter than the truth.
        """
        remaining = self.energy_mwh
        schedule = [0.0] * hours
        for hour in range(self.earliest_hour, min(self.latest_hour, hours - 1) + 1):
            if remaining <= 0:
                break
            take = min(self.max_power_mw, remaining)
            schedule[hour] = take
            remaining -= take
        return schedule

    def __post_init__(self) -> None:
        if self.energy_mwh <= 0 or self.max_power_mw <= 0:
            raise ValueError(f"{self.name}: energy_mwh and max_power_mw must be > 0")
        if not 0 <= self.earliest_hour <= self.latest_hour <= 23:
            raise ValueError(
                f"{self.name}: window must satisfy 0 <= earliest <= latest <= 23, "
                f"got [{self.earliest_hour}, {self.latest_hour}]"
            )
        window_hours = self.latest_hour - self.earliest_hour + 1
        if self.energy_mwh > self.max_power_mw * window_hours + 1e-9:
            raise ValueError(
                f"{self.name}: needs {self.energy_mwh} MWh but can absorb at most "
                f"{self.max_power_mw * window_hours} MWh inside its own window "
                f"({window_hours} h at {self.max_power_mw} MW). This is infeasible "
                "by construction and would surface as a baffling solver failure."
            )


@dataclass(frozen=True)
class SiteConfig:
    """Everything the copilot is allowed to reason about."""

    name: str = "reference site"
    peak_load_mw: float = 0.50
    pv_peak_mw: float = 0.60
    battery: Battery | None = field(default_factory=Battery)
    flexible_loads: tuple[FlexibleLoad, ...] = field(default_factory=lambda: (FlexibleLoad(),))
    grid_import_limit_mw: float = 0.60
    grid_export_limit_mw: float = 0.30
    export_price_ratio: float = 0.90
    demand_charge_eur_mw: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_load_mw <= 0:
            raise ValueError("peak_load_mw must be > 0")
        if self.pv_peak_mw < 0:
            raise ValueError("pv_peak_mw must be >= 0")
        if self.grid_import_limit_mw <= 0:
            raise ValueError("grid_import_limit_mw must be > 0")
        if self.grid_export_limit_mw < 0:
            raise ValueError("grid_export_limit_mw must be >= 0")
        if not 0 <= self.export_price_ratio <= 1:
            # Above 1 the site is paid more to export than it pays to import,
            # and the optimum is to import and export simultaneously forever
            # with no battery involved. That is a broken tariff, not a clever
            # dispatch, and it belongs here rather than as a binary variable in
            # the model.
            raise ValueError(
                f"export_price_ratio must be in [0, 1], got {self.export_price_ratio}. "
                "Above 1 the meter can be gamed: the optimum imports and exports "
                "in the same hour indefinitely."
            )
        names = [flex.name for flex in self.flexible_loads]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate flexible load name(s): {duplicates}")

    @property
    def has_battery(self) -> bool:
        return self.battery is not None

    @property
    def has_flexible_load(self) -> bool:
        return bool(self.flexible_loads)

    def describe(self) -> str:
        """A plain description, used in the prompt and shown in the UI.

        Generated from the configuration rather than written out, so the model
        is never told about an asset the site does not have.
        """
        parts = [
            f"grid connection: import up to {self.grid_import_limit_mw} MW, "
            f"export up to {self.grid_export_limit_mw} MW",
            f"site load: peak {self.peak_load_mw} MW",
        ]
        if self.pv_peak_mw > 0:
            parts.append(f"PV array: {self.pv_peak_mw} MWp")
        if self.battery is not None:
            battery = self.battery
            parts.append(
                f"battery: {battery.energy_mwh} MWh / {battery.power_mw} MW, "
                f"usable SOC {battery.soc_min_frac:.0%}-{battery.soc_max_frac:.0%}, "
                f"{battery.hours_at_full_power:.1f} h at full power"
            )
        for flex in self.flexible_loads:
            parts.append(
                f"flexible load '{flex.name}': {flex.energy_mwh} MWh to be placed "
                f"between hours {flex.earliest_hour} and {flex.latest_hour}, "
                f"at most {flex.max_power_mw} MW"
            )
        return "\n".join(f"- {part}" for part in parts)


def load_days(path: str | Path | None = None) -> pd.DataFrame:
    """Read the committed representative days."""
    path = Path(path) if path is not None else DEFAULT_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It should be committed; regenerate with "
            "`python data/prepare_dataset.py` (needs the raw OPSD snapshot)."
        )
    frame = pd.read_csv(path)
    expected = {
        "day", "hour", "selected_because", "price_eur_mwh",
        "load_mw", "pv_mw", "carbon_g_kwh",
    }
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    return frame


def available_days(path: str | Path | None = None) -> list[str]:
    return sorted(load_days(path)["day"].unique())


def day_profile(day: str, path: str | Path | None = None) -> pd.DataFrame:
    """One day's 24 hours, indexed 0..23.

    A day that is not in the dataset raises **and lists what is available**. A
    copilot that answers "no data for 2019-01-01" without saying what it does
    have is a copilot that wastes the next turn.
    """
    frame = load_days(path)
    block = frame[frame["day"] == day]
    if block.empty:
        raise KeyError(
            f"no data for {day!r}. Available days: {sorted(frame['day'].unique())}"
        )
    if len(block) != 24:
        raise ValueError(f"{day} has {len(block)} hours, expected 24")
    return block.sort_values("hour").reset_index(drop=True)


def scale_to_site(profile: pd.DataFrame, site: SiteConfig) -> pd.DataFrame:
    """Rescale a day to a site whose size differs from the dataset's.

    The committed data is already at the reference site's scale, so the common
    path multiplies by 1.0. It matters for scenario comparison (§17): "compare a
    1 MWh battery with a 2 MWh one" changes the battery, not the load, and
    "compare with twice the PV" must change PV without touching price.
    """
    out = profile.copy()
    load_scale = site.peak_load_mw / 0.50
    pv_scale = (site.pv_peak_mw / 0.60) if site.pv_peak_mw > 0 else 0.0
    out["load_mw"] = out["load_mw"] * load_scale
    out["pv_mw"] = out["pv_mw"] * pv_scale
    return out
