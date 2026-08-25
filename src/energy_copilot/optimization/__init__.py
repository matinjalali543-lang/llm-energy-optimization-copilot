"""The deterministic core. Accepts validated structures; never text."""

from energy_copilot.optimization.builder import build_model
from energy_copilot.optimization.solve import (
    DispatchResult,
    InfeasibleRequest,
    check_no_simultaneous_flows,
    check_power_balance,
    solve_intent,
    solve_model,
)

__all__ = [
    "DispatchResult", "InfeasibleRequest", "build_model",
    "check_no_simultaneous_flows", "check_power_balance", "solve_intent", "solve_model",
]
