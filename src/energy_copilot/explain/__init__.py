"""Explanations, and a check that every number in one is real."""

from energy_copilot.explain.factuality import (
    FactualityReport,
    check_factuality,
    extract_numbers,
    fact_sheet,
)
from energy_copilot.explain.render import describe, explain, explain_with_model

__all__ = [
    "FactualityReport", "check_factuality", "describe", "explain",
    "explain_with_model", "extract_numbers", "fact_sheet",
]
