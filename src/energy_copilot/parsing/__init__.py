"""Two parsers, one interface, so the comparison is a comparison."""

from energy_copilot.parsing.llm_parser import LLMParser, interpret_response
from energy_copilot.parsing.rule_based import parse as rule_based_parse

__all__ = ["LLMParser", "interpret_response", "rule_based_parse"]
