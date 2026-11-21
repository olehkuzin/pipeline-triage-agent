"""Agent package: LLM tool-calling loop over the MCP server."""

from .loop import TriageAgent, ToolCallBudgetExceeded
from .schemas import Diagnosis, EvidenceItem

__all__ = ["TriageAgent", "ToolCallBudgetExceeded", "Diagnosis", "EvidenceItem"]
