"""Agent package: LLM tool-calling loop over the MCP server."""

from .loop import TriageAgent, Trace
from .schemas import Diagnosis, EvidenceItem, Incident
from .tools import MCPToolClient

__all__ = ["Diagnosis", "EvidenceItem", "Incident", "MCPToolClient", "Trace", "TriageAgent"]
