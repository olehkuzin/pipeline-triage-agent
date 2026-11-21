"""Structured output schema for the triage agent."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from pipeline_triage.models import RootCause


class EvidenceItem(BaseModel):
    """A single piece of evidence the agent cites for its diagnosis.

    source is one of: 'run_logs' | 'table_schema' | 'recent_changes' | 'run_history'
    quote must be a verbatim snippet (substring) that actually appears in the
    corresponding tool output — this is checked by the eval harness.
    """

    source: str = Field(description="Which tool the evidence came from: run_logs, table_schema, recent_changes, or run_history.")
    quote: str = Field(description="Verbatim quote from that tool's output.", min_length=4)
    note: str = Field(default="", description="Optional one-line interpretation of this evidence.")

    @field_validator("quote")
    @classmethod
    def _quote_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 4:
            raise ValueError("evidence quote must be a non-trivial verbatim quote")
        return v


class Diagnosis(BaseModel):
    """The agent's final structured diagnosis of a failed run."""

    run_id: str
    root_cause: RootCause
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(min_length=1)
    proposed_next_step: str = Field(min_length=8, description="Concrete next action for the on-call engineer.")
    requires_human_approval: bool = True

    @field_validator("requires_human_approval")
    @classmethod
    def _always_approval(cls, v: bool) -> bool:
        # Guardrail: the agent is advisory-only; a human always approves.
        return True

    @field_validator("confidence")
    @classmethod
    def _calibrate(cls, v: float) -> float:
        # Round to 2 decimals to discourage false precision.
        return round(v, 2)
