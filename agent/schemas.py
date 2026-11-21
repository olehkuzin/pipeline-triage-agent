"""Structured input/output schemas for the triage agent."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from pipeline_triage.models import FailedRun, RootCause

EvidenceSource = Literal["run_logs", "table_schema", "recent_changes", "run_history"]

# evidence source -> the MCP tool whose output must contain the quote
SOURCE_TOOL: dict[str, str] = {
    "run_logs": "get_run_logs",
    "table_schema": "get_table_schema",
    "recent_changes": "get_recent_changes",
    "run_history": "get_run_history",
}


class Incident(BaseModel):
    """What the alert hands the agent: which run failed, where, and when."""

    run_id: str
    pipeline: str
    failed_task: str
    started_at: datetime

    @classmethod
    def from_run(cls, run: FailedRun) -> Incident:
        return cls(run_id=run.run_id, pipeline=run.pipeline, failed_task=run.failed_task, started_at=run.started_at)


class EvidenceItem(BaseModel):
    """A single piece of evidence the agent cites for its diagnosis.

    `quote` must be a verbatim snippet of the output of the tool named by
    `source`; the agent loop rejects diagnoses whose quotes cannot be found.
    """

    source: EvidenceSource = Field(
        description="Which tool the evidence came from: run_logs, table_schema, recent_changes, or run_history."
    )
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
    evidence: list[EvidenceItem] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def _evidence_required(self) -> Diagnosis:
        # 'unknown' means "not enough evidence", so it may cite none; every real cause must cite some.
        if self.root_cause != RootCause.UNKNOWN and not self.evidence:
            raise ValueError("evidence must contain at least one item unless root_cause is 'unknown'")
        return self
