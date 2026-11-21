"""Failure taxonomy and shared Pydantic models for the synthetic environment."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class RootCause(StrEnum):
    """Ground-truth taxonomy of pipeline failure causes."""

    SCHEMA_DRIFT = "schema_drift"
    LATE_OR_MISSING_DATA = "late_or_missing_data"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PERMISSION_OR_AUTH = "permission_or_auth"
    BAD_DEPLOY = "bad_deploy"
    DATA_QUALITY_VIOLATION = "data_quality_violation"
    UPSTREAM_DEPENDENCY_FAILURE = "upstream_dependency_failure"
    TRANSIENT_INFRA = "transient_infra"
    UNKNOWN = "unknown"


ALL_CAUSES: list[RootCause] = [
    RootCause.SCHEMA_DRIFT,
    RootCause.LATE_OR_MISSING_DATA,
    RootCause.RESOURCE_EXHAUSTION,
    RootCause.PERMISSION_OR_AUTH,
    RootCause.BAD_DEPLOY,
    RootCause.DATA_QUALITY_VIOLATION,
    RootCause.UPSTREAM_DEPENDENCY_FAILURE,
    RootCause.TRANSIENT_INFRA,
    RootCause.UNKNOWN,
]


class ColumnSpec(BaseModel):
    """A single column in a synthetic table schema."""

    name: str
    dtype: str  # e.g. "string", "int64", "timestamp", "float64"


class TableSchema(BaseModel):
    """Versioned schema for a synthetic table."""

    table: str
    columns: list[ColumnSpec]
    version: int = 1
    changed_at: datetime | None = None
    change_description: str | None = None

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class Change(BaseModel):
    """A code or config change to a pipeline, used for bad_deploy detection."""

    pipeline: str = ""
    kind: str  # "code_change" or "config_change"
    description: str = ""
    committed_at: datetime
    author: str
    commit_sha: str

    def summarize(self) -> str:
        return (
            f"[{self.committed_at.isoformat()}] {self.kind} by {self.author} "
            f"({self.commit_sha[:8]}): {self.description}"
        )


class RunSummary(BaseModel):
    """Compact record of one DAG run."""

    run_id: str
    pipeline: str
    started_at: datetime
    duration_s: float
    status: str  # "success" | "failed"
    failed_task: str | None = None


class FailedRun(BaseModel):
    """A complete synthetic failed-run incident with ground truth."""

    run_id: str
    pipeline: str
    started_at: datetime
    failed_task: str
    root_cause: RootCause
    ambiguous: bool = Field(
        default=False,
        description="True when multiple causes are plausible; grading accepts either.",
    )
    accepted_causes: list[RootCause] = Field(default_factory=list)
    logs: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Task name -> log lines (ordered). The failing task logs contain the signal.",
    )
    schemas: dict[str, TableSchema] = Field(
        default_factory=dict, description="table -> schema (latest snapshot per table)."
    )
    changes: list[Change] = Field(default_factory=list)
    run_history: list[RunSummary] = Field(default_factory=list)
