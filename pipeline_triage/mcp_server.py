"""Read-only MCP server exposing the synthetic pipeline environment.

Tools:
- get_run_logs(run_id, tail_lines)
- get_recent_changes(pipeline, since_hours)
- get_table_schema(table)
- get_run_history(pipeline, last_n)

All tools are strictly read-only. The agent triages; a human approves.
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from .env import Environment
from .models import FailedRun


class ToolError(Exception):
    """User-facing tool error (returned as error by the server)."""


def _require_run(env: Environment, run_id: str) -> FailedRun:
    run = env.get_run(run_id)
    if run is None:
        raise ToolError(f"Run '{run_id}' not found.")
    return run


def get_run_logs(env: Environment, run_id: str, tail_lines: int = 200) -> str:
    """Full logs for a run: succeeded upstream tasks first, then the failing task."""
    run = _require_run(env, run_id)
    parts: list[str] = []

    for task, lines in run.logs.items():
        if task == run.failed_task:
            continue
        parts.append(f"===== task {task} (succeeded) =====")
        parts.extend(lines)

    parts.append(f"===== FAILING TASK: {run.failed_task} =====")
    failing = run.logs[run.failed_task][-tail_lines:]
    parts.extend(failing)

    return "\n".join(parts)


def get_recent_changes(env: Environment, pipeline: str, since_hours: int = 48) -> str:
    """Code/config changes for a pipeline, most recent first, within a window relative to each run's start.

    Since our environment is a snapshot, `since_hours` is applied relative to
    the most recent run's start time of that pipeline.
    """
    runs = [r for r in env.all_runs() if r.pipeline == pipeline]
    if not runs:
        raise ToolError(f"Pipeline '{pipeline}' not found.")
    latest_start = max(r.started_at for r in runs)

    cutoff = latest_start.timestamp() - since_hours * 3600
    changes = [
        {
            "kind": c.kind,
            "description": c.description,
            "committed_at": c.committed_at.isoformat(),
            "author": c.author,
            "commit_sha": c.commit_sha[:12],
        }
        for r in runs
        for c in r.changes
        if c.pipeline == pipeline and c.committed_at.timestamp() >= cutoff
    ]
    changes.sort(key=lambda c: c["committed_at"], reverse=True)
    return json.dumps(changes, indent=2)


def get_table_schema(env: Environment, table: str) -> str:
    """Latest known schema for a table."""
    for run in env.all_runs():
        if table in run.schemas:
            s = run.schemas[table]
            return json.dumps(
                {
                    "table": s.table,
                    "version": s.version,
                    "changed_at": s.changed_at.isoformat() if s.changed_at else None,
                    "change_description": s.change_description,
                    "columns": [{"name": c.name, "dtype": c.dtype} for c in s.columns],
                },
                indent=2,
            )
    raise ToolError(f"Table '{table}' not found.")


def get_run_history(env: Environment, pipeline: str, last_n: int = 10) -> str:
    """Recent run history for a pipeline, oldest first."""
    runs = sorted(
        (r for r in env.all_runs() if r.pipeline == pipeline),
        key=lambda r: r.started_at,
    )
    if not runs:
        raise ToolError(f"Pipeline '{pipeline}' not found.")
    return json.dumps(
        [
            {
                "run_id": r.run_id,
                "pipeline": r.pipeline,
                "started_at": r.started_at.isoformat(),
                "status": "failed",
                "failed_task": r.failed_task,
            }
            for r in runs[-last_n:]
        ],
        indent=2,
    )


def create_server(env: Environment) -> MCPServer:
    """Build an MCPServer bound to the given environment (all tools read-only)."""
    from mcp.types import ToolAnnotations

    server = MCPServer(
        name="pipeline-triage",
        instructions=(
            "Read-only tools over a synthetic data-pipeline environment "
            "(Airflow-style runs, table schemas, code/config change history). "
            "Use them to triage the root cause of failed pipeline runs."
        ),
    )
    ro = ToolAnnotations(readOnlyHint=True)  # every tool here is read-only

    @server.tool(name="get_run_logs", description="Get the logs of a pipeline run, including the failing task.", annotations=ro)
    def _get_run_logs(run_id: str, tail_lines: int = 200) -> str:
        return get_run_logs(env, run_id, tail_lines)

    @server.tool(
        name="get_recent_changes",
        description="Get code/config changes for a pipeline within the last N hours (relative to its latest run).",
        annotations=ro,
    )
    def _get_recent_changes(pipeline: str, since_hours: int = 48) -> str:
        return get_recent_changes(env, pipeline, since_hours)

    @server.tool(name="get_table_schema", description="Get the current schema of a table.", annotations=ro)
    def _get_table_schema(table: str) -> str:
        return get_table_schema(env, table)

    @server.tool(
        name="get_run_history",
        description="Get the recent run history (success/failure) of a pipeline.",
        annotations=ro,
    )
    def _get_run_history(pipeline: str, last_n: int = 10) -> str:
        return get_run_history(env, pipeline, last_n)

    return server
