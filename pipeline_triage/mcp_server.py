"""Read-only MCP server exposing the synthetic pipeline environment.

Tools (all scoped to the failed run under investigation):
- get_run_logs(run_id, tail_lines)
- get_recent_changes(run_id, since_hours)
- get_table_schema(run_id, table)
- get_run_history(run_id, last_n)

Every synthetic incident is an independent snapshot of the world (its own
schemas, change history and run history), so each tool answers from the
snapshot of the run being triaged, and time windows are anchored at that run's
start. Querying by pipeline or table across incidents would mix unrelated
worlds and hand the agent another incident's evidence.

All tools are strictly read-only. The agent triages; a human approves.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as _MCPToolError
from pydantic import Field

from .env import Environment
from .models import FailedRun


class ToolError(_MCPToolError):
    """Deliberate, user-facing tool error.

    Subclasses the MCP SDK's ToolError so the server returns the message to the
    model as an `is_error` result. Any other exception type is treated by the SDK
    as a crash: it is logged with a traceback and the model only sees a generic
    "Error executing tool" with the actual reason stripped.
    """


MAX_TAIL_LINES = 2000
MAX_SINCE_HOURS = 24 * 90
MAX_HISTORY = 100


def _require_run(env: Environment, run_id: str) -> FailedRun:
    run = env.get_run(run_id)
    if run is None:
        raise ToolError(f"Run '{run_id}' not found.")
    return run


def _require_range(name: str, value: int, lo: int, hi: int) -> None:
    if not lo <= value <= hi:
        raise ToolError(f"{name} must be between {lo} and {hi}, got {value}.")


def get_run_logs(env: Environment, run_id: str, tail_lines: int = 200) -> str:
    """Logs for a run: succeeded upstream tasks first, then the last `tail_lines` of the failing task."""
    _require_range("tail_lines", tail_lines, 1, MAX_TAIL_LINES)
    run = _require_run(env, run_id)
    parts: list[str] = []

    for task, lines in run.logs.items():
        if task == run.failed_task:
            continue
        parts.append(f"===== task {task} (succeeded) =====")
        parts.extend(lines)

    parts.append(f"===== FAILING TASK: {run.failed_task} =====")
    parts.extend(run.logs.get(run.failed_task, [])[-tail_lines:])
    return "\n".join(parts)


def get_recent_changes(env: Environment, run_id: str, since_hours: int = 48) -> str:
    """Code/config changes to the run's pipeline in the `since_hours` before the run started, newest first."""
    _require_range("since_hours", since_hours, 1, MAX_SINCE_HOURS)
    run = _require_run(env, run_id)
    window_start = run.started_at - timedelta(hours=since_hours)

    changes = [c for c in run.changes if window_start <= c.committed_at <= run.started_at]
    changes.sort(key=lambda c: c.committed_at, reverse=True)
    # an object, not a bare list: an empty window reads as an explicit, quotable "no changes"
    return json.dumps(
        {
            "pipeline": run.pipeline,
            "window": f"{since_hours}h before run start {run.started_at.isoformat()}",
            "changes": [
                {
                    "kind": c.kind,
                    "description": c.description,
                    "committed_at": c.committed_at.isoformat(),
                    "hours_before_run": round((run.started_at - c.committed_at).total_seconds() / 3600, 1),
                    "author": c.author,
                    "commit_sha": c.commit_sha[:12],
                }
                for c in changes
            ],
        },
        indent=2,
    )


def get_table_schema(env: Environment, run_id: str, table: str) -> str:
    """Schema of `table` as it was when the run started."""
    run = _require_run(env, run_id)
    schema = run.schemas.get(table)
    if schema is None:
        known = sorted(run.schemas) or ["(none)"]
        raise ToolError(f"Table '{table}' not found. Known tables: {', '.join(known)}.")
    return json.dumps(
        {
            "table": schema.table,
            "version": schema.version,
            "changed_at": schema.changed_at.isoformat() if schema.changed_at else None,
            "change_description": schema.change_description,
            "columns": [{"name": c.name, "dtype": c.dtype} for c in schema.columns],
        },
        indent=2,
    )


def get_run_history(env: Environment, run_id: str, last_n: int = 10) -> str:
    """The pipeline's most recent runs up to and including this one, oldest first."""
    _require_range("last_n", last_n, 1, MAX_HISTORY)
    run = _require_run(env, run_id)
    return json.dumps(
        [
            {
                "run_id": h.run_id,
                "pipeline": h.pipeline,
                "started_at": h.started_at.isoformat(),
                "duration_s": round(h.duration_s),
                "status": h.status,
                "failed_task": h.failed_task,
            }
            for h in run.run_history[-last_n:]
        ],
        indent=2,
    )


RunId = Annotated[str, Field(description="The failed run under investigation, e.g. run_00042.")]


def create_server(env: Environment) -> MCPServer:
    """Build an MCPServer bound to the given environment (all tools read-only).

    The tool names, descriptions and parameter schemas declared here are the
    single source of truth: the agent builds its function-calling specs from
    this server's `list_tools` response.
    """
    from mcp.types import ToolAnnotations

    server = MCPServer(
        name="pipeline-triage",
        instructions=(
            "Read-only tools over a synthetic data-pipeline environment "
            "(Airflow-style runs, table schemas, code/config change history). "
            "Every tool is scoped to one failed run (run_id) and answers as of that run's start. "
            "Use them to triage the root cause of the failure."
        ),
    )
    ro = ToolAnnotations(readOnlyHint=True)  # every tool here is read-only

    @server.tool(
        name="get_run_logs",
        description=(
            "Get the logs of a failed run: succeeded upstream tasks first, then the failing task's section "
            "(marked 'FAILING TASK'), limited to its last tail_lines lines."
        ),
        annotations=ro,
    )
    def _get_run_logs(
        run_id: RunId,
        tail_lines: Annotated[
            int, Field(ge=1, le=MAX_TAIL_LINES, description="Lines to keep from the end of the failing task's log.")
        ] = 200,
    ) -> str:
        return get_run_logs(env, run_id, tail_lines)

    @server.tool(
        name="get_recent_changes",
        description=(
            "Get code/config changes to this run's pipeline committed in the since_hours before the run "
            "started, newest first. A meaningful change shortly before the failure is a strong bad_deploy signal; "
            "docs/chore/style/test commits usually are not."
        ),
        annotations=ro,
    )
    def _get_recent_changes(
        run_id: RunId,
        since_hours: Annotated[
            int, Field(ge=1, le=MAX_SINCE_HOURS, description="Look-back window in hours before the run start.")
        ] = 48,
    ) -> str:
        return get_recent_changes(env, run_id, since_hours)

    @server.tool(
        name="get_table_schema",
        description=(
            "Get a table's schema as of this run's start (version, changed_at, change description, columns). "
            "A version bump with a change description shortly before the run indicates upstream schema drift."
        ),
        annotations=ro,
    )
    def _get_table_schema(
        run_id: RunId,
        table: Annotated[str, Field(description="Fully qualified table name, e.g. raw.orders.")],
    ) -> str:
        return get_table_schema(env, run_id, table)

    @server.tool(
        name="get_run_history",
        description=(
            "Get this pipeline's most recent runs up to and including the failed one (status, duration). "
            "Repeated earlier failures point to a persistent cause; a long healthy streak broken right after "
            "a change points to bad_deploy."
        ),
        annotations=ro,
    )
    def _get_run_history(
        run_id: RunId,
        last_n: Annotated[int, Field(ge=1, le=MAX_HISTORY, description="How many recent runs to return.")] = 10,
    ) -> str:
        return get_run_history(env, run_id, last_n)

    return server
