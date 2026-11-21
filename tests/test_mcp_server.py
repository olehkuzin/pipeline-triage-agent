"""Tests for the read-only MCP server, via both direct calls and the MCP protocol (in-memory transport)."""

import asyncio
import json

import pytest
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession  # noqa: F401  (re-export clarity)

from pipeline_triage.env import Environment
from pipeline_triage.mcp_server import ToolError, create_server, get_run_logs, get_recent_changes, get_run_history, get_table_schema


@pytest.fixture(scope="module")
def env():
    return Environment.generate(n=80, seed=99)


@pytest.fixture(scope="module")
def drift_run(env):
    from pipeline_triage.models import RootCause

    runs = [r for r in env.all_runs() if r.root_cause == RootCause.SCHEMA_DRIFT]
    assert runs
    return runs[0]


# ---------- direct (unit) tests ----------


def test_get_run_logs_includes_failing_task(env, drift_run):
    logs = get_run_logs(env, drift_run.run_id)
    assert f"FAILING TASK: {drift_run.failed_task}" in logs
    for task in drift_run.logs:
        assert task in logs


def test_get_run_logs_tail(env, drift_run):
    """tail_lines only limits the failing task's log, keeping the header lines."""
    logs = get_run_logs(env, drift_run.run_id, tail_lines=3)
    failing_section = logs.split("FAILING TASK:", 1)[1]
    body = [l for l in failing_section.splitlines() if l][1:]
    assert len(body) == 3


def test_run_not_found_raises(env):
    with pytest.raises(ToolError):
        get_run_logs(env, "run_nonexistent", 200)


def test_get_recent_changes_respects_window(env):
    runs = [r for r in env.all_runs() if r.changes]
    run = runs[0]
    out = json.loads(get_recent_changes(env, run.pipeline, since_hours=24))
    latest_start = max(r.started_at for r in env.all_runs() if r.pipeline == run.pipeline)
    for ch in out:
        assert __import__("datetime").datetime.fromisoformat(ch["committed_at"]) >= latest_start - __import__("datetime").timedelta(hours=24)


def test_get_recent_changes_pipeline_not_found(env):
    with pytest.raises(ToolError):
        get_recent_changes(env, "no_such_pipeline", 48)


def test_get_table_schema(env, drift_run):
    table = next(iter(drift_run.schemas))
    out = json.loads(get_table_schema(env, table))
    assert out["table"] == table
    assert out["columns"]


def test_get_table_schema_not_found(env):
    with pytest.raises(ToolError):
        get_table_schema(env, "raw.nope")


def test_get_run_history(env, drift_run):
    out = json.loads(get_run_history(env, drift_run.pipeline, last_n=5))
    assert 1 <= len(out) <= 5
    assert out[-1]["status"] == "failed"


# ---------- protocol-level tests via in-memory transport ----------


async def _list_tools_and_call(env):
    server = create_server(env)
    async with InMemoryTransport(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert names == {"get_run_logs", "get_recent_changes", "get_table_schema", "get_run_history"}

            run = env.all_runs()[0]
            result = await session.call_tool("get_run_logs", {"run_id": run.run_id, "tail_lines": 100})
            assert not result.is_error
            text = result.content[0].text
            assert run.failed_task in text

            # read-only annotation present
            lg = next(t for t in tools.tools if t.name == "get_run_logs")
            assert lg.annotations is not None and lg.annotations.read_only_hint

            # errors surface as tool errors, not crashes
            err = await session.call_tool("get_run_logs", {"run_id": "missing"})
            assert err.is_error


def test_mcp_protocol(env):
    asyncio.run(_list_tools_and_call(env))


# ---------- end-to-end test: spawn the real server subprocess over stdio ----------


def test_stdio_end_to_end(env):
    """Spawn `python -m pipeline_triage.mcp_server_main` as a subprocess and speak MCP over pipes."""
    import asyncio
    import sys
    from pathlib import Path

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        repo_root = str(Path(__file__).resolve().parents[1])
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pipeline_triage.mcp_server_main", "--n", "30", "--seed", "5"],
            env={"PYTHONPATH": repo_root},
            cwd=repo_root,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {t.name for t in tools.tools} == {
                    "get_run_logs",
                    "get_recent_changes",
                    "get_table_schema",
                    "get_run_history",
                }
                import json as _json

                r = await session.call_tool("get_run_history", {"pipeline": "daily_sales_etl", "last_n": 5})
                hist = _json.loads(r.content[0].text)
                run_id = hist[-1]["run_id"]
                assert hist[-1]["status"] == "failed"

                logs = await session.call_tool("get_run_logs", {"run_id": run_id, "tail_lines": 50})
                assert "FAILING TASK" in logs.content[0].text

                err = await session.call_tool("get_table_schema", {"table": "raw.no_such_table"})
                assert err.is_error

    asyncio.run(scenario())
