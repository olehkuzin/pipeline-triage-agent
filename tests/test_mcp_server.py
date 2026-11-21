"""Tests for the read-only MCP server, via both direct calls and the MCP protocol (in-memory transport)."""

import asyncio
import json
from datetime import datetime, timedelta

import pytest
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport

from pipeline_triage.env import Environment
from pipeline_triage.mcp_server import (
    ToolError,
    create_server,
    get_recent_changes,
    get_run_history,
    get_run_logs,
    get_table_schema,
)
from pipeline_triage.models import RootCause

MEANINGFUL = lambda c: not c["description"].startswith(("Docs:", "Chore:", "Style:", "Test:"))  # noqa: E731
TOOL_NAMES = {"get_run_logs", "get_recent_changes", "get_table_schema", "get_run_history"}


@pytest.fixture(scope="module")
def env():
    return Environment.generate(n=120, seed=99)


@pytest.fixture(scope="module")
def drift_run(env):
    return next(r for r in env.all_runs() if r.root_cause == RootCause.SCHEMA_DRIFT)


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
    body = [line for line in failing_section.splitlines() if line][1:]
    assert len(body) == 3


@pytest.mark.parametrize(
    "call",
    [
        lambda env, rid: get_run_logs(env, rid, tail_lines=0),
        lambda env, rid: get_recent_changes(env, rid, since_hours=0),
        lambda env, rid: get_run_history(env, rid, last_n=0),
    ],
)
def test_out_of_range_arguments_raise(env, drift_run, call):
    with pytest.raises(ToolError, match="must be between"):
        call(env, drift_run.run_id)


def test_run_not_found_raises(env):
    for call in (
        lambda: get_run_logs(env, "run_nonexistent"),
        lambda: get_recent_changes(env, "run_nonexistent"),
        lambda: get_table_schema(env, "run_nonexistent", "raw.orders"),
        lambda: get_run_history(env, "run_nonexistent"),
    ):
        with pytest.raises(ToolError, match="not found"):
            call()


def test_schema_tool_returns_the_runs_own_snapshot(env):
    """Every drift run sees its own drifted schema; the lookup never leaks another incident's table."""
    for run in env.all_runs():
        for table, schema in run.schemas.items():
            out = json.loads(get_table_schema(env, run.run_id, table))
            assert out["version"] == schema.version
            assert out["change_description"] == schema.change_description
        if run.root_cause == RootCause.SCHEMA_DRIFT:
            assert any(s.version >= 2 for s in run.schemas.values())


def test_schema_not_found_lists_known_tables(env, drift_run):
    with pytest.raises(ToolError, match="Known tables: raw\\."):
        get_table_schema(env, drift_run.run_id, "raw.nope")


def test_recent_changes_are_the_runs_own_and_windowed(env):
    for run in env.all_runs():
        out = json.loads(get_recent_changes(env, run.run_id, since_hours=24))
        assert out["pipeline"] == run.pipeline
        for ch in out["changes"]:
            committed = datetime.fromisoformat(ch["committed_at"])
            assert run.started_at - timedelta(hours=24) <= committed <= run.started_at
            assert ch["commit_sha"] in {c.commit_sha[:12] for c in run.changes}

        recent = json.loads(get_recent_changes(env, run.run_id, since_hours=48))["changes"]
        meaningful = [c for c in recent if MEANINGFUL(c)]
        deploy_expected = run.root_cause == RootCause.BAD_DEPLOY or (
            run.ambiguous and RootCause.BAD_DEPLOY in run.accepted_causes
        )
        # the culprit commit is visible to its own incident, and no other incident is shown one
        assert bool(meaningful) == deploy_expected, run.run_id


def test_run_history_is_the_pipelines_own_and_ends_with_this_run(env, drift_run):
    out = json.loads(get_run_history(env, drift_run.run_id, last_n=5))
    assert len(out) == 5
    assert out[-1]["run_id"] == drift_run.run_id
    assert out[-1]["status"] == "failed"
    assert {h["status"] for h in out[:-1]} == {"success"}
    assert {h["pipeline"] for h in out} == {drift_run.pipeline}


# ---------- protocol-level tests via in-memory transport ----------


async def _list_tools_and_call(env):
    server = create_server(env)
    async with InMemoryTransport(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == TOOL_NAMES

            run = env.all_runs()[0]
            result = await session.call_tool("get_run_logs", {"run_id": run.run_id, "tail_lines": 100})
            assert not result.is_error
            assert run.failed_task in result.content[0].text

            lg = next(t for t in tools.tools if t.name == "get_run_logs")
            # read-only annotation present, bounds and descriptions published in the schema
            assert lg.annotations is not None and lg.annotations.read_only_hint
            props = lg.input_schema["properties"]
            assert props["tail_lines"]["minimum"] == 1 and props["tail_lines"]["maximum"] == 2000
            assert props["run_id"]["description"]
            for tool in tools.tools:
                assert "run_id" in tool.input_schema["required"]

            # deliberate tool errors reach the client with their message, not a generic crash
            err = await session.call_tool("get_run_logs", {"run_id": "missing"})
            assert err.is_error
            assert "Run 'missing' not found" in err.content[0].text

            # out-of-range arguments are rejected by the published schema
            bad = await session.call_tool("get_run_logs", {"run_id": run.run_id, "tail_lines": 0})
            assert bad.is_error


def test_mcp_protocol(env):
    asyncio.run(_list_tools_and_call(env))


# ---------- end-to-end test: spawn the real server subprocess over stdio ----------


def test_stdio_end_to_end():
    """Spawn `python -m pipeline_triage.mcp_server_main` as a subprocess and speak MCP over pipes."""
    import sys
    from pathlib import Path

    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    local = Environment.generate(n=30, seed=5)  # the server builds the identical seeded dataset
    run = local.all_runs()[0]

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
                assert {t.name for t in tools.tools} == TOOL_NAMES

                r = await session.call_tool("get_run_history", {"run_id": run.run_id, "last_n": 5})
                hist = json.loads(r.content[0].text)
                assert hist[-1]["run_id"] == run.run_id
                assert hist[-1]["status"] == "failed"

                logs = await session.call_tool("get_run_logs", {"run_id": run.run_id, "tail_lines": 50})
                assert "FAILING TASK" in logs.content[0].text

                err = await session.call_tool("get_table_schema", {"run_id": run.run_id, "table": "raw.no_such_table"})
                assert err.is_error

    asyncio.run(scenario())
