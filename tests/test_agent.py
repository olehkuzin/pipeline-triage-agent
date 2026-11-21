"""Agent loop tests with a scripted fake LLM (no API calls), tools served over MCP (in-memory)."""

import json

import pytest

from agent.llm import LLMResponse, ToolCallRequest
from agent.loop import MAX_TURNS, TriageAgent
from agent.schemas import Diagnosis, Incident
from agent.tools import MCPToolClient
from pipeline_triage.env import Environment
from pipeline_triage.mcp_server import get_run_logs, get_table_schema
from pipeline_triage.models import RootCause


class FakeLLM:
    """Scripted LLM: returns queued responses; records everything it's asked."""

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, json_mode=False, tool_choice="auto"):
        self.calls.append(
            {"messages": [m["role"] for m in messages], "tools": tools, "json_mode": json_mode, "tool_choice": tool_choice}
        )
        return self.script.pop(0)


@pytest.fixture(scope="module")
def env():
    return Environment.generate(n=40, seed=11)


@pytest.fixture(scope="module")
def tools(env):
    with MCPToolClient.in_memory(env) as client:
        yield client


@pytest.fixture(scope="module")
def run(env):
    return next(r for r in env.all_runs() if r.root_cause == RootCause.SCHEMA_DRIFT and not r.ambiguous)


def _diagnosis(run, **overrides):
    payload = {
        "run_id": run.run_id,
        "root_cause": "schema_drift",
        "confidence": 0.85,
        "evidence": [{"source": "run_logs", "quote": "not found in source table", "note": "column error"}],
        "proposed_next_step": "Fix the column mapping and re-run the task.",
    }
    payload.update(overrides)
    return LLMResponse(content=json.dumps(payload))


def _call(name, call_id="c1", **arguments):
    return LLMResponse(tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)])


def test_tool_specs_come_from_the_mcp_server(tools):
    specs = {s["function"]["name"]: s["function"] for s in tools.function_specs()}
    assert set(specs) == {"get_run_logs", "get_recent_changes", "get_table_schema", "get_run_history"}
    for fn in specs.values():
        assert fn["description"]
        assert "run_id" in fn["parameters"]["required"]
        assert "title" not in fn["parameters"]


def test_agent_happy_path_tools_then_diagnosis(env, tools, run):
    table = next(t for t, s in run.schemas.items() if s.version >= 2)
    quote = json.loads(get_table_schema(env, run.run_id, table))["change_description"]
    final = _diagnosis(
        run,
        confidence=0.9,
        evidence=[
            {"source": "run_logs", "quote": "not found in source table", "note": "missing column error"},
            {"source": "table_schema", "quote": quote, "note": "schema changed before run"},
        ],
    )
    fake = FakeLLM([_call("get_run_logs", run_id=run.run_id), _call("get_table_schema", "c2", run_id=run.run_id, table=table), final])
    agent = TriageAgent(fake, tools)
    d = agent.diagnose(Incident.from_run(run))

    assert d.root_cause == "schema_drift"
    assert d.requires_human_approval is True  # guardrail never disabled
    tr = agent.last_trace
    assert tr.tool_call_count == 2 and tr.turns == 3 and tr.fallback_reason is None
    assert [c["is_error"] for c in tr.tool_calls] == [False, False]
    # tools were offered (from MCP) on every turn
    assert all(c["tools"] and c["tool_choice"] == "auto" for c in fake.calls)


def test_invalid_evidence_is_rejected_then_repaired(tools, run):
    fake = FakeLLM(
        [
            _call("get_run_logs", run_id=run.run_id),
            _diagnosis(run, evidence=[{"source": "run_logs", "quote": "this text is nowhere", "note": "invented"}]),
            _diagnosis(run),  # repair: valid
        ]
    )
    agent = TriageAgent(fake, tools)
    d = agent.diagnose(Incident.from_run(run))
    assert d.root_cause == "schema_drift"
    assert agent.last_trace.tool_call_count == 1 and agent.last_trace.repairs == 1
    # repair turn is answer-only: json mode and no new tool calls allowed
    assert fake.calls[-1]["json_mode"] and fake.calls[-1]["tool_choice"] == "none"


def test_evidence_must_come_from_the_named_tool(tools, run):
    """A real log line cited as table_schema evidence is rejected."""
    fake = FakeLLM(
        [
            _call("get_run_logs", run_id=run.run_id),
            _diagnosis(run, evidence=[{"source": "table_schema", "quote": "not found in source table"}]),
            _diagnosis(run),
        ]
    )
    agent = TriageAgent(fake, tools)
    agent.diagnose(Incident.from_run(run))
    assert agent.last_trace.repairs == 1


def test_falls_back_to_unknown_on_garbage(tools, run):
    fake = FakeLLM([LLMResponse(content="I think it broke."), LLMResponse(content="still not json"), LLMResponse(content="nope")])
    agent = TriageAgent(fake, tools)
    d = agent.diagnose(Incident.from_run(run))
    assert d.root_cause == "unknown" and d.confidence == 0.0
    assert d.evidence == []  # no tool output was fetched, so nothing can be quoted
    assert agent.last_trace.fallback_reason == "invalid_output"


def test_fallback_cites_a_verbatim_log_line(env, tools, run):
    fake = FakeLLM([_call("get_run_logs", run_id=run.run_id)] + [LLMResponse(content="nope")] * 3)
    agent = TriageAgent(fake, tools)
    d = agent.diagnose(Incident.from_run(run))
    assert d.root_cause == "unknown"
    assert d.evidence[0].quote in get_run_logs(env, run.run_id)
    assert " ERROR " in d.evidence[0].quote


def test_tool_budget_forces_an_answer_instead_of_crashing(tools, run):
    fake = FakeLLM([_call("get_run_logs", f"c{i}", run_id=run.run_id) for i in range(4)] + [_diagnosis(run)])
    agent = TriageAgent(fake, tools, max_tool_calls=3)
    d = agent.diagnose(Incident.from_run(run))
    tr = agent.last_trace
    assert d.root_cause == "schema_drift"
    assert tr.tool_call_count == 3 and tr.budget_exhausted and tr.tool_calls[-1]["skipped"]
    assert tr.fallback_reason is None
    # every tool_call id got a tool message, and the post-budget turn is answer-only
    assert fake.calls[-1]["messages"].count("tool") == 4
    assert fake.calls[-1]["tool_choice"] == "none" and fake.calls[-1]["json_mode"]


def test_tool_budget_then_garbage_falls_back(tools, run):
    fake = FakeLLM([_call("get_run_logs", f"c{i}", run_id=run.run_id) for i in range(2)] + [LLMResponse(content="?")] * 3)
    agent = TriageAgent(fake, tools, max_tool_calls=1)
    d = agent.diagnose(Incident.from_run(run))
    assert d.root_cause == "unknown"
    assert agent.last_trace.fallback_reason == "tool_budget_exceeded"


def test_max_turns_is_reported_as_such(tools, run):
    fake = FakeLLM([_call("get_run_history", f"c{i}", run_id=run.run_id) for i in range(MAX_TURNS)])
    agent = TriageAgent(fake, tools, max_tool_calls=100)
    d = agent.diagnose(Incident.from_run(run))
    assert d.root_cause == "unknown"
    assert agent.last_trace.fallback_reason == "max_turns_exceeded"


def test_unknown_tool_and_tool_errors_reach_the_model(tools, run):
    fake = FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(id="c1", name="get_secret_logs", arguments={}),
                    ToolCallRequest(id="c2", name="get_run_logs", arguments={"run_id": "run_missing"}),
                ]
            ),
            _diagnosis(run, root_cause="unknown", confidence=0.1, evidence=[]),
        ]
    )
    agent = TriageAgent(fake, tools)
    d = agent.diagnose(Incident.from_run(run))
    assert d.root_cause == "unknown"
    assert [c["is_error"] for c in agent.last_trace.tool_calls] == [True, True]


def test_model_cannot_change_the_run_id(tools, run):
    fake = FakeLLM([_call("get_run_logs", run_id=run.run_id), _diagnosis(run, run_id="run_other")])
    d = TriageAgent(fake, tools).diagnose(Incident.from_run(run))
    assert d.run_id == run.run_id


def test_diagnosis_schema_guardrails():
    base = {"run_id": "x", "root_cause": "unknown", "confidence": 0.5, "proposed_next_step": "check the thing manually"}
    # requires_human_approval is forced True even if the model says False
    d2 = Diagnosis.model_validate({**base, "evidence": [{"source": "run_logs", "quote": "boom!"}], "requires_human_approval": False})
    assert d2.requires_human_approval is True
    # confidence is rounded
    assert Diagnosis.model_validate({**base, "confidence": 0.123456}).confidence == 0.12
    # a real cause needs evidence; unknown may have none
    with pytest.raises(ValueError, match="evidence"):
        Diagnosis.model_validate({**base, "root_cause": "schema_drift", "evidence": []})
    # evidence sources are restricted to the four tools
    with pytest.raises(ValueError):
        Diagnosis.model_validate({**base, "evidence": [{"source": "intuition", "quote": "trust me"}]})
