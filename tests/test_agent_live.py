"""Live integration test against the real Mistral API.

Skipped unless MISTRAL_API_KEY is set:

    MISTRAL_API_KEY=... .venv/bin/python -m pytest tests/test_agent_live.py -s -q

Uses a small generated dataset and triages a handful of runs end-to-end
(real chat completions, real tool-calling through the MCP server over stdio,
validated structured output).
"""

import os

import pytest

from pipeline_triage.env import Environment
from pipeline_triage.models import RootCause

pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set (live test)",
)

MODEL = os.environ.get("TRIAGE_AGENT_MODEL", "mistral-large-latest")


N, SEED = 80, 13


@pytest.fixture(scope="module")
def env():
    return Environment.generate(n=N, seed=SEED)


@pytest.fixture(scope="module")
def agent():
    from agent.llm import MistralLLM
    from agent.loop import TriageAgent
    from agent.tools import MCPToolClient

    with MCPToolClient.stdio(n=N, seed=SEED) as tools:  # the server builds the same seeded dataset
        yield TriageAgent(MistralLLM(model=MODEL), tools)


def test_live_single_incident(env, agent):
    from agent.schemas import Incident

    run = next(r for r in env.all_runs() if r.root_cause == RootCause.SCHEMA_DRIFT and not r.ambiguous)
    d = agent.diagnose(Incident.from_run(run))
    print(f"root_cause={d.root_cause} conf={d.confidence}")
    print("evidence[0]:", d.evidence[0].quote[:80])
    print("next_step:", d.proposed_next_step[:100])
    tr = agent.last_trace
    print(f"trace: turns={tr.turns} tool_calls={tr.tool_call_count} tokens={tr.prompt_tokens}/{tr.completion_tokens}")
    assert d.run_id == run.run_id
    assert d.root_cause == "schema_drift"
    assert 0.3 <= d.confidence <= 1.0


def test_live_small_sample(env, agent):
    """A few runs across causes; accuracy sanity check (not a strict eval)."""
    import random

    from agent.schemas import Incident

    runs = random.Random(3).sample(env.all_runs(), 5)
    correct = fallbacks = 0
    for run in runs:
        d = agent.diagnose(Incident.from_run(run))
        tr = agent.last_trace
        fallbacks += tr.fallback_reason is not None
        for reason in tr.rejections:
            print(f"  {run.run_id} rejected answer: {reason[:160]}")
        accepted = {run.root_cause.value} | {a.value for a in run.accepted_causes}
        ok = d.root_cause.value in accepted
        correct += ok
        print(f"{run.run_id}: truth={run.root_cause.value:<30} pred={d.root_cause.value:<30} ok={ok}")
    assert correct >= 3  # expect most to be right on synthetic signals
    assert fallbacks <= 1  # the repair loop should rescue nearly every malformed answer
