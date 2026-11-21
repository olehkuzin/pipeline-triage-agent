"""Live integration test against the real Mistral API.

Skipped unless MISTRAL_API_KEY is set:

    MISTRAL_API_KEY=... .venv/bin/python -m pytest tests/test_agent_live.py -s -q

Uses a small generated dataset and triages a handful of runs end-to-end
(real chat completions, real tool-calling, validated structured output).
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


@pytest.fixture(scope="module")
def agent():
    from agent.llm import MistralLLM
    from agent.loop import TriageAgent

    env = Environment.generate(n=80, seed=13)
    return TriageAgent(env, MistralLLM(model=MODEL))


def test_live_single_incident(agent):
    run = next(r for r in agent.env.all_runs() if r.root_cause == RootCause.SCHEMA_DRIFT)
    d = agent.diagnose(run.run_id)
    print(f"root_cause={d.root_cause} conf={d.confidence}")
    print("evidence[0]:", d.evidence[0].quote[:80])
    print("next_step:", d.proposed_next_step[:100])
    tr = agent.last_trace
    print(f"trace: turns={tr.turns} tool_calls={tr.tool_call_count} tokens={tr.prompt_tokens}/{tr.completion_tokens}")
    assert d.run_id == run.run_id
    assert d.root_cause == "schema_drift"
    assert 0.3 <= d.confidence <= 1.0


def test_live_small_sample(agent):
    """A few runs across causes; accuracy sanity check (not a strict eval)."""
    import random

    runs = random.Random(3).sample(agent.env.all_runs(), 5)
    correct = 0
    for run in runs:
        d = agent.diagnose(run.run_id)
        accepted = {run.root_cause.value} | {a.value for a in run.accepted_causes}
        ok = d.root_cause.value in accepted
        correct += ok
        print(f"{run.run_id}: truth={run.root_cause.value:<30} pred={d.root_cause.value:<30} ok={ok}")
    assert correct >= 3  # expect most to be right on synthetic signals
