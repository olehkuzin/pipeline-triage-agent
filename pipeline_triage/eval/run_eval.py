"""Run the evaluation for the rule-based baseline or the LLM agent.

Usage:
  python -m pipeline_triage.eval.run_eval [--n 120] [--seed 42] [--report eval/reports/baseline.json]
  MISTRAL_API_KEY=... python -m pipeline_triage.eval.run_eval --predictor agent --limit 30 \\
      --report eval/reports/agent.json

The agent reaches its tools through the real MCP server, spawned over stdio on
the same dataset. `--limit` picks a deterministic, class-stratified subset
(useful because every agent incident costs LLM calls).

Environment can also be pre-generated and saved via:
  python -m pipeline_triage.eval.run_eval --save-dataset data/eval_runs.jsonl
  python -m pipeline_triage.eval.run_eval --dataset data/eval_runs.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..baseline.classifier import classify_run
from ..env import Environment
from .harness import Predictor, evaluate


def stratified_subset(env: Environment, limit: int) -> list[str]:
    """Deterministic subset of `limit` run ids, round-robin across root causes."""
    by_cause: dict[str, list[str]] = defaultdict(list)
    for run in sorted(env.all_runs(), key=lambda r: r.run_id):
        by_cause[run.root_cause.value].append(run.run_id)
    queues = [by_cause[c] for c in sorted(by_cause)]
    picked: list[str] = []
    while len(picked) < limit and any(queues):
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
    return picked


def baseline_predictor(env: Environment, run_id: str) -> dict[str, Any]:
    return dataclasses.asdict(classify_run(env, run_id))


def agent_predictor(agent: Any) -> Predictor:
    """Wrap a TriageAgent as a harness predictor reporting operational fields."""
    from agent.schemas import Incident

    def predict(env: Environment, run_id: str) -> dict[str, Any]:
        run = env.get_run(run_id)
        assert run is not None
        diagnosis = agent.diagnose(Incident.from_run(run))
        trace = agent.last_trace
        return {
            "root_cause": diagnosis.root_cause.value,
            "confidence": diagnosis.confidence,
            "tool_calls": trace.tool_call_count,
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "wall_time_s": trace.wall_time_s,
            "fallback": trace.fallback_reason is not None,
        }

    return predict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage-eval")
    parser.add_argument("--predictor", choices=["baseline", "agent"], default="baseline")
    parser.add_argument("--model", type=str, default="mistral-large-latest", help="LLM for --predictor agent.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to a saved runs.jsonl dataset (eval set).")
    parser.add_argument("--n", type=int, default=120, help="Dataset size when generating an eval set.")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed when generating.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate a class-stratified subset of N runs.")
    parser.add_argument(
        "--save-dataset", type=str, default=None, help="Also save the generated dataset to this .jsonl path."
    )
    parser.add_argument("--report", type=str, default=None, help="Write the result as JSON here.")
    args = parser.parse_args(argv)

    if args.dataset:
        if not Path(args.dataset).exists():
            parser.error(f"dataset not found: {args.dataset}")
        env = Environment.load(args.dataset)
        print(f"Loaded dataset with {len(env.all_runs())} runs from {args.dataset}")
    else:
        env = Environment.generate(n=args.n, seed=args.seed)
        print(f"Generated dataset with {len(env.all_runs())} runs (n={args.n}, seed={args.seed})")
        if args.save_dataset:
            env.save(args.save_dataset)
            print(f"Saved dataset to {args.save_dataset}")

    run_ids = stratified_subset(env, args.limit) if args.limit else None

    t0 = time.perf_counter()
    if args.predictor == "baseline":
        result = evaluate(env, baseline_predictor, run_ids)
    else:
        from agent.llm import MistralLLM
        from agent.loop import TriageAgent
        from agent.tools import MCPToolClient

        with MCPToolClient.stdio(dataset=args.dataset, n=args.n, seed=args.seed) as tools:
            agent = TriageAgent(MistralLLM(model=args.model), tools)
            result = evaluate(env, agent_predictor(agent), run_ids)
    elapsed = time.perf_counter() - t0

    print(result.summary())
    print(f"\n{args.predictor} evaluated {result.n} runs in {elapsed:.1f}s")

    if args.report:
        result.save(args.report)
        print(f"Report saved to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
