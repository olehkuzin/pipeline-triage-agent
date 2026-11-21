"""Diagnose failed runs with the LLM agent.

Usage:
  MISTRAL_API_KEY=... python -m agent.run --run-id run_00042 [--model mistral-large-latest]

Or triage a random sample:
  python -m agent.run --sample 5
"""

from __future__ import annotations

import argparse
import json

from pipeline_triage.env import Environment
from pipeline_triage.mcp_server import ToolError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage-agent")
    parser.add_argument("--dataset", type=str, default=None, help="Path to runs.jsonl; generate if absent.")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, action="append", help="Run id(s) to diagnose. Repeatable.")
    parser.add_argument("--sample", type=int, default=0, help="Diagnose a random sample of N runs.")
    parser.add_argument("--model", type=str, default="mistral-large-latest")
    parser.add_argument("--dump-trace", action="store_true", help="Print the tool-call trace to stderr.")
    args = parser.parse_args(argv)

    if args.dataset:
        env = Environment.load(args.dataset)
    else:
        env = Environment.generate(n=args.n, seed=args.seed)

    from .llm import MistralLLM
    from .loop import TriageAgent

    llm = MistralLLM(model=args.model)
    agent = TriageAgent(env, llm)

    run_ids = list(args.run_id or [])
    if args.sample:
        import random

        run_ids += random.Random(7).sample([r.run_id for r in env.all_runs()], args.sample)
    if not run_ids:
        parser.error("provide --run-id and/or --sample")

    for rid in run_ids:
        diagnosis = agent.diagnose(rid)
        print(json.dumps(diagnosis.model_dump(), indent=2))
        tr = agent.last_trace
        if tr and args.dump_trace:
            import sys

            print(json.dumps(tr.as_dict(), indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
