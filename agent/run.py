"""Diagnose failed runs with the LLM agent, using tools through the MCP server over stdio.

Usage:
  MISTRAL_API_KEY=... python -m agent.run --run-id run_00042 [--model mistral-large-latest]

Or triage a random sample:
  python -m agent.run --sample 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from pipeline_triage.env import Environment


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

    if args.dataset and not Path(args.dataset).exists():
        parser.error(f"dataset not found: {args.dataset}")
    env = Environment.load(args.dataset) if args.dataset else Environment.generate(n=args.n, seed=args.seed)

    run_ids = list(args.run_id or [])
    if args.sample:
        run_ids += random.Random(7).sample([r.run_id for r in env.all_runs()], args.sample)
    if not run_ids:
        parser.error("provide --run-id and/or --sample")
    unknown = [rid for rid in run_ids if env.get_run(rid) is None]
    if unknown:
        parser.error(f"run id(s) not in the dataset: {', '.join(unknown)}")

    from .llm import MistralLLM
    from .loop import TriageAgent
    from .schemas import Incident
    from .tools import MCPToolClient

    llm = MistralLLM(model=args.model)
    # the server process builds the same seeded dataset (or loads the same file)
    with MCPToolClient.stdio(dataset=args.dataset, n=args.n, seed=args.seed) as tools:
        agent = TriageAgent(llm, tools)
        for rid in run_ids:
            diagnosis = agent.diagnose(Incident.from_run(env.get_run(rid)))
            print(json.dumps(diagnosis.model_dump(mode="json"), indent=2))
            if args.dump_trace and agent.last_trace:
                print(json.dumps(agent.last_trace.as_dict(), indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
