"""Run the evaluation for the baseline classifier.

Usage:
  python -m pipeline_triage.eval.run_eval [--n 120] [--seed 42] [--report reports/baseline.json]

Environment can also be pre-generated and saved via:
  python -c "from pipeline_triage.env import Environment; Environment.generate(n=120).save('data/eval_runs.jsonl')"
  python -m pipeline_triage.eval.run_eval --dataset data/eval_runs.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from ..baseline.classifier import classify_run
from ..env import Environment
from .harness import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage-eval")
    parser.add_argument("--dataset", type=str, default=None, help="Path to a saved runs.jsonl dataset (eval set).")
    parser.add_argument("--n", type=int, default=120, help="Dataset size when generating an eval set.")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed when generating.")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--save-dataset", type=str, default=None, help="Also save the generated dataset to this .jsonl path (for reproducible evals).")
    parser.add_argument("--report", type=str, default=None, help="Write the result as JSON here.")
    args = parser.parse_args(argv)

    if args.dataset and Path(args.dataset).exists():
        env = Environment.load(args.dataset)
        print(f"Loaded dataset with {len(env.all_runs())} runs from {args.dataset}")
    else:
        env = Environment.generate(n=args.n, seed=args.seed)
        print(f"Generated dataset with {len(env.all_runs())} runs (n={args.n}, seed={args.seed})")
        if args.save_dataset:
            env.save(args.save_dataset)
            print(f"Saved dataset to {args.save_dataset}")

    t0 = time.perf_counter()
    result = evaluate(env, lambda e, run_id: dataclasses.asdict(classify_run(e, run_id)))
    elapsed = time.perf_counter() - t0

    print(result.summary())
    print(f"\nBaseline evaluated {result.n} runs in {elapsed:.1f}s")

    if args.report:
        result.save(args.report)
        print(f"Report saved to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
