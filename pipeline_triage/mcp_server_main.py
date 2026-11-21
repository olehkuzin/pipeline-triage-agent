"""Run the triage MCP server over stdio.

Usage:
  python -m pipeline_triage.mcp_server [--dataset runs.jsonl] [--n 120] [--seed 42]

Without --dataset, a dataset is generated in-memory (seeded).
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from .env import Environment
from .mcp_server import ToolError, create_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline-triage-mcp")
    parser.add_argument("--dataset", type=str, default=None, help="Path to runs.jsonl (previously saved dataset).")
    parser.add_argument("--n", type=int, default=120, help="Dataset size when generating.")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed when generating.")
    args = parser.parse_args(argv)

    if args.dataset:
        env = Environment.load(args.dataset)
    else:
        env = Environment.generate(n=args.n, seed=args.seed)

    print(
        f"Starting pipeline-triage MCP server over stdio "
        f"({'dataset: ' + args.dataset if args.dataset else f'generated: n={args.n} seed={args.seed}'} "
        f"with {len(env.all_runs())} runs)...",
        file=sys.stderr,
    )
    server = create_server(env)
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
