"""In-memory environment holding the generated dataset.

Single source of truth shared by generator, MCP server, baseline, and eval.
"""

from __future__ import annotations

import json
from pathlib import Path

from .generator import generate_dataset
from .models import FailedRun


class Environment:
    def __init__(self, runs: list[FailedRun]):
        self._runs: dict[str, FailedRun] = {r.run_id: r for r in runs}

    @classmethod
    def generate(cls, n: int = 120, seed: int = 42, **kwargs) -> "Environment":
        return cls(generate_dataset(n=n, seed=seed, **kwargs))

    def get_run(self, run_id: str) -> FailedRun | None:
        return self._runs.get(run_id)

    def all_runs(self) -> list[FailedRun]:
        return list(self._runs.values())

    @classmethod
    def load(cls, path: str | Path) -> "Environment":
        runs = [FailedRun.model_validate(json.loads(line)) for line in Path(path).read_text().splitlines() if line]
        return cls(runs)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in self._runs.values():
                f.write(r.model_dump_json() + "\n")
