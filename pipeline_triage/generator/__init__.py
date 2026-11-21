"""Synthetic failed-run generator.

Produces N failed runs across the failure taxonomy, seeded for reproducibility,
with noisy logs, ambiguous cases, and ~`unknown_pct` unlabeled-cause incidents.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import datetime, timedelta

from ..models import Change, FailedRun, RootCause, RunSummary, TableSchema
from . import builders, common
from .common import PIPELINES, SOURCES, TARGETS, airflow_log_line, make_task_name, rand_hex


def _relabel_id_col(columns: list[tuple[str, str]], source: str) -> list[tuple[str, str]]:
    return columns


def _history_for(rng: random.Random, pipeline: str, start: datetime, failed: bool) -> list[RunSummary]:
    """Small plausible run history; most runs succeed."""
    history = []
    now = start
    for i in range(rng.randint(6, 10)):
        now = now - timedelta(hours=rng.uniform(20, 26))
        history.append(
            RunSummary(
                run_id=f"run_{rand_hex(rng, 10)}",
                pipeline=pipeline,
                started_at=now,
                duration_s=rng.uniform(120, 900),
                status="success",
            )
        )
    history.reverse()
    history.append(
        RunSummary(
            run_id=f"run_{rand_hex(rng, 10)}",
            pipeline=pipeline,
            started_at=start,
            duration_s=rng.uniform(30, 600),
            status="failed",
            failed_task=None,
        )
    )
    return history


def _default_changes(rng: random.Random, pipeline: str, start: datetime) -> list[Change]:
    """Random old changes (usually unrelated to the failure) for noise."""
    changes = []
    for _ in range(rng.randint(0, 2)):
        changes.append(
            Change(
                pipeline=pipeline,
                kind=rng.choice(["code_change", "config_change"]),
                description=rng.choice([
                    "Docs: update README runbook",
                    "Chore: bump dependency minor version",
                    "Style: black formatting for tasks.py",
                    "Test: add unit test for validation step",
                ]),
                committed_at=start - timedelta(days=rng.uniform(3, 30)),
                author=rng.choice(common.ENGINEERS),
                commit_sha=rand_hex(rng),
            )
        )
    return changes


def _gen_rules_for_cause(cause: RootCause, rng: random.Random, start: datetime, source: str):
    """Modifier patterns that produce a context change, overriding runs/history."""
    if cause == RootCause.SCHEMA_DRIFT:
        return {}, None
    return {}, None


def generate_run(run_id: str, cause: RootCause, seed: int, ambiguous: bool = False) -> FailedRun:
    """Generate a single failed run for a given cause."""
    rng = random.Random(seed)
    start = datetime.fromtimestamp(1700000000 + rng.randint(0, 10_000_000))  # deterministic 2023-2024 window
    pipeline = rng.choice(PIPELINES)
    source = rng.choice(SOURCES)
    target = rng.choice(TARGETS)
    failed_task = make_task_name(rng, source, target)

    builder = builders.BUILDERS[cause] if cause != RootCause.UNKNOWN else builders.no_clear_signal
    context, signal_lines = builder(rng, start, source, target)

    schemas: dict[str, TableSchema] = context.get("schemas", {})
    # ensure the source table exists even if the builder didn't provide schema context
    if not schemas and cause != RootCause.LATE_OR_MISSING_DATA:
        schemas[f"raw.{source}"] = builders._mk_source(source, 1, common.base_schema(source), None, None)

    changes: list[Change] = context.get("changes", [])
    for ch in changes:
        ch.pipeline = pipeline
    changes.extend(_default_changes(rng, pipeline, start))
    changes.sort(key=lambda c: c.committed_at)

    history = _history_for(rng, pipeline, start, failed=True)
    history[-1].failed_task = failed_task

    # Assemble per-task logs
    logs: dict[str, list[str]] = {}
    task_order = [make_task_name(rng, source, target) for _ in range(2)] + [failed_task]
    task_order = list(dict.fromkeys(task_order))
    if task_order[-1] != failed_task:
        task_order.append(failed_task)

    # upstream tasks "succeed" with trivial logs
    prev = start
    for t in task_order[:-1]:
        delay = timedelta(seconds=rng.uniform(30, 240))
        logs[t] = [
            airflow_log_line(prev, "INFO", "airflow.task", f"Task {t} started"),
            airflow_log_line(prev + delay, "INFO", "airflow.task", f"Task {t} finished ok"),
        ]
        prev = prev + delay
    prev = prev + timedelta(seconds=rng.uniform(10, 60))

    noise_n = rng.randint(2, 6)
    lines = []
    for i, ln in enumerate(signal_lines):
        lines.append(airflow_log_line(prev, "INFO", "airflow.task", f"Task {failed_task} started") if i == 0 else ln)
    common.add_noise(rng, lines, noise_n, prev)
    logs[failed_task] = lines

    accepted = [cause]
    if ambiguous:
        # schema_drift and bad_deploy share "column not found"-style errors
        if cause == RootCause.BAD_DEPLOY:
            accepted = [RootCause.BAD_DEPLOY, RootCause.SCHEMA_DRIFT]
        elif cause == RootCause.SCHEMA_DRIFT:
            accepted = [RootCause.SCHEMA_DRIFT, RootCause.BAD_DEPLOY]
        elif cause == RootCause.TRANSIENT_INFRA:
            accepted = [RootCause.TRANSIENT_INFRA, RootCause.UPSTREAM_DEPENDENCY_FAILURE]
        elif cause == RootCause.UPSTREAM_DEPENDENCY_FAILURE:
            accepted = [RootCause.UPSTREAM_DEPENDENCY_FAILURE, RootCause.TRANSIENT_INFRA]

    return FailedRun(
        run_id=run_id,
        pipeline=pipeline,
        started_at=start,
        failed_task=failed_task,
        root_cause=cause,
        ambiguous=ambiguous,
        accepted_causes=accepted,
        logs=logs,
        schemas=schemas,
        changes=changes,
        run_history=history,
    )


def generate_dataset(
    n: int = 120,
    seed: int = 42,
    ambiguous_pct: float = 0.15,
    unknown_pct: float = 0.10,
) -> list[FailedRun]:
    """Generate a balanced dataset of failed runs.

    ~ambiguous_pct are ambiguous (either of two causes accepted),
    ~unknown_pct have no clear signal and are labeled `unknown`.
    """
    rng = random.Random(seed)
    classifiable = [c for c in builders.BUILDERS]
    runs: list[FailedRun] = []

    n_unknown = int(round(n * unknown_pct))
    n_classifiable = n - n_unknown
    # distribute classifiable runs as evenly as possible
    quotas: dict[RootCause, int] = {c: n_classifiable // len(classifiable) for c in classifiable}
    for c in classifiable[:n_classifiable % len(classifiable)]:
        quotas[c] += 1

    run_counter = 0
    per_run_seed = seed

    def spawn(cause: RootCause, ambiguous: bool) -> FailedRun:
        nonlocal run_counter, per_run_seed
        run_counter += 1
        per_run_seed += 1
        return generate_run(f"run_{per_run_seed:05d}", cause, seed=per_run_seed * 7919, ambiguous=ambiguous)

    n_amb = int(round(n_classifiable * ambiguous_pct))
    # mark a deterministic subset as ambiguous
    amb_marks = set()
    while len(amb_marks) < n_amb:
        amb_marks.add(rng.randrange(n_classifiable))

    idx = 0
    for cause in classifiable:
        for _ in range(quotas[cause]):
            runs.append(spawn(cause, ambiguous=idx in amb_marks))
            idx += 1

    for _ in range(n_unknown):
        runs.append(spawn(RootCause.UNKNOWN, ambiguous=False))

    rng.shuffle(runs)
    return runs
