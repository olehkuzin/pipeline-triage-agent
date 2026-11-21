"""Synthetic failed-run generator.

Produces N failed runs across the failure taxonomy, seeded for reproducibility,
with noisy logs, genuinely ambiguous cases, and ~`unknown_pct` incidents with no
clear signal (labelled `unknown`).

All timestamps are UTC, so a given (n, seed) yields the same dataset on every
machine regardless of its local timezone.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from ..models import Change, FailedRun, RootCause, RunSummary, TableSchema
from . import builders, common
from .builders import AMBIGUOUS_PAIRS
from .common import PIPELINES, SOURCES, TARGETS, airflow_log_line, make_task_name, rand_hex

EPOCH_START = 1_700_000_000  # 2023-11-14 UTC
EPOCH_SPAN_S = 10_000_000  # ~116 days


def _history_for(
    rng: random.Random, pipeline: str, start: datetime, run_id: str, failed_task: str
) -> list[RunSummary]:
    """Plausible recent history: a streak of daily successes, then this failed run."""
    history = []
    now = start
    for _ in range(rng.randint(6, 10)):
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
            run_id=run_id,
            pipeline=pipeline,
            started_at=start,
            duration_s=rng.uniform(30, 600),
            status="failed",
            failed_task=failed_task,
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


def generate_run(run_id: str, cause: RootCause, seed: int, ambiguous: bool = False) -> FailedRun:
    """Generate a single failed run for a given cause."""
    if ambiguous and cause not in AMBIGUOUS_PAIRS:
        raise ValueError(f"{cause.value} has no ambiguous counterpart; see AMBIGUOUS_PAIRS")

    rng = random.Random(seed)
    start = datetime.fromtimestamp(EPOCH_START + rng.randint(0, EPOCH_SPAN_S), tz=UTC)
    pipeline = rng.choice(PIPELINES)
    source = rng.choice(SOURCES)
    target = rng.choice(TARGETS)
    failed_task = make_task_name(rng, source, target)

    # upstream tasks "succeed" with trivial logs, then the failing task starts
    task_order = list(dict.fromkeys([make_task_name(rng, source, target) for _ in range(2)]))
    task_order = [t for t in task_order if t != failed_task]
    logs: dict[str, list[str]] = {}
    clock = start
    for task in task_order:
        delay = timedelta(seconds=rng.uniform(30, 240))
        logs[task] = [
            airflow_log_line(clock, "INFO", "airflow.task", f"Task {task} started"),
            airflow_log_line(clock + delay, "INFO", "airflow.task", f"Task {task} finished ok"),
        ]
        clock = clock + delay
    task_start = clock + timedelta(seconds=rng.uniform(10, 60))

    builder = builders.BUILDERS[cause] if cause != RootCause.UNKNOWN else builders.no_clear_signal
    context, signal_lines = builder(rng, task_start, source, target, ambiguous=ambiguous)

    # the source table is always in the catalog, whatever went wrong
    schemas: dict[str, TableSchema] = dict(context.get("schemas", {}))
    schemas.setdefault(f"raw.{source}", builders._mk_source(source, 1, common.base_schema(source), None, None))

    changes: list[Change] = list(context.get("changes", []))
    for ch in changes:
        ch.pipeline = pipeline
    changes.extend(_default_changes(rng, pipeline, start))
    changes.sort(key=lambda c: c.committed_at)

    history = _history_for(rng, pipeline, start, run_id, failed_task)

    lines = [airflow_log_line(task_start, "INFO", "airflow.task", f"Task {failed_task} started"), *signal_lines]
    common.add_noise(rng, lines, rng.randint(2, 6), red_herring=rng.random() < common.RED_HERRING_RATE)
    logs[failed_task] = lines

    accepted = [cause, AMBIGUOUS_PAIRS[cause]] if ambiguous else [cause]

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

    ~ambiguous_pct of all runs are ambiguous (both causes of an AMBIGUOUS_PAIRS
    pair are plausible and accepted); ~unknown_pct have no clear signal and are
    labelled `unknown`.
    """
    rng = random.Random(seed)
    classifiable = list(builders.BUILDERS)

    n_unknown = int(round(n * unknown_pct))
    n_classifiable = n - n_unknown
    # distribute classifiable runs as evenly as possible
    quotas: dict[RootCause, int] = {c: n_classifiable // len(classifiable) for c in classifiable}
    for c in classifiable[: n_classifiable % len(classifiable)]:
        quotas[c] += 1

    plan = [cause for cause in classifiable for _ in range(quotas[cause])]
    eligible = [i for i, cause in enumerate(plan) if cause in AMBIGUOUS_PAIRS]
    n_amb = min(int(round(n * ambiguous_pct)), len(eligible))
    amb_marks = set(rng.sample(eligible, n_amb))

    runs: list[FailedRun] = []
    run_seed = seed

    def spawn(cause: RootCause, ambiguous: bool) -> FailedRun:
        nonlocal run_seed
        run_seed += 1
        return generate_run(f"run_{run_seed:05d}", cause, seed=run_seed * 7919, ambiguous=ambiguous)

    for i, cause in enumerate(plan):
        runs.append(spawn(cause, ambiguous=i in amb_marks))
    for _ in range(n_unknown):
        runs.append(spawn(RootCause.UNKNOWN, ambiguous=False))

    rng.shuffle(runs)
    return runs
