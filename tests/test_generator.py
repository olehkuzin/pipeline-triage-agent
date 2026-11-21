"""Unit tests for the synthetic failure generator."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline_triage.env import Environment
from pipeline_triage.generator import generate_dataset, generate_run
from pipeline_triage.generator.builders import AMBIGUOUS_PAIRS
from pipeline_triage.generator.common import log_line_ts
from pipeline_triage.models import ALL_CAUSES, RootCause

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n=90, seed=123)


def test_dataset_size_and_balance(dataset):
    assert len(dataset) == 90
    unknown = [r for r in dataset if r.root_cause == RootCause.UNKNOWN]
    assert len(unknown) == 9  # ~10%
    assert 10 <= sum(r.ambiguous for r in dataset) <= 16  # ~15%


def test_every_classifiable_cause_is_represented(dataset):
    seen = {r.root_cause for r in dataset if r.root_cause != RootCause.UNKNOWN}
    assert seen == {c for c in ALL_CAUSES if c != RootCause.UNKNOWN}


def test_all_causes_can_be_generated():
    for cause in ALL_CAUSES:
        run = generate_run(f"run_t_{cause.value}", cause, seed=1)
        assert run.root_cause == cause
        assert run.failed_task in run.logs
        assert any(t.startswith("raw.") for t in run.schemas)  # the source table is always in the catalog


def test_ambiguous_runs_carry_both_signals(dataset):
    """Ambiguity is real: both accepted causes have evidence in the run, not just a second label."""
    amb = [r for r in dataset if r.ambiguous]
    assert amb
    for run in amb:
        assert run.root_cause in AMBIGUOUS_PAIRS
        assert run.accepted_causes == [run.root_cause, AMBIGUOUS_PAIRS[run.root_cause]]
        joined = "\n".join(run.logs[run.failed_task])
        deploys = [c for c in run.changes if not c.description.startswith(("Docs:", "Chore:", "Style:", "Test:"))]
        drifted = [s for s in run.schemas.values() if s.version >= 2]
        if set(run.accepted_causes) == {RootCause.SCHEMA_DRIFT, RootCause.BAD_DEPLOY}:
            assert deploys and drifted, run.run_id
        else:
            assert "Upstream task" in joined and ("retried" in joined or "Connection reset" in joined), run.run_id
    for run in dataset:
        if not run.ambiguous:
            assert run.accepted_causes == [run.root_cause]


def test_ambiguous_flag_rejected_for_causes_without_a_pair():
    with pytest.raises(ValueError):
        generate_run("run_x", RootCause.DATA_QUALITY_VIOLATION, seed=1, ambiguous=True)


def test_logs_are_chronological_and_utc(dataset):
    for run in dataset:
        assert run.started_at.utcoffset().total_seconds() == 0
        for task, lines in run.logs.items():
            stamps = [log_line_ts(line) for line in lines]
            assert stamps == sorted(stamps), f"{run.run_id}/{task} timestamps go backwards"
            assert all(line.split("]", 1)[0].endswith("+00:00") for line in lines)


def test_first_signal_line_is_kept_and_no_label_hints(dataset):
    for run in dataset:
        joined = "\n".join(line for lines in run.logs.values() for line in lines)
        assert "git blame" not in joined and "latest deployment" not in joined
        if run.root_cause == RootCause.BAD_DEPLOY:
            assert "config version" in joined  # the builder's first line used to be overwritten
        assert sum(f"Task {run.failed_task} started" in line for line in run.logs[run.failed_task]) == 1


def test_history_ends_with_this_run(dataset):
    for run in dataset:
        last = run.run_history[-1]
        assert last.run_id == run.run_id and last.status == "failed" and last.failed_task == run.failed_task
        assert last.started_at == run.started_at


def test_dataset_is_independent_of_local_timezone():
    code = (
        "import hashlib; from pipeline_triage.generator import generate_dataset; "
        "print(hashlib.sha256(''.join(r.model_dump_json() for r in generate_dataset(n=30, seed=4)).encode()).hexdigest())"
    )
    digests = {
        tz: subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "TZ": tz, "PYTHONPATH": str(REPO_ROOT)},
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for tz in ("UTC", "Asia/Tokyo", "America/Los_Angeles")
    }
    assert len(set(digests.values())) == 1, digests


def test_reproducibility():
    a = generate_dataset(n=40, seed=7)
    b = generate_dataset(n=40, seed=7)
    assert [r.model_dump_json() for r in a] == [r.model_dump_json() for r in b]


def test_logs_contain_signal_lines(dataset):
    """Every classifiable run should have more than a trivial empty log."""
    for run in dataset:
        assert run.logs, f"{run.run_id} has no logs"
        failing = run.logs[run.failed_task]
        assert len(failing) >= 3
        joined = "\n".join(failing)
        if run.root_cause == RootCause.SCHEMA_DRIFT:
            assert "Column" in joined or "cast" in joined
        if run.root_cause == RootCause.LATE_OR_MISSING_DATA:
            assert "Sensor" in joined or "Partition" in joined
        if run.root_cause == RootCause.RESOURCE_EXHAUSTION:
            assert any(k in joined for k in ("OutOfMemory", "No space left", "SIGKILL"))
        if run.root_cause == RootCause.PERMISSION_OR_AUTH:
            assert any(k in joined for k in ("403", "expired", "denied", "Login failed"))
        if run.root_cause == RootCause.DATA_QUALITY_VIOLATION:
            assert "QUALITY CHECK FAILED" in joined
        if run.root_cause == RootCause.UNKNOWN:
            assert "QUALITY CHECK FAILED" not in joined


def test_health_schema_versioning_and_history(dataset):
    drift = [r for r in dataset if r.root_cause == RootCause.SCHEMA_DRIFT][0]
    schema = list(drift.schemas.values())[0]
    assert schema.version >= 2
    assert schema.changed_at is not None

    deploys = [r for r in dataset if r.root_cause == RootCause.BAD_DEPLOY]
    for run in deploys:
        recent = [c for c in run.changes if (run.started_at - c.committed_at).total_seconds() < 48 * 3600]
        assert recent, "bad_deploy runs should have a change shortly before the run"


def test_env_save_load_roundtrip(tmp_path):
    env = Environment.generate(n=40, seed=3)
    p = tmp_path / "runs.jsonl"
    env.save(p)
    loaded = Environment.load(p)
    assert {r.run_id for r in loaded.all_runs()} == {r.run_id for r in env.all_runs()}
    orig = {r.run_id: r for r in env.all_runs()}
    for r in loaded.all_runs():
        assert r.model_dump_json() == orig[r.run_id].model_dump_json()
