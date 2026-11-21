"""Unit tests for the synthetic failure generator."""

import pytest

from pipeline_triage.env import Environment
from pipeline_triage.generator import generate_dataset, generate_run
from pipeline_triage.models import ALL_CAUSES, RootCause


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n=90, seed=123)


def test_dataset_size_and_balance(dataset):
    assert len(dataset) == 90
    unknown = [r for r in dataset if r.root_cause == RootCause.UNKNOWN]
    assert len(unknown) == 9  # ~10%
    assert sum(r.ambiguous for r in dataset) >= 5  # some ambiguous cases


def test_every_classifiable_cause_is_represented(dataset):
    seen = {r.root_cause for r in dataset if r.root_cause != RootCause.UNKNOWN}
    assert seen == {c for c in ALL_CAUSES if c != RootCause.UNKNOWN}


def test_all_causes_can_be_generated():
    for cause in ALL_CAUSES:
        run = generate_run(f"run_t_{cause.value}", cause, seed=1)
        assert run.root_cause == cause
        assert run.failed_task in run.logs


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
