"""Tests for the baseline classifier and eval harness."""

import pytest

from pipeline_triage.baseline.classifier import classify_run
from pipeline_triage.env import Environment
from pipeline_triage.eval.harness import evaluate
from pipeline_triage.models import RootCause


@pytest.fixture(scope="module")
def env():
    return Environment.generate(n=120, seed=42)


def _pred(env, run_id):
    from dataclasses import asdict

    return asdict(classify_run(env, run_id))


def test_baseline_beats_majority(env):
    result = evaluate(env, _pred)
    assert result.n == 120
    # Any sensible triage should get well above 8% (majority class share)
    assert result.accuracy > 0.5
    # Distinctive causes should be near-perfect for a keyword baseline
    for cls in ("schema_drift", "late_or_missing_data", "permission_or_auth", "resource_exhaustion", "data_quality_violation", "upstream_dependency_failure"):
        assert result.per_class[cls]["recall"] >= 0.9, f"{cls} recall too low"


def test_confusion_matrix_shape(env):
    result = evaluate(env, _pred)
    classes = {c.value for c in RootCause}
    for truth, row in result.confusion.items():
        assert truth in classes
        assert sum(row.values()) == result.per_class[truth]["support"]
    assert sum(sum(r.values()) for r in result.confusion.values()) == result.n


def test_reproducible(env):
    r1 = evaluate(env, _pred)
    r2 = evaluate(env, _pred)
    assert r1.to_dict() == r2.to_dict()


def test_perfect_predictor_scores_full_marks(env):
    def oracle(e, run_id):
        truth = e.get_run(run_id).root_cause
        return {"root_cause": truth.value, "confidence": 0.99}

    result = evaluate(env, oracle)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.calibration["0.8+"] == 1.0
    assert result.ambiguous_resolved == 0


def test_ambiguous_accepts_alternates(env):
    """When the predictor returns the alternate accepted cause, it counts as correct."""
    amb = next(r for r in env.all_runs() if r.ambiguous and len(r.accepted_causes) > 1)

    # choose the second accepted cause for one ambiguous run
    def alt2(e, run_id):
        run = e.get_run(run_id)
        if run.run_id == amb.run_id:
            return {"root_cause": run.accepted_causes[1].value, "confidence": 0.7}
        return {"root_cause": run.root_cause.value, "confidence": 0.7}

    result = evaluate(env, alt2)
    # ambiguous case was credited despite being 'wrong' vs primary label
    assert result.accuracy == 1.0
    assert result.ambiguous_resolved >= 1


def test_calibration_buckets_error_more_at_low_conf(env):
    """A predictor that always guesses wrong with low confidence should show bad calibration."""
    def wrong(e, run_id):
        return {"root_cause": "unknown" if e.get_run(run_id).root_cause != RootCause.UNKNOWN else "bad_deploy", "confidence": 0.2}

    result = evaluate(env, wrong)
    assert result.accuracy < 0.2
    assert result.calibration["<0.4"] < 0.2
