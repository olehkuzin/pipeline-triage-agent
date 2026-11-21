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


def test_baseline_beats_majority_but_is_not_saturated(env):
    result = evaluate(env, _pred)
    assert result.n == 120
    # Any sensible triage should get well above 8% (majority class share)...
    assert result.accuracy > 0.6 and result.macro_f1 > 0.6
    # ...but benign lines with other categories' vocabulary keep a keyword matcher below perfect,
    # leaving room for a model that reads context to show what it adds.
    assert result.accuracy < 0.98
    # with run-scoped tools the culprit commit is visible, so deploys are no longer invisible
    assert result.per_class["bad_deploy"]["recall"] >= 0.5


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


def test_predictions_and_operational_metrics(env):
    run_ids = [r.run_id for r in env.all_runs()[:10]]

    def pred(e, run_id):
        return {"root_cause": "unknown", "confidence": 0.3, "tool_calls": 2, "wall_time_s": 0.5, "fallback": run_id == run_ids[0]}

    result = evaluate(env, pred, run_ids)
    assert result.n == 10
    assert [p["run_id"] for p in result.predictions] == run_ids
    assert result.operational == {"tool_calls": 2.0, "wall_time_s": 0.5, "fallback_rate": 0.1}
    assert "operational" in result.summary()


def test_stratified_subset_covers_every_class(env):
    from pipeline_triage.eval.run_eval import stratified_subset

    picked = stratified_subset(env, 18)
    assert len(picked) == len(set(picked)) == 18
    assert stratified_subset(env, 18) == picked  # deterministic
    causes = {env.get_run(rid).root_cause for rid in picked}
    assert causes == set(RootCause)


def test_red_herrings_are_present_but_a_minority(env):
    from pipeline_triage.generator.common import RED_HERRING_LINES

    herrings = {msg for _, _, msg in RED_HERRING_LINES}
    hit = sum(any(h in line for h in herrings for line in r.logs[r.failed_task]) for r in env.all_runs())
    n = len(env.all_runs())
    assert 0.15 * n < hit < 0.6 * n
