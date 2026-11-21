"""Evaluation harness.

Scores a predictor (run_id -> {"root_cause": str, "confidence": float, ...})
against ground truth:
accuracy, macro-F1, per-class precision/recall, confusion matrix, and
confidence calibration (is high confidence correlated with correctness?).
Ambiguous cases accept any of `accepted_causes`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..env import Environment

Predictor = Callable[[Environment, str], dict[str, Any]]


@dataclass
class EvalResult:
    n: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]  # rows = truth, cols = predicted
    calibration: dict[str, float] = field(default_factory=dict)  # bucket -> accuracy
    ambiguous_resolved: int = 0  # ambiguous cases where an alternate accepted cause was predicted

    def summary(self) -> str:
        lines = [
            f"n={self.n}",
            f"accuracy={self.accuracy:.3f}  macro_f1={self.macro_f1:.3f}  "
            f"ambiguous_resolved={self.ambiguous_resolved}",
            "",
            f"{'class':<30}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}",
        ]
        for cls, m in sorted(self.per_class.items()):
            lines.append(f"{cls:<30}{m['precision']:>10.2f}{m['recall']:>10.2f}{m['f1']:>10.2f}{m['support']:>9}")
        lines.append("")
        cols = list(self.confusion)
        lines.append(f"{'truth \\ pred':<28}" + "".join(f"{c[:12]:>13}" for c in cols))
        for truth, row in self.confusion.items():
            lines.append(f"{truth[:26]:<28}" + "".join(f"{row.get(c, 0):>13}" for c in cols))
        if self.calibration:
            lines.append("")
            lines.append("calibration (confidence bucket -> accuracy):")
            for bucket, acc in sorted(self.calibration.items()):
                lines.append(f"  {bucket:<6} {acc:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        import dataclasses

        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _bucket(conf: float) -> str:
    if conf >= 0.8:
        return "0.8+"
    if conf >= 0.6:
        return "0.6-0.8"
    if conf >= 0.4:
        return "0.4-0.6"
    return "<0.4"


def evaluate(env: Environment, predictor: Predictor) -> EvalResult:
    runs = env.all_runs()
    classes = sorted({r.root_cause.value for r in runs})

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = {t: defaultdict(int) for t in classes}
    cal_correct: dict[str, int] = defaultdict(int)
    cal_total: dict[str, int] = defaultdict(int)

    correct = 0
    ambiguous_resolved = 0

    for run in runs:
        pred_map = predictor(env, run.run_id)
        pred = str(pred_map["root_cause"])
        conf = float(pred_map.get("confidence", 0.0))

        truth = run.root_cause.value
        accepted = {truth} | {str(a) for a in run.accepted_causes} if run.ambiguous else {truth}

        confusion[truth][pred] += 1
        is_correct = pred in accepted
        if is_correct:
            correct += 1
            tp[truth] += 1
            if pred != truth:
                ambiguous_resolved += 1
        else:
            fp[pred] += 1
            fn[truth] += 1

        cal_total[_bucket(conf)] += 1
        if is_correct:
            cal_correct[_bucket(conf)] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for cls in classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(1 for r in runs if r.root_cause.value == cls)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)

    n = len(runs)
    calibration = {b: cal_correct[b] / cal_total[b] for b in ("0.8+", "0.6-0.8", "0.4-0.6", "<0.4") if cal_total[b]}

    return EvalResult(
        n=n,
        accuracy=correct / n if n else 0.0,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        per_class=per_class,
        confusion={t: dict(row) for t, row in confusion.items()},
        calibration=calibration,
        ambiguous_resolved=ambiguous_resolved,
    )
