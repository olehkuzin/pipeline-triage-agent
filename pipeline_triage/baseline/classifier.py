"""Rule-based baseline classifier.

Consumes the same evidence the LLM agent will see (MCP tool outputs), then
applies ordered keyword/regex rules. Serves as the benchmark the agent must beat.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..env import Environment
from ..mcp_server import get_recent_changes, get_run_history, get_run_logs, get_table_schema, ToolError


@dataclass
class Prediction:
    root_cause: str
    confidence: float
    matched_rules: list[str] = field(default_factory=list)


TABLE_RE = re.compile(r"[\w.]*raw\.[a-z_]+")
TASK_RE = re.compile(r"FAILING TASK: ([\w]+)")

# Rules are tried in order (first match wins at that tier).
# Each rule: (name, regex, cause, confidence)
LOG_RULES: list[tuple[str, re.Pattern[str], str, float]] = [
    ("upstream_fail", re.compile(r"UpstreamTaskFailure|Upstream task .* failed|missing dependency output", re.I),
     "upstream_dependency_failure", 0.85),
    ("dq_check", re.compile(r"QUALITY CHECK FAILED|DQ gate failed", re.I),
     "data_quality_violation", 0.85),
    ("sensor_timeout", re.compile(r"Sensor timeout|S3KeySensor.*Timeout|Partition not found", re.I),
     "late_or_missing_data", 0.85),
    ("oom_disk", re.compile(r"OutOfMemoryError|No space left on device|exit code 137|ExecutorLostFailure", re.I),
     "resource_exhaustion", 0.85),
    ("auth", re.compile(r"403|AccessDenied|ExpiredToken|security token .* expired|Login failed|Kerberos ticket expired|not authorized", re.I),
     "permission_or_auth", 0.85),
    ("schema_col_missing", re.compile(r"Column '.*' not found in source table|Did you mean|Cannot cast column", re.I),
     "schema_drift", 0.7),  # could also be bad_deploy; schema check disambiguates below
    ("retry_succeeded", re.compile(r"succeeded after retry", re.I),
     "transient_infra", 0.8),
    ("net_retry", re.compile(r"ConnectionResetError|Read timeout .* retrying|server closed the connection unexpectedly", re.I),
     "transient_infra", 0.7),
]

# Deploy-flavored runtime errors: strong signal when a recent change exists.
DEPLOY_ERRORS = re.compile(r"KeyError:|TypeError: argument of type|strict_schema_validation|KeyError|NameError|ImportError|AttributeError", re.I)

RETRIES_ONLY = re.compile(r"Attempt \d+/\d+ exited with return code 1|All \d+ attempts failed", re.I)


def classify_run(env: Environment, run_id: str) -> Prediction:
    """Classify root cause from evidence only (logs + recent changes + schemas)."""
    run = env.get_run(run_id)
    assert run is not None
    logs = get_run_logs(env, run_id, tail_lines=200)

    matched: list[str] = []

    def schema_changed_for_log_tables() -> str | None:
        for table in TABLE_RE.findall(logs):
            try:
                schema = json.loads(get_table_schema(env, table))
            except ToolError:
                continue
            if schema.get("version", 0) > 1 and schema.get("changed_at"):
                return table
        return None

    def recent_deploy() -> bool:
        try:
            changes = json.loads(get_recent_changes(env, run.pipeline, since_hours=48))
        except ToolError:
            return False
        recent = [c for c in changes if "typing" not in c.get("description", "").lower()
                 and not c["description"].startswith(("Docs:", "Chore:", "Style:", "Test:"))]
        return bool(recent)

    # 1) Direct log-signal rules, cheapest and most specific first.
    for name, pattern, cause, conf in LOG_RULES:
        if pattern.search(logs):
            matched.append(name)
            return Prediction(cause, conf, matched)

    # 2) Error string + recent meaningful change => bad_deploy.
    if DEPLOY_ERRORS.search(logs) and recent_deploy():
        matched.append("deploy_error_recent_change")
        return Prediction("bad_deploy", 0.75, matched)

    # 3) Missing-column error without schema version bump, but with a recent
    #    meaningful change => also plausibly bad_deploy (schema_drift was tried first).
    if schema_changed_for_log_tables():
        matched.append("schema_version_bumped")
        return Prediction("schema_drift", 0.8, matched)

    # 4) A recent meaningful change with any error => bad_deploy.
    if recent_deploy() and "ERROR" in logs:
        matched.append("recent_change_with_error")
        return Prediction("bad_deploy", 0.6, matched)

    # 5) Generic retries and nothing else => unknown.
    if RETRIES_ONLY.search(logs):
        matched.append("generic_retries")
        return Prediction("unknown", 0.4, matched)

    return Prediction("unknown", 0.2, matched + ["no_rule_matched"])


def classify_dataset(env: Environment) -> list[tuple[str, Prediction]]:
    return [(run.run_id, classify_run(env, run.run_id)) for run in env.all_runs()]
