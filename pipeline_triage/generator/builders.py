"""Per-cause failure builders.

Each builder returns (context_dict, signal_lines) where
- context_dict feeds schemas/changes/run_history on the FailedRun model, and
- signal_lines are log lines inserted into the failing task's logs.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from ..models import Change, ColumnSpec, RootCause, RunSummary, TableSchema
from . import common


def _seq_ts(start: datetime, i: int, step_s: int = 45) -> datetime:
    return start + timedelta(seconds=step_s * i)


def _mk_source(source: str, version: int, columns: list[tuple[str, str]], changed_at: datetime | None, desc: str | None) -> TableSchema:
    return TableSchema(
        table=f"raw.{source}",
        columns=[ColumnSpec(name=n, dtype=d) for n, d in columns],
        version=version,
        changed_at=changed_at,
        change_description=desc,
    )


def schema_drift(rng: random.Random, start: datetime, source: str, target: str):
    """Upstream column renamed/removed/type-changed right before the run."""
    old_columns = common.base_schema(source)
    variant = rng.random()
    drift_desc = None
    new_columns = list(old_columns)
    if variant < 0.5:
        # rename
        victim = rng.choice([c for c in new_columns if not c[0].endswith("_id")])
        renamed = victim[0] + "_v2"
        new_columns = [(renamed if n == victim[0] else n, d) for n, d in new_columns]
        drift_desc = f"column '{victim[0]}' renamed to '{renamed}' upstream"
        missing, actual = victim[0], renamed
    elif variant < 0.8:
        # removal
        victim = rng.choice([c for c in new_columns if not c[0].endswith("_id")])
        new_columns = [c for c in new_columns if c[0] != victim[0]]
        drift_desc = f"column '{victim[0]}' no longer provided upstream"
        missing, actual = victim[0], None
    else:
        # type change
        victim = rng.choice(new_columns)
        dtype = "string" if victim[1] != "string" else "int64"
        new_columns = [(n, dtype if n == victim[0] else d) for n, d in new_columns]
        drift_desc = f"column '{victim[0]}' dtype changed from {victim[1]} to {dtype}"
        missing, actual = victim[0], None

    schema = _mk_source(source, 2, new_columns, start - timedelta(hours=rng.uniform(1, 5)), drift_desc)
    col = missing if variant >= 0.5 else missing
    lines = [
        common.airflow_log_line(start, "INFO", "lib.loader", f"Reading partition run_date={start.date().isoformat()} from raw.{source}"),
    ]
    if variant < 0.8:
        msg = (
            f"AnalysisException: Column '{col}' not found in source table raw.{source}. "
            + (f"Did you mean '{actual}'?" if actual else "")
        )
    else:
        msg = f"InvalidInput: Cannot cast column '{col}' to expected type; source now returns {dtype} values"
    lines += [common.airflow_log_line(_seq_ts(start, 1), "ERROR", "lib.validation", msg)]
    lines += [
        common.airflow_log_line(_seq_ts(start, 2), "ERROR", "superset.dagfunction-metadata:mapped", "SQL failed: SELECT " + ", ".join(n for n, _ in old_columns)[:40] + " ... FROM raw." + source),
        common.airflow_log_line(_seq_ts(start, 3), "ERROR", "airflow.task", "Task failed with return code 1"),
    ]
    return {
        "schemas": {schema.table: schema},
    }, lines


def late_or_missing_data(rng: random.Random, start: datetime, source: str, target: str):
    awaited = f"s3://data-lake-staging/{source}/run_date={start.date().isoformat()}/"
    lines = [
        common.airflow_log_line(start, "INFO", "lib.s3io", f"Sensor checking for partition: {awaited}"),
        common.airflow_log_line(_seq_ts(start, 1), "INFO", "lib.s3io", f"Partition not found (check 1/30)"),
        common.airflow_log_line(_seq_ts(start, 8), "INFO", "lib.s3io", f"Partition not found (check 8/30)"),
        common.airflow_log_line(_seq_ts(start, 28), "WARN", "lib.s3io", "Timeout for S3KeySensor: poke_interval=60s, timeout=1800s"),
        common.airflow_log_line(_seq_ts(start, 29), "ERROR", "airflow.task", "Sensor timeout: waited 30 minutes for upstream partition, giving up"),
    ]
    return {}, lines


def resource_exhaustion(rng: random.Random, start: datetime, source: str, target: str):
    variant = rng.random()
    lines = []
    logs_extra = {}
    if variant < 0.4:
        lines = [
            common.airflow_log_line(start, "INFO", "lib.spark", "SparkSession started: 8 executors, 32g each"),
            common.airflow_log_line(_seq_ts(start, 3), "INFO", "lib.spark", f"Processing partition ({source}, run_date={start.date().isoformat()})"),
            common.airflow_log_line(_seq_ts(start, 10), "WARN", "lib.spark", "Memory pressure: executor 4 heap usage at 95%"),
            common.airflow_log_line(_seq_ts(start, 14), "ERROR", "lib.spark", "OutOfMemoryError: Java heap space"),
            common.airflow_log_line(_seq_ts(start, 15), "ERROR", "airflow.task", "Executor lost: container killed by YARN for exceeding memory limits"),
        ]
    elif variant < 0.7:
        lines = [
            common.airflow_log_line(start, "INFO", "lib.s3io", f"Downloading more than 20 parts for {source}"),
            common.airflow_log_line(_seq_ts(start, 4), "ERROR", "lib.s3io", "OSError: [Errno 28] No space left on device: '/tmp/staging'"),
            common.airflow_log_line(_seq_ts(start, 5), "ERROR", "airflow.task", "Task failed: local disk full"),
        ]
    else:
        lines = [
            common.airflow_log_line(start, "INFO", "lib.spark", f"Shuffle read for stage 12 ({source})"),
            common.airflow_log_line(_seq_ts(start, 6), "WARN", "lib.spark", "Executor 2 heartbeat lost"),
            common.airflow_log_line(_seq_ts(start, 7), "ERROR", "lib.spark", "ExecutorLostFailure: Container exited with exit code 137 (SIGKILL)"),
            common.airflow_log_line(_seq_ts(start, 8), "ERROR", "airflow.task", "job aborted due to stage failure"),
        ]
    return logs_extra, lines


def permission_or_auth(rng: random.Random, start: datetime, source: str, target: str):
    variant = rng.random()
    if variant < 0.5:
        msg = "AccessDeniedException: 403 User arn:aws:sts::123456789012:assumed-role/etl-reader/temp-creds is not authorized to perform: s3:GetObject on s3://prod-datalake/raw/"
        mid_lines = [
            common.airflow_log_line(_seq_ts(start, 1), "ERROR", "botocore.auth", msg),
        ]
    elif variant < 0.8:
        msg = "botocore.exceptions.ClientError: An error occurred (ExpiredToken) when calling the AssumeRole operation: The security token included in the request is expired"
        mid_lines = [
            common.airflow_log_line(_seq_ts(start, 1), "ERROR", "botocore.auth", msg),
        ]
    else:
        msg = "pyodbc.OperationalError: ('08S01', '[08S01] [unixODBC][Driver][dw] Login failed for user svc_etl_reader - Kerberos ticket expired')"
        mid_lines = [
            common.airflow_log_line(_seq_ts(start, 1), "ERROR", "pyodbc.auth", msg),
        ]
    lines = [
        common.airflow_log_line(start, "INFO", "lib.dbio", f"Connecting to source database for {source}"),
        *mid_lines,
        common.airflow_log_line(_seq_ts(start, 2), "ERROR", "airflow.task", "Could not open a connection to the data source"),
    ]
    return {}, lines


def bad_deploy(rng: random.Random, start: datetime, source: str, target: str):
    deadline = start - timedelta(hours=rng.uniform(1, 10))
    engineer = rng.choice(common.ENGINEERS)
    commit = Change(
        pipeline="",  # filled by caller
        kind=rng.choice(["code_change", "config_change"]),
        description="",
        committed_at=deadline,
        author=engineer,
        commit_sha=common.rand_hex(rng),
    )
    bug = rng.random()
    if bug < 0.4:
        desc = f"Refactor: change {source} filter from `==` to `.isin()` in transform_{source}"
        err = "TypeError: argument of type 'DataFrame' is not iterable"
    elif bug < 0.7:
        desc = f"Config: raise {source} parallelism and enable strict schema validation"
        err = "ValueError: strict_schema_validation=strict but 2 columns do not match mapping config"
    else:
        desc = f"Map fix: typo fix in {source} mapping file (similarity normalisation)"
        err = f"KeyError: 'customer_typo' not in mapping payload for {source}"
    commit.description = desc

    lines = [
        common.airflow_log_line(start, "INFO", "airflow.task", f"Running {source} DAG with latest deployment (config version {rng.randint(100, 250)})"),
        common.airflow_log_line(_seq_ts(start, 2), "ERROR", "airflow.task", err),
        common.airflow_log_line(_seq_ts(start, 3), "ERROR", "airflow.task", "Task failed; git blame indicates no recent upstream data change"),
    ]
    return {"changes": [commit]}, lines


def data_quality_violation(rng: random.Random, start: datetime, source: str, target: str):
    checks = [
        ("not_null", f"Column 'amount' contains {rng.randint(2, 50)} null values; expect 0"),
        ("range", f"amount = {rng.uniform(-500, -0.1):.2f} is < min=0 (out of range values found)"),
        ("unique", f"Duplicate key detected for column 'order_id', found {rng.randint(2, 30)} duplicate values"),
        ("freshness", f"Table raw.{source} not refreshed since Monday (expect daily refresh 5h ago)"),
        ("volume", f"Row count anomaly: {rng.randint(100, 500)} rows vs expected {rng.uniform(15000, 50000):.0f}"),
        ("enum", f"found unknown value in column 'status': 'PROCESSED' not in ['pending', 'completed', 'failed']"),
    ]
    selected = rng.sample(checks, k=1)[0]
    check_name, msg = selected
    lines = [
        common.airflow_log_line(start, "INFO", "dq_engine", f"Running {check_name} check for table raw.{source}"),
        common.airflow_log_line(_seq_ts(start, 1), "ERROR", "dq_engine", f"QUALITY CHECK FAILED: [strict] {check_name} -> {msg}" if check_name != "freshness" else f"QUALITY CHECK FAILED: freshness -> {msg}"),
        common.airflow_log_line(_seq_ts(start, 2), "ERROR", "airflow.task", f"DQ gate failed: check '{check_name}' for raw.{source} rejected the batch, downstream loads halted"),
    ]
    return {}, lines


def upstream_dependency_failure(rng: random.Random, start: datetime, source: str, target: str):
    parent = f"extract_{source}" if target is None else f"extract_{source}"
    upstream_source = rng.choice([s for s in common.SOURCES if s != source])
    upstream_task = f"extract_{upstream_source}"
    lines = [
        common.airflow_log_line(start, "INFO", "airflow.task", f"Triggering upstream dependencies for {source} pipeline"),
        common.airflow_log_line(_seq_ts(start, 1), "WARN", "airflow.task", f"Upstream task '{upstream_task}' (upstream DAG) produced no output"),
        common.airflow_log_line(_seq_ts(start, 2), "ERROR", "airflow.task", f"Upstream task 'ingest_{upstream_source}' failed; see logs for run_id={common.rand_hex(rng, 12)}"),
        common.airflow_log_line(_seq_ts(start, 3), "ERROR", "airflow.task", f"Downstream task transform_{source} skipped or failed due to missing dependency output"),
        common.airflow_log_line(_seq_ts(start, 4), "ERROR", "airflow.task", "Task failed with error: UpstreamTaskFailure"),
    ]
    return {}, lines


def transient_infra(rng: random.Random, start: datetime, source: str, target: str):
    variant = rng.random()
    if variant < 0.4:
        mid = [
            common.airflow_log_line(_seq_ts(start, 1), "WARN", "http.client", f"ConnectionResetError: [Errno 54] Connection reset by peer while calling {source} API"),
            common.airflow_log_line(_seq_ts(start, 2), "INFO", "http.client", "Reconnect attempt 3/retro succeeded"),
        ]
        tail = [("INFO", "Task succeeded after retry (3rd attempt)")]
    elif variant < 0.7:
        mid = [
            common.airflow_log_line(_seq_ts(start, 1), "WARN", "botocore.readtimeout", f"Read timeout contacting S3 (region us-east-1): Retrying, attempt {rng.randint(2,5)}/5"),
            common.airflow_log_line(_seq_ts(start, 2), "INFO", "botocore.readtimeout", "Backoff 2.4s before next retry"),
        ]
        tail = [("INFO", "Task succeeded after retry (2nd attempt)")]
    else:
        mid = [
            common.airflow_log_line(_seq_ts(start, 1), "WARN", "sqlalchemy.operationalerror", "(psycopg2.OperationalError) server closed the connection unexpectedly"),
            common.airflow_log_line(_seq_ts(start, 2), "INFO", "lib.dbio", "Reconnect + re-issued query succeeded"),
        ]
        tail = [("INFO", "Task succeeded after retry (4th attempt)")]
    lines = [common.airflow_log_line(start, "INFO", "lib.dbio", f"Reading source {source}")]
    lines += mid
    lines += [common.airflow_log_line(_seq_ts(start, 3), lvl, "airflow.task", m) for lvl, m in tail]
    return {}, lines


def no_clear_signal(rng: random.Random, start: datetime, source: str, target: str):
    return {}, [
        common.airflow_log_line(start, "INFO", "airflow.task", f"Task transform_{source} started, 3 attempts allowed"),
        common.airflow_log_line(_seq_ts(start, 2), "WARN", "airflow.task", "Attempt 1/3 exited with return code 1"),
        common.airflow_log_line(_seq_ts(start, 4), "WARN", "airflow.task", "Attempt 2/3 exited with return code 1"),
        common.airflow_log_line(_seq_ts(start, 6), "ERROR", "airflow.task", "All 3 attempts failed, marking task as FAILED"),
    ]


BUILDERS = {
    RootCause.SCHEMA_DRIFT: schema_drift,
    RootCause.LATE_OR_MISSING_DATA: late_or_missing_data,
    RootCause.RESOURCE_EXHAUSTION: resource_exhaustion,
    RootCause.PERMISSION_OR_AUTH: permission_or_auth,
    RootCause.BAD_DEPLOY: bad_deploy,
    RootCause.DATA_QUALITY_VIOLATION: data_quality_violation,
    RootCause.UPSTREAM_DEPENDENCY_FAILURE: upstream_dependency_failure,
    RootCause.TRANSIENT_INFRA: transient_infra,
}
