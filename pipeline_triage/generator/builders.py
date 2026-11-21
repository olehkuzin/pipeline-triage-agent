"""Per-cause failure builders.

Each builder takes `(rng, start, source, target, ambiguous)` where `start` is
the failing task's start time, and returns `(context_dict, signal_lines)`:
- context_dict may carry `schemas` and `changes` for the run's snapshot, and
- signal_lines are the failing task's log lines, timestamped from `start`.

`ambiguous=True` (only meaningful for causes in `AMBIGUOUS_PAIRS`) mixes in the
paired cause's signal, so two root causes are genuinely plausible.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from ..models import Change, ColumnSpec, RootCause, TableSchema
from . import common

# cause -> the alternative cause an ambiguous incident of that cause also supports
AMBIGUOUS_PAIRS: dict[RootCause, RootCause] = {
    RootCause.SCHEMA_DRIFT: RootCause.BAD_DEPLOY,
    RootCause.BAD_DEPLOY: RootCause.SCHEMA_DRIFT,
    RootCause.TRANSIENT_INFRA: RootCause.UPSTREAM_DEPENDENCY_FAILURE,
    RootCause.UPSTREAM_DEPENDENCY_FAILURE: RootCause.TRANSIENT_INFRA,
}


def _seq_ts(start: datetime, i: int, step_s: int = 45) -> datetime:
    return start + timedelta(seconds=step_s * i)


def _log(start: datetime, i: int, level: str, logger: str, msg: str) -> str:
    return common.airflow_log_line(_seq_ts(start, i), level, logger, msg)


def _mk_source(
    source: str, version: int, columns: list[tuple[str, str]], changed_at: datetime | None, desc: str | None
) -> TableSchema:
    return TableSchema(
        table=f"raw.{source}",
        columns=[ColumnSpec(name=n, dtype=d) for n, d in columns],
        version=version,
        changed_at=changed_at,
        change_description=desc,
    )


def _commit(rng: random.Random, start: datetime, description: str, max_hours: float = 10) -> Change:
    return Change(
        pipeline="",  # filled by caller
        kind=rng.choice(["code_change", "config_change"]),
        description=description,
        committed_at=start - timedelta(hours=rng.uniform(1, max_hours)),
        author=rng.choice(common.ENGINEERS),
        commit_sha=common.rand_hex(rng),
    )


def _type_change(rng: random.Random, source: str) -> tuple[str, str, str, list[tuple[str, str]]]:
    """Pick a column and change its dtype. Returns (column, old_dtype, new_dtype, new_columns)."""
    columns = common.base_schema(source)
    name, old = rng.choice(columns)
    new = "string" if old != "string" else "int64"
    return name, old, new, [(n, new if n == name else d) for n, d in columns]


def schema_drift(rng: random.Random, start: datetime, source: str, target: str, ambiguous: bool = False):
    """Upstream column renamed/removed/type-changed shortly before the run."""
    old_columns = common.base_schema(source)
    variant = rng.random()
    if variant < 0.5:
        victim = rng.choice([c for c in old_columns if not c[0].endswith("_id")])[0]
        renamed = victim + "_v2"
        new_columns = [(renamed if n == victim else n, d) for n, d in old_columns]
        drift_desc = f"column '{victim}' renamed to '{renamed}' upstream"
        msg = f"AnalysisException: Column '{victim}' not found in source table raw.{source}. Did you mean '{renamed}'?"
    elif variant < 0.8:
        victim = rng.choice([c for c in old_columns if not c[0].endswith("_id")])[0]
        new_columns = [c for c in old_columns if c[0] != victim]
        drift_desc = f"column '{victim}' no longer provided upstream"
        msg = f"AnalysisException: Column '{victim}' not found in source table raw.{source}."
    else:
        victim, old_dtype, dtype, new_columns = _type_change(rng, source)
        drift_desc = f"column '{victim}' dtype changed from {old_dtype} to {dtype}"
        msg = f"InvalidInput: Cannot cast column '{victim}' to expected type; source now returns {dtype} values"

    schema = _mk_source(source, 2, new_columns, start - timedelta(hours=rng.uniform(1, 5)), drift_desc)
    lines = [
        _log(start, 0, "INFO", "lib.loader", f"Reading partition run_date={start.date().isoformat()} from raw.{source}"),
        _log(start, 1, "ERROR", "lib.validation", msg),
        _log(start, 2, "ERROR", "lib.sql", "SQL failed: SELECT " + ", ".join(n for n, _ in old_columns)[:40] + " ... FROM raw." + source),
        _log(start, 3, "ERROR", "airflow.task", "Task failed with return code 1"),
    ]
    context: dict[str, object] = {"schemas": {schema.table: schema}}
    if ambiguous:
        # a meaningful deploy touching the same model landed around the drift
        context["changes"] = [_commit(rng, start, f"Refactor: switch {source} staging model to an explicit column list", 8)]
    return context, lines


def late_or_missing_data(rng: random.Random, start: datetime, source: str, target: str, ambiguous: bool = False):
    awaited = f"s3://data-lake-staging/{source}/run_date={start.date().isoformat()}/"
    lines = [
        _log(start, 0, "INFO", "lib.s3io", f"Sensor checking for partition: {awaited}"),
        _log(start, 1, "INFO", "lib.s3io", "Partition not found (check 1/30)"),
        _log(start, 8, "INFO", "lib.s3io", "Partition not found (check 8/30)"),
        _log(start, 28, "WARN", "lib.s3io", "Timeout for S3KeySensor: poke_interval=60s, timeout=1800s"),
        _log(start, 29, "ERROR", "airflow.task", "Sensor timeout: waited 30 minutes for upstream partition, giving up"),
    ]
    return {}, lines


def resource_exhaustion(rng: random.Random, start: datetime, source: str, target: str, ambiguous: bool = False):
    variant = rng.random()
    if variant < 0.4:
        lines = [
            _log(start, 0, "INFO", "lib.spark", "SparkSession started: 8 executors, 32g each"),
            _log(start, 3, "INFO", "lib.spark", f"Processing partition ({source}, run_date={start.date().isoformat()})"),
            _log(start, 10, "WARN", "lib.spark", "Memory pressure: executor 4 heap usage at 95%"),
            _log(start, 14, "ERROR", "lib.spark", "OutOfMemoryError: Java heap space"),
            _log(start, 15, "ERROR", "airflow.task", "Executor lost: container killed by YARN for exceeding memory limits"),
        ]
    elif variant < 0.7:
        lines = [
            _log(start, 0, "INFO", "lib.s3io", f"Downloading more than 20 parts for {source}"),
            _log(start, 4, "ERROR", "lib.s3io", "OSError: [Errno 28] No space left on device: '/tmp/staging'"),
            _log(start, 5, "ERROR", "airflow.task", "Task failed: local disk full"),
        ]
    else:
        lines = [
            _log(start, 0, "INFO", "lib.spark", f"Shuffle read for stage 12 ({source})"),
            _log(start, 6, "WARN", "lib.spark", "Executor 2 heartbeat lost"),
            _log(start, 7, "ERROR", "lib.spark", "ExecutorLostFailure: Container exited with exit code 137 (SIGKILL)"),
            _log(start, 8, "ERROR", "airflow.task", "job aborted due to stage failure"),
        ]
    return {}, lines


def permission_or_auth(rng: random.Random, start: datetime, source: str, target: str, ambiguous: bool = False):
    variant = rng.random()
    if variant < 0.5:
        logger = "botocore.auth"
        msg = (
            "AccessDeniedException: 403 User arn:aws:sts::123456789012:assumed-role/etl-reader/temp-creds "
            "is not authorized to perform: s3:GetObject on s3://prod-datalake/raw/"
        )
    elif variant < 0.8:
        logger = "botocore.auth"
        msg = (
            "botocore.exceptions.ClientError: An error occurred (ExpiredToken) when calling the AssumeRole "
            "operation: The security token included in the request is expired"
        )
    else:
        logger = "pyodbc.auth"
        msg = (
            "pyodbc.OperationalError: ('08S01', '[08S01] [unixODBC][Driver][dw] Login failed for user "
            "svc_etl_reader - Kerberos ticket expired')"
        )
    lines = [
        _log(start, 0, "INFO", "lib.dbio", f"Connecting to source database for {source}"),
        _log(start, 1, "ERROR", logger, msg),
        _log(start, 2, "ERROR", "airflow.task", "Could not open a connection to the data source"),
    ]
    return {}, lines


def bad_deploy(rng: random.Random, start: datetime, source: str, target: str, ambiguous: bool = False):
    """A code/config change shortly before the run broke the task."""
    context: dict[str, object] = {}
    if ambiguous:
        # the deploy coincides with an upstream type change, and the error is a cast error
        victim, old_dtype, dtype, new_columns = _type_change(rng, source)
        desc = f"Refactor: cast {victim} explicitly in transform_{source}"
        err = f"InvalidInput: Cannot cast column '{victim}' to expected type; source now returns {dtype} values"
        schema = _mk_source(
            source,
            2,
            new_columns,
            start - timedelta(hours=rng.uniform(1, 12)),
            f"column '{victim}' dtype changed from {old_dtype} to {dtype}",
        )
        context["schemas"] = {schema.table: schema}
    else:
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
    context["changes"] = [_commit(rng, start, desc)]

    lines = [
        _log(start, 0, "INFO", "airflow.task", f"Running {source} job (config version {rng.randint(100, 250)})"),
        _log(start, 2, "ERROR", "airflow.task", err),
        _log(start, 3, "ERROR", "airflow.task", "Task failed with return code 1"),
    ]
    return context, lines


def data_quality_violation(rng: random.Random, start: datetime, source: str, target: str, ambiguous: bool = False):
    checks = [
        ("not_null", f"Column 'amount' contains {rng.randint(2, 50)} null values; expect 0"),
        ("range", f"amount = {rng.uniform(-500, -0.1):.2f} is < min=0 (out of range values found)"),
        ("unique", f"Duplicate key detected for column 'order_id', found {rng.randint(2, 30)} duplicate values"),
        ("freshness", f"Table raw.{source} not refreshed since Monday (expect daily refresh 5h ago)"),
        ("volume", f"Row count anomaly: {rng.randint(100, 500)} rows vs expected {rng.uniform(15000, 50000):.0f}"),
        ("enum", "found unknown value in column 'status': 'PROCESSED' not in ['pending', 'completed', 'failed']"),
    ]
    check_name, msg = rng.choice(checks)
    failed = (
        f"QUALITY CHECK FAILED: freshness -> {msg}"
        if check_name == "freshness"
        else f"QUALITY CHECK FAILED: [strict] {check_name} -> {msg}"
    )
    lines = [
        _log(start, 0, "INFO", "dq_engine", f"Running {check_name} check for table raw.{source}"),
        _log(start, 1, "ERROR", "dq_engine", failed),
        _log(
            start, 2, "ERROR", "airflow.task",
            f"DQ gate failed: check '{check_name}' for raw.{source} rejected the batch, downstream loads halted",
        ),
    ]
    return {}, lines


def upstream_dependency_failure(rng: random.Random, start: datetime, source: str, target: str, ambiguous: bool = False):
    upstream_source = rng.choice([s for s in common.SOURCES if s != source])
    upstream_task = f"extract_{upstream_source}"
    lines = [_log(start, 0, "INFO", "airflow.task", f"Triggering upstream dependencies for {source} pipeline")]
    if ambiguous:
        # the upstream failure may itself have been a network blip
        lines.append(
            _log(start, 1, "WARN", "http.client",
                 "ConnectionResetError: [Errno 54] Connection reset by peer while fetching upstream output (attempt 1/3)")
        )
    lines += [
        _log(start, 2, "WARN", "airflow.task", f"Upstream task '{upstream_task}' (upstream DAG) produced no output"),
        _log(
            start, 3, "ERROR", "airflow.task",
            f"Upstream task 'ingest_{upstream_source}' failed; see logs for run_id={common.rand_hex(rng, 12)}",
        ),
        _log(start, 4, "ERROR", "airflow.task", "This task cannot run: missing dependency output from upstream"),
        _log(start, 5, "ERROR", "airflow.task", "Task failed with error: UpstreamTaskFailure"),
    ]
    return {}, lines


def transient_infra(rng: random.Random, start: datetime, source: str, target: str, ambiguous: bool = False):
    variant = rng.random()
    if variant < 0.4:
        mid = [
            ("WARN", "http.client", f"ConnectionResetError: [Errno 54] Connection reset by peer while calling {source} API"),
            ("INFO", "http.client", "Reconnect attempt 3/3 succeeded"),
        ]
        tail = "Task succeeded after retry (3rd attempt)"
    elif variant < 0.7:
        mid = [
            ("WARN", "botocore.readtimeout",
             f"Read timeout contacting S3 (region us-east-1): Retrying, attempt {rng.randint(2, 5)}/5"),
            ("INFO", "botocore.readtimeout", "Backoff 2.4s before next retry"),
        ]
        tail = "Task succeeded after retry (2nd attempt)"
    else:
        mid = [
            ("WARN", "sqlalchemy.operationalerror", "(psycopg2.OperationalError) server closed the connection unexpectedly"),
            ("INFO", "lib.dbio", "Reconnect + re-issued query succeeded"),
        ]
        tail = "Task succeeded after retry (4th attempt)"
    if ambiguous:
        # an upstream DAG was also flaky: the delay could be upstream, not the network
        other = rng.choice([s for s in common.SOURCES if s != source])
        mid.insert(0, ("WARN", "airflow.task",
                       f"Upstream task 'extract_{other}' (upstream DAG) was retried 2 times before producing output"))

    lines = [_log(start, 0, "INFO", "lib.dbio", f"Reading source {source}")]
    lines += [_log(start, i, level, logger, msg) for i, (level, logger, msg) in enumerate(mid, start=1)]
    lines.append(_log(start, len(mid) + 1, "INFO", "airflow.task", tail))
    return {}, lines


def no_clear_signal(rng: random.Random, start: datetime, source: str, target: str, ambiguous: bool = False):
    return {}, [
        _log(start, 0, "INFO", "airflow.task", "Retry policy: 3 attempts allowed"),
        _log(start, 2, "WARN", "airflow.task", "Attempt 1/3 exited with return code 1"),
        _log(start, 4, "WARN", "airflow.task", "Attempt 2/3 exited with return code 1"),
        _log(start, 6, "ERROR", "airflow.task", "All 3 attempts failed, marking task as FAILED"),
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
