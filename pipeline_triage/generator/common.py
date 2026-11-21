"""Common building blocks for generating realistic synthetic pipeline artifacts."""

from __future__ import annotations

import random
from datetime import datetime

SOURCES = [
    "customers",
    "orders",
    "payments",
    "events",
    "inventory",
    "shipments",
    "subscriptions",
    "refunds",
]

TARGETS = ["warehouse", "lake", "analytics", "reporting"]

PIPELINES = [
    "daily_sales_etl",
    "customer_360_ingest",
    "payments_reconciliation",
    "events_enrichment",
    "inventory_sync",
]

TASK_TEMPLATES = [
    "extract_{source}",
    "validate_{source}",
    "transform_{source}",
    "load_{source}_to_{target}",
    "publish_{source}_metrics",
]

BASE_SCHEMA_BY_SOURCE = {
    "customers": [("customer_id", "int64"), ("email", "string"), ("country", "string"), ("signup_date", "timestamp")],
    "orders": [("order_id", "int64"), ("customer_id", "int64"), ("amount", "float64"), ("order_ts", "timestamp")],
    "payments": [("payment_id", "int64"), ("order_id", "int64"), ("status", "string"), ("paid_at", "timestamp")],
    "events": [("event_id", "string"), ("user_id", "int64"), ("event_type", "string"), ("event_ts", "timestamp")],
    "inventory": [("sku", "string"), ("warehouse_id", "int64"), ("quantity", "int64"), ("updated_at", "timestamp")],
    "shipments": [("shipment_id", "int64"), ("order_id", "int64"), ("carrier", "string"), ("shipped_at", "timestamp")],
    "subscriptions": [("sub_id", "int64"), ("customer_id", "int64"), ("plan", "string"), ("renewed_at", "timestamp")],
    "refunds": [("refund_id", "int64"), ("order_id", "int64"), ("reason", "string"), ("refunded_at", "timestamp")],
}

ENGINEERS = ["akira", "mchen", "jsilva", "dpetrov", "lfontaine", "twong"]

# (level, logger, message): irrelevant lines sprinkled into every failing task.
NOISE_LINES = [
    ("WARN", "lib.http", "Retrying HTTP request to metadata service (attempt 1/3)"),
    ("INFO", "airflow.task", "Graph visualization too large; 12 nodes omitted"),
    ("WARN", "py.warnings", "DeprecationWarning: 'datetime.utcnow' is deprecated, use 'datetime.now'"),
    ("INFO", "airflow.task", "Task instance heartbeat recorded"),
    ("WARN", "airflow.scheduler", "Scheduler heartbeat took 2.3s (threshold 2.0s)"),
    ("INFO", "airflow.task", "Using execution timezone: UTC"),
    ("DEBUG", "lib.dbio", "connection pool size: 10"),
    ("INFO", "airflow.task", "Skipping SLA check: no SLA configured for task"),
]

# Benign lines that happen to contain another failure category's vocabulary, as
# real logs do. A careful reader sees they are harmless; a keyword matcher does
# not. Injected into a minority of runs, never as the actual cause.
RED_HERRING_LINES = [
    ("WARN", "lib.metrics", "Optional metrics push to https://metrics.internal/v1 returned 403; continuing without metrics"),
    ("INFO", "lib.cache", "Partition not found in local cache, fetching from object store"),
    ("INFO", "lib.http", "Request to schema registry succeeded after retry (2nd attempt)"),
    ("WARN", "lib.spark", "Memory pressure: executor 3 heap usage at 81%, spilling to disk"),
    ("WARN", "dq_engine", "Soft check 'row_count_drift' warned (non-blocking): 3.1% below 7-day average"),
]
RED_HERRING_RATE = 0.35

STACK_TRACES = {
    "python": [
        "Traceback (most recent call last):",
        '  File "/opt/airflow/dags/lib/loader.py", line 88, in run',
        "    df = self.load(source_path)",
        '  File "/opt/airflow/dags/lib/loader.py", line 141, in load',
        "    return self._read_parquet(source_path)",
    ],
    "pyspark": [
        "Traceback (most recent call last):",
        '  File "/opt/jobs/spark_job.py", line 212, in main',
        "    df = spark.read.table(src).where(col(\"ingest_date\") == run_date)",
        "py4j.protocol.Py4JJavaError: An error occurred while calling o123.load.",
    ],
}


def base_schema(source: str) -> list[tuple[str, str]]:
    return list(BASE_SCHEMA_BY_SOURCE[source])


def rand_hex(rng: random.Random, n: int = 40) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def make_task_name(rng: random.Random, source: str, target: str | None = None) -> str:
    template = rng.choice(TASK_TEMPLATES)
    return template.replace("{source}", source).replace("{target}", target or rng.choice(TARGETS))


def airflow_log_line(ts: datetime, level: str, logger: str, msg: str) -> str:
    """Format an Airflow-style log line: [ts] LEVEL [logger] message."""
    return f"[{ts.isoformat(timespec='seconds')}] {level:<5} [{logger}] {msg}"


def log_line_ts(line: str) -> datetime:
    """Timestamp of a line produced by `airflow_log_line`."""
    return datetime.fromisoformat(line[1 : line.index("]")])


def add_noise(rng: random.Random, lines: list[str], n: int, red_herring: bool = False) -> None:
    """Insert irrelevant lines in-place, after the first line, keeping timestamps in order.

    Each inserted line takes the timestamp of the line it follows, so the log
    stays chronological.
    """
    picks = [rng.choice(NOISE_LINES) for _ in range(n)]
    if red_herring:
        picks.append(rng.choice(RED_HERRING_LINES))
    for level, logger, msg in picks:
        pos = rng.randint(1, len(lines))
        lines.insert(pos, airflow_log_line(log_line_ts(lines[pos - 1]), level, logger, msg))
