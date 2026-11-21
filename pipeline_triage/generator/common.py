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

NOISE_LINES = [
    "WARN  Retrying HTTP request to metadata service (attempt 1/3)",
    "INFO  Graph visualization too large; 12 nodes omitted",
    "WARN  DeprecationWarning: 'datetime.utcnow' is deprecated, use 'datetime.now'",
    "INFO  Task instance heartbeat recorded",
    "WARN  Scheduler heartbeat took 2.3s (threshold 2.0s)",
    "INFO  Using execution timezone: UTC",
    "DEBUG connection pool size: 10",
    "INFO  Skipping SLA check: no SLA configured for task",
]

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
    t = rng.choice(TASK_TEMPLATES)
    t = t.replace("{source}", source).replace("{target}", target or rng.choice(TARGETS))
    head, *rest = t.split("_", 1)
    return head + ("_" + rest[0] if rest else "")


def airflow_log_line(ts: datetime, level: str, logger: str, msg: str) -> str:
    """Format an Airflow-style log line.

    The timestamps inside the message body are what the agent will actually
    read, so we keep the format realistic: [ts] LEVEL [logger] message.
    """
    return f"[{ts.isoformat()}] {level:<5} [{logger}] {msg}"


def add_noise(rng: random.Random, lines: list[str], n: int, start: datetime) -> None:
    """Sprinkle irrelevant log lines in-place around the existing ones."""
    for _ in range(n):
        msg = rng.choice(NOISE_LINES)
        level = msg.split()[0]
        logger = rng.choice(["airflow.task", "airflow.scheduler", "lib.dbio", "lib.s3io"])
        pos = rng.randint(0, len(lines))
        lines.insert(pos, airflow_log_line(start, level, logger, msg))
