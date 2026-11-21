# Pipeline Failure Triage Agent (MCP)

When a data pipeline fails, an on-call engineer has to read noisy task logs, check
recent deploys, compare table schemas and look at run history before they can even
say *why* it broke, and that first diagnosis is slow, manual and repetitive. This
project is an LLM agent that does that investigation through read-only tools,
classifies the likely root cause, cites verbatim evidence, and proposes a next step
for a human to approve.

> **Synthetic data only. The agent is advisory and read-only:** every tool is
> read-only, the agent never executes a fix, and every diagnosis carries
> `requires_human_approval: true`.

## Architecture

```mermaid
flowchart LR
    subgraph Synthetic environment
        G[Failure generator<br/>seeded, UTC] --> R[(Failed runs:<br/>logs, schemas,<br/>changes, history,<br/>ground truth)]
    end

    subgraph MCP server - read-only, scoped by run_id
        T1[get_run_logs]
        T2[get_recent_changes]
        T3[get_table_schema]
        T4[get_run_history]
    end

    R --> T1 & T2 & T3 & T4

    subgraph Agent
        I[Incident: run_id, pipeline,<br/>failed task, start] --> LOOP[Tool-calling loop<br/>Mistral]
        LOOP <-- MCP client, stdio --> T1 & T2 & T3 & T4
        LOOP --> V{Validate:<br/>schema + verbatim<br/>evidence}
        V -- invalid --> LOOP
        V -- valid --> OUT[Diagnosis - Pydantic]
    end

    OUT --> H[Human review / approval]
    OUT --> EV[Evaluation harness]
    R -- labels --> EV
    R --> B[Rule-based baseline] --> EV
```

The agent reaches its tools **only** through an MCP client session; the function
specs it shows the model come from the server's `list_tools`, so the server is the
single source of truth. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design
decisions (incident-scoped tools, guardrails, the generator, evaluation).

### Failure taxonomy

| Label | Example signal |
|---|---|
| `schema_drift` | Column missing/renamed/type changed upstream; schema version bump just before the run |
| `late_or_missing_data` | Sensor timeout, partition not arrived |
| `resource_exhaustion` | OOM, disk full, executor lost (exit 137) |
| `permission_or_auth` | 403, expired token, login failure |
| `bad_deploy` | A meaningful code/config change shortly before the first failure |
| `data_quality_violation` | Null/range/duplicate/freshness/volume check failed |
| `upstream_dependency_failure` | An upstream task/DAG failed; this task is a symptom |
| `transient_infra` | Network reset/timeout where a retry succeeded |
| `unknown` | Not enough evidence (the agent is allowed, and encouraged, to say so) |

### Output

```json
{
  "run_id": "run_00042",
  "root_cause": "schema_drift",
  "confidence": 0.9,
  "evidence": [
    {"source": "run_logs", "quote": "AnalysisException: Column 'amount' not found in source table raw.orders.", "note": "missing column"},
    {"source": "table_schema", "quote": "column 'amount' no longer provided upstream", "note": "schema v2, 3h before the run"}
  ],
  "proposed_next_step": "Update the mapping for the removed column and re-run from the failing task.",
  "requires_human_approval": true
}
```

Every quote must appear verbatim in the output of the tool named in `source`;
otherwise the answer is rejected and the model gets a bounded number of repair turns
before the agent falls back to `unknown`.

## Results

Dataset: 120 synthetic failed runs (seed 42), balanced across classes: 12 `unknown`
(10%), 18 genuinely ambiguous (15%: both causes' evidence present, either accepted),
and noisy logs, with 42 runs (35%) containing a benign line that mentions another
category.

| Variant | Runs | Accuracy | Macro-F1 | Tool calls / incident | Tokens / incident (in / out) | Latency / incident |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based baseline | 120 | 0.875 | 0.871 | n/a (direct calls) | 0 | < 1 ms |
| Rule-based baseline | 27 (subset) | 0.889 | 0.875 | n/a (direct calls) | 0 | < 1 ms |
| Agent (`mistral-large-latest`, MCP tools) | 27 (subset) | 0.852 | 0.848 | 5.6 | 13.3k / 2.6k | 51 s |

The subset is class-stratified (3 incidents per class, `--limit 27`) because every
agent incident costs LLM calls. Reports: [`eval/reports/baseline.json`](eval/reports/baseline.json),
[`eval/reports/agent-subset27.json`](eval/reports/agent-subset27.json) (per-run
predictions included).

**Honest reading:**

- The agent does **not** beat the keyword baseline yet; the difference is one
  incident out of 27, within noise. 3.7% of agent incidents fell back to `unknown`
  after failing validation.
- The baseline was written against the generator's vocabulary, so it is strong on
  synthetic logs. Its misses are the ones a keyword matcher should make: benign
  lines such as an optional endpoint returning 403, or a cache "Partition not found".
- The agent's misses: generic retries (`Attempt 1/3 exited with return code 1`) read
  as `transient_infra` when the truth is `unknown` (2), and a failed freshness check
  read as late data (1). This confusion matrix is the input to the next prompt
  iteration (milestone 5).

Reproduce:

```bash
.venv/bin/triage-eval --report eval/reports/baseline.json
MISTRAL_API_KEY=... .venv/bin/triage-eval --predictor agent --limit 27 --report eval/reports/agent-subset27.json
```

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"

.venv/bin/pytest                      # 46 offline tests (+2 live tests, skipped without a key)
```

Triage incidents with the agent (needs a Mistral key; the MCP server is spawned over
stdio on the same seeded dataset):

```bash
export MISTRAL_API_KEY=...
.venv/bin/triage-agent --sample 3 --dump-trace      # or: --run-id run_00042
```

Run the MCP server on its own, e.g. to connect another MCP client:

```bash
.venv/bin/triage-mcp --n 120 --seed 42              # or: --dataset runs.jsonl
```

Save a dataset for reproducible evals: `triage-eval --save-dataset data/runs.jsonl`,
then pass `--dataset data/runs.jsonl` to any command.

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12+ |
| LLM | Mistral (`mistral-large-latest`) via the official `mistralai` SDK, with retries |
| Tool protocol | MCP, official Python SDK (`mcp` 2.x); stdio transport, in-memory for tests |
| Agent | plain tool-calling loop (no framework) |
| Validation | Pydantic |
| Evaluation | custom harness: accuracy, macro-F1, per-class P/R, confusion matrix, calibration, cost/latency |
| Testing | pytest; LLM mocked by a scripted fake, tools served over in-memory MCP |

## Repository layout

```
pipeline_triage/
  generator/        synthetic failure generator (builders per cause, noise, ambiguity)
  mcp_server.py     the four read-only tools + MCP server (single source of tool specs)
  mcp_server_main.py  stdio entry point (triage-mcp)
  baseline/         rule-based keyword/regex classifier
  eval/             harness + run_eval (triage-eval)
  env.py, models.py dataset container and shared models
agent/
  loop.py           tool-calling loop, guardrails, repair, fallback, tracing
  tools.py          MCP client (sync facade over an async session)
  llm.py            Mistral client wrapper (LLMClient protocol)
  schemas.py        Incident and Diagnosis models
  prompts.py        system prompt
  run.py            CLI (triage-agent)
eval/reports/       committed evaluation reports
tests/              unit, protocol, stdio end-to-end and (opt-in) live tests
```

## Limitations and next steps

- **Synthetic logs are cleaner than real ones.** Phrasing is limited, signals are
  explicit, and the baseline knows the vocabulary. A realism check on public logs
  (e.g. Loghub) is a stretch goal.
- **Small agent eval.** 27 incidents; the full 120 costs roughly 120 × 16k tokens.
- **Not done yet:** a "single prompt, logs only" LLM variant for comparison, prompt
  iteration from the confusion matrix (milestone 5), Docker images and CI
  (milestone 7), cost in dollars (only tokens are reported), and the optional
  `propose_action` approval flow.
