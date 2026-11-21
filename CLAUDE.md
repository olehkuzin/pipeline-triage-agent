# CLAUDE.md

Guidance for Claude Code when working in this repository. Architecture and design
rationale: [ARCHITECTURE.md](ARCHITECTURE.md). Usage and results: [README.md](README.md).

## Commands

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -e ".[dev]"   # setup
.venv/bin/pytest                                    # offline suite (~1s); live tests skip without a key
.venv/bin/pytest tests/test_mcp_server.py -k stdio  # stdio end-to-end against the real server process
.venv/bin/triage-eval --report eval/reports/baseline.json    # regenerate the baseline report
```

Live and agent runs call the Mistral API and **cost money**. Only run them when asked:

```bash
MISTRAL_API_KEY=... .venv/bin/pytest tests/test_agent_live.py -s        # ~5 min
MISTRAL_API_KEY=... .venv/bin/triage-eval --predictor agent --limit 27  # ~25 min, ~430k tokens
```

There is no `.env` loading in this project; `MISTRAL_API_KEY` must be in the
environment. Never commit keys.

## Invariants: keep these true

- **Tools are read-only and scoped by `run_id`.** Every tool answers from the
  incident's own snapshot (`run.schemas`, `run.changes`, `run.run_history`), anchored
  at `run.started_at`. Never query across incidents by pipeline or table.
- **`create_server()` is the single source of tool specs.** The agent builds its
  function specs from `list_tools`. Don't hand-write specs in the agent.
- **The agent reaches tools only through `MCPToolClient`.** In tests, use
  `MCPToolClient.in_memory(env)`, not direct function calls. The baseline may call
  the functions directly.
- **User-facing tool errors raise `pipeline_triage.mcp_server.ToolError`** (an SDK
  `ToolError` subclass). Other exceptions are hidden from the model as generic crashes.
- **Every Mistral tool call needs a matching `tool` message**, including calls
  refused by the budget.
- **Generator determinism:** timestamps are UTC, and all randomness comes from the
  seeded `random.Random`. `test_dataset_is_independent_of_local_timezone` guards this.
- **No label leakage.** No generated log line should name or hint at the root cause,
  and the system prompt must not describe the generator's specific red herrings.

## When you change things

- Changing the generator, tools or baseline changes the dataset or the scores. Rerun
  `triage-eval --report eval/reports/baseline.json` and update the README results. The
  agent report (`eval/reports/agent-subset27.json`) goes stale too; say so rather
  than re-running it unasked.
- Agent tests use the scripted `FakeLLM` in `tests/test_agent.py`, and its `chat`
  signature must match `LLMClient.chat`.
- `pytest` works from any directory (`pythonpath = ["."]`). Target Python is 3.12+.
