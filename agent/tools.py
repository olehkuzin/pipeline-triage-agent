"""MCP client used by the agent to reach the read-only triage tools.

The agent never calls tool implementations directly: it lists tools and calls
them through an MCP `ClientSession`, either over stdio against the real server
process (`python -m pipeline_triage.mcp_server_main`) or over the SDK's
in-memory transport (tests). Tool names, descriptions and parameter schemas
come from the server's `list_tools` response, so there is one source of truth.

The MCP SDK is async and its transports use anyio cancel scopes that must be
entered and exited by the same task. The session therefore lives in a single
long-running task on a private event loop in a background thread; the
synchronous agent loop submits calls to it.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client._memory import InMemoryTransport
from mcp.client.stdio import get_default_environment, stdio_client

from pipeline_triage.env import Environment
from pipeline_triage.mcp_server import create_server

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALL_TIMEOUT_S = 60.0
START_TIMEOUT_S = 30.0

Connect = Callable[[], AbstractAsyncContextManager[Any]]


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool


def _strip_titles(schema: Any) -> Any:
    """Drop pydantic-generated `title` keys; they add tokens and no information."""
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


class MCPToolClient:
    """Synchronous facade over an MCP client session. Use as a context manager."""

    def __init__(self, connect: Connect, label: str = "mcp") -> None:
        self._connect = connect
        self._label = label
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._tools: list[Any] = []

    # ---------- constructors ----------

    @classmethod
    def stdio(cls, dataset: str | Path | None = None, n: int = 120, seed: int = 42) -> MCPToolClient:
        """Spawn the real MCP server as a subprocess and talk to it over stdio."""
        args = ["-m", "pipeline_triage.mcp_server_main"]
        args += ["--dataset", str(dataset)] if dataset else ["--n", str(n), "--seed", str(seed)]
        params = StdioServerParameters(
            command=sys.executable,
            args=args,
            # the SDK's minimal default env (no API keys) plus the project on the path
            env={**get_default_environment(), "PYTHONPATH": str(PROJECT_ROOT)},
            cwd=str(PROJECT_ROOT),
        )
        return cls(lambda: stdio_client(params), label="stdio")

    @classmethod
    def in_memory(cls, env: Environment) -> MCPToolClient:
        """Run the MCP server in-process over the SDK's in-memory transport (for tests)."""
        server = create_server(env)
        return cls(lambda: InMemoryTransport(server), label="in-memory")

    # ---------- lifecycle ----------

    def __enter__(self) -> MCPToolClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, name=f"mcp-client-{self._label}", daemon=True)
        self._thread.start()
        if not self._ready.wait(START_TIMEOUT_S):
            raise TimeoutError(f"MCP {self._label} session did not start within {START_TIMEOUT_S}s")
        if self._error is not None:
            raise RuntimeError(f"MCP {self._label} session failed to start: {self._error}") from self._error

    def close(self) -> None:
        if self._loop is not None and self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # surfaced to start() via _error
            self._error = self._error or exc
        finally:
            self._ready.set()
            loop.close()

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with self._connect() as streams:
                read_stream, write_stream = streams[0], streams[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._tools = list((await session.list_tools()).tools)
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            self._session = None

    # ---------- tools ----------

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    def function_specs(self) -> list[dict[str, Any]]:
        """Function-calling specs for the LLM, built from the server's tool list."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": _strip_titles(t.input_schema),
                },
            }
            for t in self._tools
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self.tool_names:
            return ToolResult(f"ERROR: unknown tool '{name}'. Available: {sorted(self.tool_names)}", True)
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP session is not running; use MCPToolClient as a context manager")
        future = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, arguments), self._loop)
        result = future.result(timeout=CALL_TIMEOUT_S)
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        return ToolResult(f"ERROR: {text}" if result.is_error else text, bool(result.is_error))
