"""Execution helpers — run blocking tools off the event loop and serialize results.

OpenMontage tools are synchronous and blocking (FFmpeg/Remotion can run for tens
of seconds to minutes). The MCP server's message loop must not stall while a
tool runs, so we dispatch ``tool.execute()`` to a worker thread via
``anyio.to_thread.run_sync`` (the bridge FastMCP's async handlers use).

This module also serializes a ``ToolResult`` dataclass into a plain dict the MCP
wire format can carry, and scrubs values that must never leak to an MCP client
(API keys / tokens).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from tools.base_tool import BaseTool, ToolResult


# Substrings in artifact/field names that hint at secret material. Used to keep
# credentials out of MCP responses — the server reuses the repo .env, but clients
# (which may be remote) should never see raw keys.
_SECRET_HINTS = ("key", "token", "secret", "password", "passwd", "credential", "api_key")


def _looks_secret(field_name: str) -> bool:
    lowered = field_name.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)


def _scrub_secrets(value: Any) -> Any:
    """Recursively redact dict values whose key looks like a secret.

    Only string values are redacted; nested structures are walked. Non-secret
    data passes through unchanged so legitimate fields like ``input_path`` survive.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and _looks_secret(k) and isinstance(v, str):
                out[k] = "<redacted>"
            else:
                out[k] = _scrub_secrets(v)
        return out
    if isinstance(value, list):
        return [_scrub_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_secrets(item) for item in value)
    return value


def serialize_result(result: ToolResult) -> dict[str, Any]:
    """Convert a ToolResult into a JSON-safe dict for MCP transport.

    The dict mirrors the dataclass field names so clients get a stable shape:
    success, data (secret-scrubbed), artifacts, error, cost_usd, duration_seconds,
    seed, model.
    """
    return {
        "success": result.success,
        "data": _scrub_secrets(result.data),
        "artifacts": list(result.artifacts),
        "error": result.error,
        "cost_usd": result.cost_usd,
        "duration_seconds": result.duration_seconds,
        "seed": result.seed,
        "model": result.model,
    }


async def run_blocking(func: Callable[..., Any], *args: Any) -> Any:
    """Run a blocking callable in a worker thread and await its result.

    Uses anyio (bundled with the MCP SDK) so we stay compatible with FastMCP's
    event loop. Falls back to the default executor.
    """
    # Imported lazily so the module imports cleanly even if anyio is unavailable
    # at collection time (tests import functions directly without a server).
    import anyio

    return await anyio.to_thread.run_sync(lambda: func(*args))


async def execute_tool_async(tool: BaseTool, inputs: dict[str, Any]) -> ToolResult:
    """Execute a tool off the event loop, returning the raw ToolResult.

    The caller is expected to have resolved the tool (e.g. via the registry) and
    validated inputs. Exceptions from ``execute()`` are caught by the tool itself
    and surfaced as ``ToolResult(success=False, error=...)`` in nearly all cases;
    we additionally guard here so an unexpected raise becomes a failed result
    rather than crashing the MCP request.
    """
    def _run() -> ToolResult:
        try:
            return tool.execute(inputs)
        except Exception as exc:  # noqa: BLE001 — surface any raise as a result
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")

    return await run_blocking(_run)
