"""Tests for mcp_server.execution — thread-pool execution + ToolResult serialization.

Mirrors the DummyTool pattern from tests/contracts/test_phase0_contracts.py.
Does not require API keys, GPU, or the full registry.
"""

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import BaseTool, ToolResult, ToolTier

from mcp_server.execution import (
    execute_tool_async,
    run_blocking,
    serialize_result,
    _scrub_secrets,
)


class DummyTool(BaseTool):
    """Minimal tool that echoes inputs and flags the success path."""

    name = "dummy_echo"
    tier = ToolTier.CORE
    capabilities = ["test"]

    def execute(self, inputs):  # type: ignore[override]
        if inputs.get("fail"):
            return ToolResult(success=False, error="boom")
        return ToolResult(
            success=True,
            data={"echo": inputs, "api_key": "sk-secret-123"},
            artifacts=["/tmp/out.mp4"],
            cost_usd=0.01,
            duration_seconds=1.2,
            model="dummy-model",
        )


class RaisingTool(BaseTool):
    """Tool whose execute raises — execute_tool_async must not crash the loop."""

    name = "dummy_raise"
    tier = ToolTier.CORE
    capabilities = ["test"]

    def execute(self, inputs):  # type: ignore[override]
        raise RuntimeError("unexpected")


def test_run_blocking_runs_in_thread():
    """run_blocking should await a blocking callable and return its value."""
    result = asyncio.run(run_blocking(sum, [1, 2, 3]))
    assert result == 6


def test_serialize_result_preserves_fields():
    result = ToolResult(
        success=True,
        data={"x": 1},
        artifacts=["a.mp4"],
        cost_usd=0.5,
        duration_seconds=2.0,
        model="m",
    )
    out = serialize_result(result)
    assert out["success"] is True
    assert out["data"] == {"x": 1}
    assert out["artifacts"] == ["a.mp4"]
    assert out["cost_usd"] == 0.5
    assert out["duration_seconds"] == 2.0
    assert out["model"] == "m"
    assert out["error"] is None


def test_serialize_result_scrubs_secret_values():
    """A value under a secret-looking key must be redacted, others kept."""
    result = ToolResult(success=True, data={"api_key": "sk-secret", "input_path": "/a.mp4"})
    out = serialize_result(result)
    assert out["data"]["api_key"] == "<redacted>"
    assert out["data"]["input_path"] == "/a.mp4"


def test_scrub_secrets_nested():
    assert _scrub_secrets({"token": "abc", "nested": {"TOKEN": "x", "ok": 1}}) == {
        "token": "<redacted>",
        "nested": {"TOKEN": "<redacted>", "ok": 1},
    }
    assert _scrub_secrets(["a", {"password": "p"}]) == ["a", {"password": "<redacted>"}]


def test_execute_tool_async_success():
    out = asyncio.run(execute_tool_async(DummyTool(), {"hello": "world"}))
    assert out.success is True
    assert out.data["echo"] == {"hello": "world"}
    assert out.model == "dummy-model"


def test_execute_tool_async_failure_result():
    out = asyncio.run(execute_tool_async(DummyTool(), {"fail": True}))
    assert out.success is False
    assert out.error == "boom"


def test_execute_tool_async_swallows_unexpected_raise():
    """An uncaught exception becomes a failed result, not a crash."""
    out = asyncio.run(execute_tool_async(RaisingTool(), {}))
    assert out.success is False
    assert "unexpected" in (out.error or "")
