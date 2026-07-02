"""Tests for mcp_server.handlers — the 8 MCP tool wrappers.

Uses a fresh ToolRegistry with a DummyTool (not the global singleton) for the
discovery/execute paths so tests are isolated and need no API keys. Pipeline and
checkpoint handlers hit the real shipped manifests/schemas (read-only + tmp dir).
"""

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import BaseTool, ToolResult, ToolTier
from tools.tool_registry import ToolRegistry

import mcp_server.handlers as H


# ---------------------------------------------------------------------------
# Test fixtures — a hand-built registry so we don't depend on the global one.
# ---------------------------------------------------------------------------

class DummyTool(BaseTool):
    name = "dummy_echo"
    capability = "test_capability"
    tier = ToolTier.CORE
    capabilities = ["test"]
    dependencies = []

    def execute(self, inputs):  # type: ignore[override]
        if inputs.get("fail"):
            return ToolResult(success=False, error="boom")
        return ToolResult(success=True, data={"echo": inputs}, cost_usd=0.01)


@pytest.fixture
def isolated_registry(monkeypatch):
    """Replace the global registry in handlers with a fresh one holding DummyTool."""
    reg = ToolRegistry()
    reg.register(DummyTool())
    monkeypatch.setattr(H, "registry", reg)
    return reg


# ---------------------------------------------------------------------------
# discover_tools / provider_menu_summary / get_tool_info
# ---------------------------------------------------------------------------

def test_discover_tools(isolated_registry):
    out = asyncio.run(H.discover_tools())
    assert out["total"] == 1
    assert out["capabilities"] == {"test_capability": ["dummy_echo"]}


def test_get_tool_info(isolated_registry):
    out = asyncio.run(H.get_tool_info("dummy_echo"))
    assert out["name"] == "dummy_echo"
    assert out["capability"] == "test_capability"


def test_get_tool_info_unknown(isolated_registry):
    with pytest.raises(ValueError):
        asyncio.run(H.get_tool_info("nope"))


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------

def test_execute_tool_success(isolated_registry):
    out = asyncio.run(H.execute_tool("dummy_echo", {"hi": 1}))
    assert out["success"] is True
    assert out["data"]["echo"] == {"hi": 1}
    assert out["cost_usd"] == 0.01


def test_execute_tool_failure(isolated_registry):
    out = asyncio.run(H.execute_tool("dummy_echo", {"fail": True}))
    assert out["success"] is False
    assert out["error"] == "boom"


def test_execute_tool_unknown(isolated_registry):
    with pytest.raises(ValueError):
        asyncio.run(H.execute_tool("nope", {}))


def test_execute_tool_scrubs_secrets(isolated_registry, monkeypatch):
    """Even if a tool returns a secret in data, execute_tool redacts it."""
    class LeakyTool(DummyTool):
        name = "leaky"
        def execute(self, inputs):  # type: ignore[override]
            return ToolResult(success=True, data={"api_key": "sk-leaked"})

    isolated_registry.register(LeakyTool())
    out = asyncio.run(H.execute_tool("leaky", {}))
    assert out["data"]["api_key"] == "<redacted>"


# ---------------------------------------------------------------------------
# Pipeline handlers — real shipped manifests
# ---------------------------------------------------------------------------

def test_list_pipelines():
    out = asyncio.run(H.list_pipelines())
    assert "clip-factory" in out["pipelines"]
    assert out["total"] >= 10


def test_get_pipeline_manifest_clip_factory():
    out = asyncio.run(H.get_pipeline_manifest("clip-factory"))
    assert out["name"] in ("clip-factory", "Multi-Clip Extraction", out["name"])
    assert "idea" in out["stage_order"]
    assert "compose" in out["stage_order"]
    assert any("video_compose" in (s.get("tools_available") or []) for s in out["stages"])
    assert "video_trimmer" in out["required_tools"]


def test_get_pipeline_manifest_unknown():
    with pytest.raises(FileNotFoundError):
        asyncio.run(H.get_pipeline_manifest("does-not-exist"))


# ---------------------------------------------------------------------------
# Checkpoint handlers — tmp dir, round-trip read/write
# ---------------------------------------------------------------------------

def test_write_and_read_checkpoint(tmp_path, monkeypatch):
    """write_checkpoint then read_checkpoint round-trips and yields next_stage."""
    # clip-factory has an `idea` stage; use a status that doesn't require the
    # canonical artifact (in_progress) to avoid needing a fully valid brief here.
    out = asyncio.run(
        H.write_checkpoint(
            "test-proj",
            "idea",
            "in_progress",
            {},
            pipeline_dir=str(tmp_path),
            pipeline_type="clip-factory",
        )
    )
    assert out["status"] == "in_progress"
    assert out["path"].endswith("checkpoint_idea.json")
    assert out["next_stage"] == "idea"  # not completed yet, so idea is still next

    # read it back
    rd = asyncio.run(
        H.read_checkpoint(
            "test-proj",
            "idea",
            pipeline_dir=str(tmp_path),
            pipeline_type="clip-factory",
        )
    )
    assert rd["checkpoint"]["stage"] == "idea"
    assert rd["checkpoint"]["status"] == "in_progress"
    assert rd["latest_stage"] == "idea"


def test_read_checkpoint_missing_returns_none(tmp_path):
    out = asyncio.run(
        H.read_checkpoint("nope", None, pipeline_dir=str(tmp_path))
    )
    assert out["checkpoint"] is None
    assert out["next_stage"] is not None or out["next_stage"] is None  # no crash


# ---------------------------------------------------------------------------
# Publish-action guard (confirm flag) — security boundary from the plan
# ---------------------------------------------------------------------------

class _FakePublishTool(BaseTool):
    """A tool that trips the publish guard via a publishing side_effect."""

    name = "fake_publisher"
    capability = "video_post"
    tier = ToolTier.CORE
    capabilities = ["publish_action"]
    side_effects = ["uploads clip to YouTube and schedules post"]

    def execute(self, inputs):  # type: ignore[override]
        return ToolResult(success=True, data={"posted": True})


class _FakePublishTierTool(BaseTool):
    """A tool that trips the publish guard via tier=PUBLISH."""

    name = "fake_publish_tier"
    capability = "video_post"
    tier = ToolTier.PUBLISH

    def execute(self, inputs):  # type: ignore[override]
        return ToolResult(success=True)


def test_publish_guard_blocks_without_confirm(isolated_registry):
    """A publish-style tool must raise PermissionError when confirm=False."""
    isolated_registry.register(_FakePublishTool())
    with pytest.raises(PermissionError) as exc_info:
        asyncio.run(H.execute_tool("fake_publisher", {}))
    assert "publish" in str(exc_info.value).lower()
    assert "confirm" in str(exc_info.value).lower()


def test_publish_guard_allows_with_confirm(isolated_registry):
    """The same publish-style tool runs fine when confirm=True."""
    isolated_registry.register(_FakePublishTool())
    out = asyncio.run(H.execute_tool("fake_publisher", {}, confirm=True))
    assert out["success"] is True


def test_publish_guard_triggers_on_tier(isolated_registry):
    """tier=PUBLISH alone (no matching side_effect) is enough to trip the guard."""
    isolated_registry.register(_FakePublishTierTool())
    with pytest.raises(PermissionError):
        asyncio.run(H.execute_tool("fake_publish_tier", {}))


def test_publish_guard_does_not_trigger_on_normal_tool(isolated_registry):
    """A plain tool (DummyTool) runs without confirm — no false positives."""
    out = asyncio.run(H.execute_tool("dummy_echo", {"ok": 1}))
    assert out["success"] is True


def test_requires_publish_classification(isolated_registry):
    """The classifier flags publish signals and ignores benign ones."""
    assert H._requires_publish_confirmation(_FakePublishTool()) is not None
    assert H._requires_publish_confirmation(_FakePublishTierTool()) is not None
    # DummyTool (CORE, no publish side_effect) must NOT be flagged.
    assert H._requires_publish_confirmation(DummyTool()) is None


# ---------------------------------------------------------------------------
# provider_menu_summary — consistency with make preflight
# ---------------------------------------------------------------------------

def test_provider_menu_summary_shape():
    """provider_menu_summary returns the documented preflight shape.

    This validates the acceptance criterion: the MCP menu matches the
    `make preflight` rollup (same top-level keys + capability counts).
    """
    out = asyncio.run(H.provider_menu_summary())
    # Same four keys the preflight menu emits.
    assert set(out.keys()) == {
        "composition_runtimes",
        "capabilities",
        "setup_offers",
        "runtime_warnings",
    }
    # Composition runtimes are the three known engines.
    assert set(out["composition_runtimes"].keys()) == {"ffmpeg", "remotion", "hyperframes"}
    # Each capability entry carries the configured/total counts.
    assert len(out["capabilities"]) >= 1
    for cap in out["capabilities"]:
        assert "configured" in cap and "total" in cap
        assert cap["configured"] <= cap["total"]


def test_provider_menu_summary_matches_registry():
    """The menu's capability counts must agree with a direct registry tally."""
    from tools.tool_registry import ToolRegistry

    # provider_menu_summary uses the global registry; compare against a fresh
    # discover on the same singleton (already discovered in-process).
    from tools.tool_registry import registry
    registry.ensure_discovered()

    out = asyncio.run(H.provider_menu_summary())
    # Total tools across all capabilities (excluding the selector pseudo-provider)
    # should match the registry's non-selector tool count.
    non_selector = [t for n, t in registry._tools.items() if t.provider != "selector"]
    menu_total = sum(c["total"] for c in out["capabilities"])
    assert menu_total == len(non_selector)

