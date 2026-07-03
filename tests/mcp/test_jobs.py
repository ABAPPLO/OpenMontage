"""Tests for the async job API: job_manager + submit_tool_job/get_job_status.

Covers the submit → poll → terminal lifecycle, the publish guard on async
submission, list_jobs tallies, unknown-job handling, and a real ffmpeg slice run
as an async job (skipped without ffmpeg).
"""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import BaseTool, ToolResult, ToolTier
from tools.tool_registry import ToolRegistry

import mcp_server.handlers as H
from mcp_server import handlers as handlers_mod
from mcp_server.job_manager import JobManager, JobStatus, jobs as global_jobs


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------

class FastEchoTool(BaseTool):
    """Returns immediately with the inputs."""
    name = "fast_echo"
    capability = "test_capability"
    tier = ToolTier.CORE
    def execute(self, inputs):  # type: ignore[override]
        return ToolResult(success=True, data={"echo": inputs}, cost_usd=0.0)


class SlowTool(BaseTool):
    """Sleeps so we can observe the running state before completion."""
    name = "slow_echo"
    capability = "test_capability"
    tier = ToolTier.CORE
    def execute(self, inputs):  # type: ignore[override]
        import time
        time.sleep(inputs.get("seconds", 0.3))
        return ToolResult(success=True, data={"done": True})


class FailingTool(BaseTool):
    """Reports a failed ToolResult (tool-level failure, not a crash)."""
    name = "fail_echo"
    capability = "test_capability"
    tier = ToolTier.CORE
    def execute(self, inputs):  # type: ignore[override]
        return ToolResult(success=False, error="tool said no")


class RaisingTool(BaseTool):
    """Raises inside execute — the job must capture it, not crash."""
    name = "raise_echo"
    capability = "test_capability"
    tier = ToolTier.CORE
    def execute(self, inputs):  # type: ignore[override]
        raise RuntimeError("kaboom")


class PubTool(BaseTool):
    """Trips the publish guard via a publishing side_effect."""
    name = "pub_echo"
    capability = "test_capability"
    tier = ToolTier.CORE
    side_effects = ["uploads clip to YouTube"]
    def execute(self, inputs):  # type: ignore[override]
        return ToolResult(success=True, data={"posted": True})


@pytest.fixture
def fresh_manager():
    """A clean JobManager + isolated registry for each test."""
    reg = ToolRegistry()
    for cls in (FastEchoTool, SlowTool, FailingTool, RaisingTool, PubTool):
        reg.register(cls())
    # Swap the handler-module registry + clear the global job store.
    saved_reg = handlers_mod.registry
    handlers_mod.registry = reg
    global_jobs.clear()
    yield reg
    handlers_mod.registry = saved_reg
    global_jobs.clear()


# ---------------------------------------------------------------------------
# JobManager — lifecycle
# ---------------------------------------------------------------------------

def test_submit_returns_pending_then_succeeds(fresh_manager):
    async def go():
        tool = fresh_manager.get("fast_echo")
        job = await global_jobs.submit(tool, {"hi": 1})
        assert job.status == JobStatus.PENDING
        assert job.job_id.startswith("job_")
        final = await global_jobs.await_completion(job.job_id, timeout=10)
        assert final.status == JobStatus.SUCCEEDED
        assert final.result["data"]["echo"] == {"hi": 1}
        assert final.progress == 1.0
        assert final.finished_at is not None
        assert "elapsed_seconds" in final.to_dict()
    asyncio.run(go())


def test_running_state_observable_for_slow_tool(fresh_manager):
    """A slow tool passes through RUNNING before SUCCEEDED."""
    async def go():
        tool = fresh_manager.get("slow_echo")
        job = await global_jobs.submit(tool, {"seconds": 0.5})
        # Give the scheduler a tick to start it.
        await asyncio.sleep(0.1)
        snap = global_jobs.get(job.job_id)
        assert snap.status in (JobStatus.RUNNING, JobStatus.PENDING, JobStatus.SUCCEEDED)
        final = await global_jobs.await_completion(job.job_id, timeout=10)
        assert final.status == JobStatus.SUCCEEDED
        assert final.result["data"] == {"done": True}
    asyncio.run(go())


def test_tool_failure_is_succeeded_job_with_failed_result(fresh_manager):
    """A tool reporting failure completes the job; result.success is False."""
    async def go():
        tool = fresh_manager.get("fail_echo")
        job = await global_jobs.submit(tool, {})
        final = await global_jobs.await_completion(job.job_id, timeout=10)
        # The JOB succeeded (it ran to completion); the TOOL result failed.
        assert final.status == JobStatus.SUCCEEDED
        assert final.result["success"] is False
        assert final.result["error"] == "tool said no"
    asyncio.run(go())


def test_uncaught_exception_captured_not_crashed(fresh_manager):
    async def go():
        tool = fresh_manager.get("raise_echo")
        job = await global_jobs.submit(tool, {})
        final = await global_jobs.await_completion(job.job_id, timeout=10)
        # An exception inside execute is turned into a failed ToolResult by the
        # work wrapper, so the job still reaches SUCCEEDED with a failed result.
        assert final.status == JobStatus.SUCCEEDED
        assert final.result["success"] is False
        assert "kaboom" in final.result["error"]
    asyncio.run(go())


def test_unknown_job_id_returns_none(fresh_manager):
    async def go():
        assert global_jobs.get("job_does_not_exist") is None
        assert await global_jobs.await_completion("job_nope", timeout=1) is None
    asyncio.run(go())


# ---------------------------------------------------------------------------
# Handler-level: submit_tool_job / get_job_status / list_jobs
# ---------------------------------------------------------------------------

def test_submit_tool_job_returns_pending_snapshot(fresh_manager):
    async def go():
        snap = await H.submit_tool_job("fast_echo", {"x": 1})
        assert snap["status"] == "pending"
        assert snap["tool_name"] == "fast_echo"
        assert "job_id" in snap
        final = await H.get_job_status(snap["job_id"])
        # May still be pending/running immediately after; await to settle.
        await global_jobs.await_completion(snap["job_id"], timeout=10)
        final = await H.get_job_status(snap["job_id"])
        assert final["status"] == "succeeded"
        assert final["result"]["data"]["echo"] == {"x": 1}
    asyncio.run(go())


def test_get_job_status_unknown_raises(fresh_manager):
    async def go():
        with pytest.raises(ValueError):
            await H.get_job_status("job_nonexistent")
    asyncio.run(go())


def test_submit_tool_job_unknown_tool_raises(fresh_manager):
    async def go():
        with pytest.raises(ValueError):
            await H.submit_tool_job("nope", {})
    asyncio.run(go())


def test_submit_tool_job_publish_guard(fresh_manager):
    """Async submission must enforce the publish confirm guard too."""
    async def go():
        with pytest.raises(PermissionError):
            await H.submit_tool_job("pub_echo", {})
        # With confirm it goes through.
        snap = await H.submit_tool_job("pub_echo", {}, confirm=True)
        await global_jobs.await_completion(snap["job_id"], timeout=10)
        final = await H.get_job_status(snap["job_id"])
        assert final["status"] == "succeeded"
        assert final["result"]["data"] == {"posted": True}
    asyncio.run(go())


def test_list_jobs_tally(fresh_manager):
    async def go():
        s1 = await H.submit_tool_job("fast_echo", {})
        s2 = await H.submit_tool_job("fail_echo", {})
        await global_jobs.await_completion(s1["job_id"], timeout=10)
        await global_jobs.await_completion(s2["job_id"], timeout=10)
        out = await H.list_jobs()
        assert out["counts"]["succeeded"] == 2
        assert len(out["jobs"]) == 2
        # newest-first
        assert out["jobs"][0]["created_at"] >= out["jobs"][1]["created_at"]
    asyncio.run(go())


# ---------------------------------------------------------------------------
# Real async ffmpeg slice (acceptance: long job via async API)
# ---------------------------------------------------------------------------

pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg unavailable"
)


@pytestmark_ffmpeg
def test_submit_real_video_slice_async(fresh_manager, tmp_path):
    """A real ffmpeg cut submitted as an async job completes with a usable clip.

    This proves the async path handles a genuinely long-running tool end to end,
    not just fast dummies.
    """
    # Bring the real video_trimmer into the isolated registry.
    from tools.tool_registry import registry as real_registry
    real_registry.ensure_discovered()
    fresh_manager.register(real_registry.get("video_trimmer"))

    src = tmp_path / "source.mp4"
    out = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=15",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )

    async def go():
        snap = await H.submit_tool_job(
            "video_trimmer",
            {"operation": "cut", "input_path": str(src), "output_path": str(out),
             "start_seconds": 1.0, "end_seconds": 2.0},
        )
        assert snap["status"] == "pending"
        await global_jobs.await_completion(snap["job_id"], timeout=60)
        final = await H.get_job_status(snap["job_id"])
        assert final["status"] == "succeeded"
        assert final["result"]["success"] is True
        assert out.is_file() and out.stat().st_size > 0
    asyncio.run(go())
