"""End-to-end tests that spawn the real MCP server over stdio and call tools.

These guard against wiring bugs that unit tests (which call handlers directly
with ctx=None) miss — e.g. passing the wrong object as the MCP Context. They
launch the server as a subprocess exactly as a client would, so the full
FastMCP tool registration + Context injection path is exercised.
"""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="MCP server requires Python 3.10+"
)


async def _call_server(tool_name: str, arguments: dict) -> dict:
    """Spawn the server over stdio, call one tool, return parsed JSON result.

    Uses the real mcp client library so the wire protocol is exercised end to end.
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    if not Path(venv_python).is_file():
        venv_python = sys.executable  # fall back to whatever ran pytest

    params = StdioServerParameters(command=venv_python, args=["-m", "mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(tool_name, arguments)
            assert not res.isError, f"tool {tool_name} returned an error: {res.content}"
            return json.loads(res.content[0].text)


@pytest.mark.asyncio
async def test_server_lists_all_tools():
    """The spawned server must advertise all expected tools (discovery,
    execution, async job, orchestration, and the 3 video-segmentation tools)."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    if not Path(venv_python).is_file():
        venv_python = sys.executable
    params = StdioServerParameters(command=venv_python, args=["-m", "mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
    assert len(names) == 14
    for expected in (
        "discover_tools", "provider_menu_summary", "get_tool_info", "execute_tool",
        "submit_tool_job", "get_job_status", "list_jobs",
        "list_pipelines", "get_pipeline_manifest",
        "read_checkpoint", "write_checkpoint",
        "segment_shots", "segment_by_face", "segment_filter",
    ):
        assert expected in names, f"missing tool {expected}"


@pytest.mark.asyncio
async def test_execute_tool_over_stdio_uses_context():
    """execute_tool must work when called through the real server (Context wired).

    Regression guard: an earlier bug passed the FastMCP server object as ctx,
    which crashed on ctx.info(). This test calls execute_tool end to end.
    """
    out = await _call_server(
        "execute_tool",
        {
            "tool_name": "video_trimmer",
            "inputs": {
                "operation": "cut",
                "input_path": "/nonexistent.mp4",
                "output_path": "/tmp/stdio_e2e.mp4",
                "start_seconds": 0,
                "end_seconds": 1,
            },
        },
    )
    # The tool fails (missing input) but the *call* must succeed and return a
    # well-formed ToolResult — proving the Context path didn't crash.
    assert out["success"] is False
    assert out["error"]


@pytest.mark.asyncio
async def test_async_job_over_stdio_uses_context():
    """submit_tool_job + get_job_status must work through the real server.

    Regression guard for the Context wiring on the async submission path.
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    if not Path(venv_python).is_file():
        venv_python = sys.executable
    params = StdioServerParameters(command=venv_python, args=["-m", "mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            j = await session.call_tool(
                "submit_tool_job",
                {
                    "tool_name": "video_trimmer",
                    "inputs": {
                        "operation": "cut",
                        "input_path": "/nonexistent.mp4",
                        "output_path": "/tmp/stdio_e2e_job.mp4",
                        "start_seconds": 0,
                        "end_seconds": 1,
                    },
                },
            )
            assert not j.isError, f"submit failed: {j.content}"
            job = json.loads(j.content[0].text)
            assert job["status"] in ("pending", "running", "succeeded")
            jid = job["job_id"]

            # Poll to terminal.
            status = job
            for _ in range(40):
                await asyncio.sleep(0.3)
                s = await session.call_tool("get_job_status", {"job_id": jid})
                assert not s.isError, f"status failed: {s.content}"
                status = json.loads(s.content[0].text)
                if status["status"] == "succeeded":
                    break
            assert status["status"] == "succeeded"
            assert "result" in status
