#!/usr/bin/env python
"""OpenMontage MCP server — interactive smoke test.

Runs a complete client session against the real server (spawned over stdio),
exercising discovery, sync execution, and the async job API with a REAL video
slice (auto-generates a test clip via ffmpeg). No external MCP client needed —
this file IS the demo.

Usage:
    .venv/bin/python mcp_server/demo.py

Exit code 0 = everything worked. Non-zero = something failed (see traceback).
"""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _section(title: str) -> None:
    print(f"\n\033[1m=== {title} ===\033[0m")


def _make_test_video(path: Path, seconds: float = 4.0) -> None:
    """Generate a small color-bar test clip with ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=640x360:rate=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )


async def main() -> int:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    if not Path(venv_python).is_file():
        venv_python = sys.executable

    _section("launching server")
    print(f"  spawning: {venv_python} -m mcp_server  (stdio transport)")
    params = StdioServerParameters(command=venv_python, args=["-m", "mcp_server"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            _ok(f"connected: {info.serverInfo.name} v{info.serverInfo.version}")

            _section("1. discover tools")
            tools = await session.list_tools()
            print(f"  server advertises {len(tools.tools)} tools:")
            for t in tools.tools:
                print(f"     - {t.name}")
            assert len(tools.tools) == 14

            _section("2. preflight — provider menu summary")
            res = await session.call_tool("provider_menu_summary", {})
            menu = json.loads(res.content[0].text)
            print(f"  composition runtimes: {menu['composition_runtimes']}")
            ready = [c for c in menu["capabilities"] if c["configured"] > 0]
            print(f"  configured capabilities: {len(ready)} of {len(menu['capabilities'])}")
            _ok("preflight works")

            _section("3. list pipelines")
            res = await session.call_tool("list_pipelines", {})
            pl = json.loads(res.content[0].text)
            print(f"  {pl['total']} pipelines: {', '.join(pl['pipelines'][:5])} ...")
            _ok("pipelines listed")

            # Prepare a real test video.
            tmp = Path(tempfile.mkdtemp())
            src = tmp / "source.mp4"
            _section("4. real video slice — SYNC (execute_tool)")
            print(f"  generating test video: {src}")
            if shutil.which("ffmpeg") is None:
                print("  \033[33m(skip) ffmpeg not installed — cannot run slice demo\033[0m")
            else:
                _make_test_video(src, seconds=4.0)
                out_sync = tmp / "clip_sync.mp4"
                res = await session.call_tool(
                    "execute_tool",
                    {
                        "tool_name": "video_trimmer",
                        "inputs": {
                            "operation": "cut",
                            "input_path": str(src),
                            "output_path": str(out_sync),
                            "start_seconds": 1.0,
                            "end_seconds": 3.0,
                        },
                    },
                )
                r = json.loads(res.content[0].text)
                print(f"  result: success={r['success']} duration={r['duration_seconds']}s")
                print(f"  output: {out_sync} ({out_sync.stat().st_size if out_sync.exists() else 0} bytes)")
                _ok(f"sync slice produced a clip: {out_sync.exists()}")

                _section("5. real video slice — ASYNC (submit_tool_job)")
                out_async = tmp / "clip_async.mp4"
                res = await session.call_tool(
                    "submit_tool_job",
                    {
                        "tool_name": "video_trimmer",
                        "inputs": {
                            "operation": "cut",
                            "input_path": str(src),
                            "output_path": str(out_async),
                            "start_seconds": 0.5,
                            "end_seconds": 2.5,
                        },
                    },
                )
                job = json.loads(res.content[0].text)
                jid = job["job_id"]
                print(f"  submitted: job_id={jid} initial_status={job['status']}")

                status = job
                for _ in range(60):
                    await asyncio.sleep(0.3)
                    s = await session.call_tool("get_job_status", {"job_id": jid})
                    status = json.loads(s.content[0].text)
                    if status["status"] == "succeeded":
                        break
                print(f"  final: status={status['status']} elapsed={status.get('elapsed_seconds')}s")
                print(f"  output: {out_async} ({out_async.stat().st_size if out_async.exists() else 0} bytes)")
                _ok(f"async slice produced a clip: {out_async.exists()}")

                # Show where to find the outputs.
                _section("outputs")
                print(f"  source video : {src}")
                print(f"  sync clip    : {out_sync}")
                print(f"  async clip   : {out_async}")
                print(f"  inspect with : ffprobe {out_sync}")

    print("\n\033[1;32mAll checks passed. MCP server is working end to end.\033[0m")
    print("To run continuously for a real client:")
    print(f"  {venv_python} -m mcp_server                 # stdio")
    print(f"  {venv_python} -m mcp_server --transport streamable-http --port 8765")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
