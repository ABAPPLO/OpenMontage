"""End-to-end test: real video slicing through execute_tool → video_trimmer.

Generates a short test video with ffmpeg, then drives the full MCP handler path
(execute_tool → registry → VideoTrimmer.execute → ffmpeg) to confirm a real clip
is produced. Skipped when ffmpeg is unavailable, so CI without ffmpeg still passes.
"""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not available — cannot run real video-slice e2e test",
)


def _make_test_video(path: Path, seconds: float = 3.0) -> None:
    """Generate a tiny deterministic testclip (color test pattern + silent audio)."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=15",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def test_execute_tool_real_video_slice(tmp_path):
    """execute_tool('video_trimmer', cut) produces a real trimmed clip.

    This is the acceptance criterion from the plan: "execute_tool('video_trimmer',
    {...}) can really slice out a video clip". Uses the real registry singleton.
    """
    import mcp_server.handlers as H

    src = tmp_path / "source.mp4"
    out = tmp_path / "clip.mp4"
    _make_test_video(src, seconds=3.0)
    assert src.is_file() and src.stat().st_size > 0

    # Cut the middle second [1.0, 2.0) out of a 3s clip.
    result = asyncio.run(
        H.execute_tool(
            "video_trimmer",
            {
                "operation": "cut",
                "input_path": str(src),
                "output_path": str(out),
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            },
        )
    )

    assert result["success"], f"expected success, got error: {result['error']}"
    assert result["duration_seconds"] >= 0
    # The trimmed clip must exist and be shorter than the source.
    assert out.is_file(), "output clip was not created"
    assert out.stat().st_size > 0
    assert _probe_duration(out) < _probe_duration(src)


def test_execute_tool_real_video_slice_failure_on_missing_input(tmp_path):
    """A non-existent input surfaces a clean failed result, not a crash."""
    import mcp_server.handlers as H

    result = asyncio.run(
        H.execute_tool(
            "video_trimmer",
            {
                "operation": "cut",
                "input_path": str(tmp_path / "does-not-exist.mp4"),
                "output_path": str(tmp_path / "out.mp4"),
                "start_seconds": 0,
                "end_seconds": 1,
            },
        )
    )
    assert result["success"] is False
    assert result["error"]
