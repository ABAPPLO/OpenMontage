"""Tests for the three video-segmentation MCP handlers.

These handlers are thin wrappers over the segment_shots / segment_by_face /
segment_filter BaseTools, which themselves orchestrate other registered tools
(scene_detect, video_trimmer, face_tracker, silence_cutter, ...). We exercise
the handlers with a hand-built registry of fake sub-tools so the tests need no
real video files, no CV libraries (insightface/opencv/mediapipe), and no network.

What's covered:
  - segment_shots: happy path (boundaries -> shot objects), error propagation
    from scene_detect, max_clips truncation, extract_clips wiring.
  - segment_by_face: missing-dependency error path (insightface not installed),
    empty-faces path.
  - segment_filter: duration predicates, shot_size normalization + AND logic,
    rejected[] reasons, missing-predicate validation.
  - Generic dispatch: unknown tool name raises ValueError.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import BaseTool, ToolResult, ToolStatus, ToolTier
from tools.tool_registry import ToolRegistry

import mcp_server.handlers as H
from tools.analysis.segment_shots import SegmentShots
from tools.analysis.segment_filter import SegmentFilter


# ---------------------------------------------------------------------------
# Fake sub-tools that the segment tools orchestrate. Each records its inputs so
# we can assert on the call, and returns canned data.
# ---------------------------------------------------------------------------

class _FakeSceneDetect(BaseTool):
    name = "scene_detect"
    capability = "analysis"
    tier = ToolTier.CORE
    capabilities = ["detect_scenes"]
    dependencies = []

    def __init__(self, scenes=None, fail=False):
        self._scenes = scenes or []
        self._fail = fail
        self.calls: list[dict] = []

    def execute(self, inputs):
        self.calls.append(inputs)
        if self._fail:
            return ToolResult(success=False, error="scene_detect boom")
        return ToolResult(
            success=True,
            data={"scene_count": len(self._scenes), "scenes": self._scenes, "method": "fake"},
        )


class _FakeVideoTrimmer(BaseTool):
    name = "video_trimmer"
    capability = "video_post"
    tier = ToolTier.CORE
    capabilities = ["cut"]
    dependencies = []

    def __init__(self):
        self.calls: list[dict] = []

    def execute(self, inputs):
        self.calls.append(inputs)
        # Pretend to write the output file so the tool's existence check passes.
        out = Path(inputs.get("output_path", "/tmp/x.mp4"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return ToolResult(success=True, data={"operation": inputs["operation"], "output": str(out)})


class _FakeVideoUnderstand(BaseTool):
    name = "video_understand"
    capability = "analysis"
    tier = ToolTier.ANALYZE
    capabilities = ["classify"]
    dependencies = []

    def __init__(self, label="medium", desc="a person talking"):
        self._label = label
        self._desc = desc

    def get_status(self):
        return ToolStatus.AVAILABLE

    def execute(self, inputs):
        return ToolResult(
            success=True,
            data={"summary": f"{self._label} shot: {self._desc}", "frames": []},
        )


class _FakeFaceTracker(BaseTool):
    name = "face_tracker"
    capability = "analysis"
    tier = ToolTier.CORE
    capabilities = ["face_detection"]
    dependencies = []

    def __init__(self, face_count=0):
        self._face_count = face_count

    def get_status(self):
        return ToolStatus.AVAILABLE

    def execute(self, inputs):
        return ToolResult(
            success=True,
            data={"faces_detected": self._face_count, "faces": []},
        )


class _FakeSilenceCutter(BaseTool):
    name = "silence_cutter"
    capability = "video_post"
    tier = ToolTier.CORE
    capabilities = ["silence_detection"]
    dependencies = []

    def __init__(self, speech_segments=None):
        self._speech = speech_segments or []

    def execute(self, inputs):
        if inputs.get("mode") == "mark":
            return ToolResult(success=True, data={"speech_segments": self._speech})
        return ToolResult(success=True, data={})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_input_video(tmp_path: Path) -> Path:
    """A tiny placeholder file; the fake sub-tools never actually read it, but
    the segment tools' existence check requires the path to exist."""
    p = tmp_path / "src.mp4"
    p.write_bytes(b"fake-video")
    return p


@pytest.fixture
def fresh_registry(monkeypatch):
    """A fresh registry wired into handlers + the segment tool modules.

    The segment tool modules import `registry as tool_registry` at import time,
    so we patch the attribute on each module that references it. We pre-seed
    ``_discovered_packages`` so the segment tools' ``ensure_discovered()`` call
    is a no-op — otherwise it would walk the real ``tools/`` package and
    overwrite our fake sub-tools with the real (CV-dependent) ones.
    """
    reg = ToolRegistry()
    reg._discovered_packages.add("tools")  # noqa: SLF001 — prevent re-discovery
    monkeypatch.setattr(H, "registry", reg)
    # Patch the tool_registry symbol the segment tools use for cross-tool calls.
    import tools.analysis.segment_shots as ss_mod
    import tools.analysis.segment_filter as sf_mod
    import tools.analysis.segment_by_face as sbf_mod
    monkeypatch.setattr(ss_mod, "tool_registry", reg, raising=False)
    monkeypatch.setattr(sf_mod, "tool_registry", reg, raising=False)
    monkeypatch.setattr(sbf_mod, "tool_registry", reg, raising=False)
    return reg


# ===========================================================================
# segment_shots
# ===========================================================================

def test_segment_shots_happy_path(fresh_registry, tmp_path):
    scenes = [
        {"index": 0, "start_seconds": 0.0, "end_seconds": 5.0, "duration_seconds": 5.0},
        {"index": 1, "start_seconds": 5.0, "end_seconds": 12.0, "duration_seconds": 7.0},
    ]
    fresh_registry.register(_FakeSceneDetect(scenes=scenes))
    fresh_registry.register(_FakeVideoUnderstand(label="close-up"))
    fresh_registry.register(SegmentShots())

    src = _make_input_video(tmp_path)
    out = asyncio.run(H.segment_shots(str(src), str(tmp_path / "out"), enrich=False))

    assert out["success"] is True
    data = out["data"]
    assert data["shot_count"] == 2
    assert data["shots"][0]["start_seconds"] == 0.0
    assert data["shots"][0]["end_seconds"] == 5.0
    assert data["shots"][1]["id"] == 1
    # shot_size stays None because enrich=False
    assert all(s["shot_size"] is None for s in data["shots"])
    assert data["enriched"] is False


def test_segment_shots_enrich_attaches_label(fresh_registry, tmp_path):
    scenes = [{"index": 0, "start_seconds": 0.0, "end_seconds": 4.0, "duration_seconds": 4.0}]
    fresh_registry.register(_FakeSceneDetect(scenes=scenes))
    fresh_registry.register(_FakeVideoUnderstand(label="wide", desc="a landscape"))
    fresh_registry.register(SegmentShots())

    src = _make_input_video(tmp_path)
    out = asyncio.run(H.segment_shots(str(src), str(tmp_path / "out"), enrich=True))

    assert out["success"] is True
    shot = out["data"]["shots"][0]
    assert shot["shot_size"] == "wide"
    assert shot["description"] is not None
    assert out["data"]["enriched"] is True


def test_segment_shots_propagates_subtool_error(fresh_registry, tmp_path):
    fresh_registry.register(_FakeSceneDetect(fail=True))
    fresh_registry.register(SegmentShots())

    src = _make_input_video(tmp_path)
    out = asyncio.run(H.segment_shots(str(src), str(tmp_path / "out"), enrich=False))

    assert out["success"] is False
    assert "scene_detect failed" in (out["error"] or "")


def test_segment_shots_max_clips_truncates(fresh_registry, tmp_path):
    scenes = [
        {"index": i, "start_seconds": float(i), "end_seconds": float(i + 1), "duration_seconds": 1.0}
        for i in range(5)
    ]
    fresh_registry.register(_FakeSceneDetect(scenes=scenes))
    fresh_registry.register(SegmentShots())

    src = _make_input_video(tmp_path)
    out = asyncio.run(
        H.segment_shots(str(src), str(tmp_path / "out"), enrich=False, max_clips=2)
    )
    assert out["success"] is True
    assert out["data"]["shot_count"] == 2


def test_segment_shots_extract_clips(fresh_registry, tmp_path):
    scenes = [{"index": 0, "start_seconds": 0.0, "end_seconds": 3.0, "duration_seconds": 3.0}]
    fresh_registry.register(_FakeSceneDetect(scenes=scenes))
    trimmer = _FakeVideoTrimmer()
    fresh_registry.register(trimmer)
    fresh_registry.register(SegmentShots())

    src = _make_input_video(tmp_path)
    out = asyncio.run(
        H.segment_shots(str(src), str(tmp_path / "out"), enrich=False, extract_clips=True)
    )
    assert out["success"] is True
    assert out["data"]["extracted"] is True
    assert out["data"]["shots"][0]["clip_path"] is not None
    assert len(trimmer.calls) == 1
    assert trimmer.calls[0]["operation"] == "cut"


# ===========================================================================
# segment_filter
# ===========================================================================

def test_segment_filter_min_duration(fresh_registry, tmp_path):
    fresh_registry.register(SegmentFilter())
    src = _make_input_video(tmp_path)

    segments = [
        {"start_seconds": 0.0, "end_seconds": 1.0},   # too short
        {"start_seconds": 1.0, "end_seconds": 5.0},   # ok
        {"start_seconds": 5.0, "end_seconds": 6.0},   # too short
    ]
    out = asyncio.run(
        H.segment_filter(
            str(src), segments,
            {"min_duration_seconds": 2.0},
            str(tmp_path / "out"),
        )
    )
    assert out["success"] is True
    data = out["data"]
    assert data["matched_count"] == 1
    assert data["rejected_count"] == 2
    assert data["matched"][0]["start_seconds"] == 1.0
    # The rejected entries record which predicate failed + why.
    assert data["rejected"][0]["failed_predicates"] == ["min_duration_seconds"]
    assert "min_duration_seconds" in data["rejected"][0]["reasons"]


def test_segment_filter_shot_size_and_normalization(fresh_registry, tmp_path):
    fresh_registry.register(SegmentFilter())
    src = _make_input_video(tmp_path)

    segments = [
        {"start_seconds": 0.0, "end_seconds": 2.0, "shot_size": "close up"},  # synonym w/ space
        {"start_seconds": 2.0, "end_seconds": 4.0, "shot_size": "wide"},
        {"start_seconds": 4.0, "end_seconds": 6.0, "shot_size": "medium"},
        {"start_seconds": 6.0, "end_seconds": 8.0},  # no label -> fails
    ]
    out = asyncio.run(
        H.segment_filter(
            str(src), segments,
            {"shot_size": ["close-up", "medium"], "min_duration_seconds": 1.0},
            str(tmp_path / "out"),
        )
    )
    assert out["success"] is True
    data = out["data"]
    # close-up (normalized) + medium pass; wide fails shot_size; no-label fails.
    matched_starts = sorted(s["start_seconds"] for s in data["matched"])
    assert matched_starts == [0.0, 4.0]
    assert data["matched_count"] == 2


def test_segment_filter_requires_predicates(fresh_registry, tmp_path):
    fresh_registry.register(SegmentFilter())
    src = _make_input_video(tmp_path)
    out = asyncio.run(
        H.segment_filter(
            str(src),
            [{"start_seconds": 0.0, "end_seconds": 1.0}],
            {},
            str(tmp_path / "out"),
        )
    )
    assert out["success"] is False
    assert "predicates" in (out["error"] or "")


def test_segment_filter_requires_segments(fresh_registry, tmp_path):
    fresh_registry.register(SegmentFilter())
    src = _make_input_video(tmp_path)
    out = asyncio.run(
        H.segment_filter(str(src), [], {"min_duration_seconds": 1.0}, str(tmp_path / "out"))
    )
    assert out["success"] is False


def test_segment_filter_has_speech_uses_silence_cutter(fresh_registry, tmp_path):
    fresh_registry.register(_FakeSilenceCutter(speech_segments=[
        {"start_seconds": 1.0, "end_seconds": 4.0},
    ]))
    fresh_registry.register(SegmentFilter())
    src = _make_input_video(tmp_path)

    segments = [
        {"start_seconds": 0.0, "end_seconds": 0.5},   # no overlap with speech
        {"start_seconds": 2.0, "end_seconds": 3.0},   # inside speech window
    ]
    out = asyncio.run(
        H.segment_filter(
            str(src), segments, {"has_speech": True}, str(tmp_path / "out"),
            extract_clips=False,
        )
    )
    assert out["success"] is True
    data = out["data"]
    assert data["matched_count"] == 1
    assert data["matched"][0]["start_seconds"] == 2.0


# ===========================================================================
# segment_by_face — only the dependency-missing path (no insightface in CI).
# ===========================================================================

def test_segment_by_face_missing_dependency(fresh_registry, tmp_path):
    """Without insightface installed, the tool must report UNAVAILABLE and the
    handler surfaces a clear install hint rather than crashing."""
    import tools.analysis.segment_by_face as sbf_mod
    # Force the backend check to fail regardless of the host env.
    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setattr(sbf_mod.SegmentByFace, "_has_backend", lambda self: False)
    try:
        fresh_registry.register(sbf_mod.SegmentByFace())
        src = _make_input_video(tmp_path)
        out = asyncio.run(H.segment_by_face(str(src), str(tmp_path / "out")))
        assert out["success"] is False
        assert "insightface" in (out["error"] or "").lower()
    finally:
        monkeypatch_obj.undo()


# ===========================================================================
# Generic dispatch
# ===========================================================================

def test_run_segment_tool_unknown_name(fresh_registry):
    with pytest.raises(ValueError):
        asyncio.run(H._run_segment_tool("does_not_exist", {}, ctx=None))


def test_segment_tool_inputs_drops_none_defaults():
    """None kwargs must not override documented defaults; explicit values do."""
    inputs = H._segment_tool_inputs(
        "segment_shots",
        input_path="x.mp4",
        output_dir=None,        # should be dropped
        enrich=False,           # should override the True default
        max_clips=None,         # should be dropped (default 50 stays)
    )
    assert inputs["input_path"] == "x.mp4"
    assert "output_dir" not in inputs          # None dropped, no default for it
    assert inputs["enrich"] is False           # explicit override
    assert inputs["max_clips"] == 50           # default preserved
    assert inputs["method"] == "content"       # default preserved


# ===========================================================================
# segment_by_face device selection (no GPU required to test the resolution)
# ===========================================================================

def test_resolve_ctx_id_cpu_force():
    from tools.analysis.segment_by_face import SegmentByFace
    assert SegmentByFace._resolve_ctx_id("cpu") == -1
    assert SegmentByFace._resolve_ctx_id("CPU") == -1


def test_resolve_ctx_id_gpu_force():
    from tools.analysis.segment_by_face import SegmentByFace
    assert SegmentByFace._resolve_ctx_id("gpu") == 0


def test_resolve_ctx_id_auto_without_gpu_returns_cpu():
    """On a machine without onnxruntime-gpu/CUDA, 'auto' must fall back to CPU
    (ctx_id=-1). This holds on the CI host which has no GPU."""
    from tools.analysis.segment_by_face import SegmentByFace
    try:
        import onnxruntime as ort  # noqa: F401
        has_cuda = any("CUDA" in p for p in ort.get_available_providers())
    except ImportError:
        has_cuda = False
    if has_cuda:
        pytest.skip("this host has CUDA; auto resolves to GPU here")
    assert SegmentByFace._resolve_ctx_id("auto") == -1


def test_segment_by_face_device_param_threaded_through(fresh_registry, tmp_path):
    """The handler must thread the `device` param into the tool inputs."""
    import tools.analysis.segment_by_face as sbf_mod
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sbf_mod.SegmentByFace, "_has_backend", lambda self: False)
    captured = {}
    real_execute = sbf_mod.SegmentByFace.execute

    def spy_execute(self, inputs):
        captured["device"] = inputs.get("device")
        return real_execute(self, inputs)

    try:
        monkey.setattr(sbf_mod.SegmentByFace, "execute", spy_execute)
        fresh_registry.register(sbf_mod.SegmentByFace())
        src = _make_input_video(tmp_path)
        asyncio.run(H.segment_by_face(str(src), str(tmp_path / "out"), device="gpu"))
        assert captured["device"] == "gpu"
    finally:
        monkey.undo()
