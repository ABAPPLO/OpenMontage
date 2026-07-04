"""Shot segmentation tool — splits a video into continuous-shot segments.

A thin orchestration layer over :mod:`scene_detect`: it turns the raw scene
boundary list into first-class *shot segments* (each a continuous run of frames
from one cut to the next), optionally enriches each shot with a size label
(close-up / medium / wide / establishing) via :mod:`video_understand`, and
optionally physically extracts each shot to its own mp4 via :mod:`video_trimmer`.

Why this exists separately from ``scene_detect``:
  - ``scene_detect`` returns *boundaries* (a list of [start, end] tuples) and
    nothing else. The agent caller must then wrap them, pick keyframes, label
    shot sizes, and cut files — repeating the same glue every time.
  - This tool produces ready-to-consume *segment objects* with stable ids,
    durations, optional semantic labels, and optional clip paths, in the unified
    ``{segments: [...]}`` shape shared by ``segment_by_face`` and
    ``segment_filter`` so the three tools compose.

Long videos: keep ``enrich=True`` but cap ``max_clips`` to avoid running a VLM on
hundreds of shots. For pure boundary detection, set ``enrich=False``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.tool_registry import registry as tool_registry

# Canonical shot-size vocabulary. Mirrors the framing terms used across the
# project (lib/shot_prompt_builder.py, video_understand classify mode).
SHOT_SIZE_LABELS = ("extreme_close-up", "close-up", "medium", "wide", "establishing")


class SegmentShots(BaseTool):
    name = "segment_shots"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "analysis"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg"]
    install_instructions = (
        "FFmpeg is required. For better boundary detection install PySceneDetect:\n"
        "  pip install scenedetect[opencv]\n"
        "For shot-size labeling (enrich), install a vision-language backend:\n"
        "  pip install transformers torch"
    )
    agent_skills = ["ffmpeg", "video-understand"]

    capabilities = [
        "segment_shots",
        "detect_shot_boundaries",
        "label_shot_size",
        "extract_shot_clips",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string"},
            "output_dir": {
                "type": "string",
                "description": "Directory for the segments JSON and (if extracted) clips.",
            },
            "method": {
                "type": "string",
                "enum": ["content", "threshold", "adaptive"],
                "default": "content",
                "description": "Passed through to scene_detect.",
            },
            "threshold": {
                "type": "number",
                "description": "Detection threshold (method-dependent). Default 27 for content.",
            },
            "min_scene_length_seconds": {
                "type": "number",
                "minimum": 0.1,
                "default": 1.0,
                "description": "Minimum shot duration; shorter runs are merged into the previous shot.",
            },
            "enrich": {
                "type": "boolean",
                "default": True,
                "description": (
                    "If true, sample one keyframe per shot and run video_understand "
                    "(mode=classify) to attach a shot_size label and a one-line "
                    "description. Skipped silently if video_understand is unavailable."
                ),
            },
            "extract_clips": {
                "type": "boolean",
                "default": False,
                "description": "If true, physically cut each shot to its own mp4 via video_trimmer.",
            },
            "max_clips": {
                "type": "integer",
                "minimum": 1,
                "default": 50,
                "description": "Hard cap on the number of shots. Extra shots beyond this are dropped.",
            },
            "clips_subdir": {
                "type": "string",
                "default": "clips",
                "description": "Subdirectory under output_dir for extracted mp4s.",
            },
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "method": {"type": "string"},
            "shot_count": {"type": "integer"},
            "shots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "start_seconds": {"type": "number"},
                        "end_seconds": {"type": "number"},
                        "duration_seconds": {"type": "number"},
                        "shot_size": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                        "clip_path": {"type": ["string", "null"]},
                    },
                },
            },
            "enriched": {"type": "boolean"},
            "extracted": {"type": "boolean"},
            "output_dir": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=1024, vram_mb=2048, disk_mb=2000, network_required=False
    )
    idempotency_key_fields = ["input_path", "method", "threshold", "min_scene_length_seconds"]
    side_effects = [
        "writes a segments JSON file to output_dir",
        "may write mp4 clip files to output_dir/clips when extract_clips=true",
    ]
    user_visible_verification = [
        "Spot-check shot boundaries against the video",
        "Verify shot_size labels match the framing",
    ]

    # ------------------------------------------------------------------
    # Status / availability
    # ------------------------------------------------------------------
    def get_status(self) -> ToolStatus:
        """AVAILABLE if ffmpeg is on PATH. Enrichment is best-effort, not gating."""
        if not self._has_ffmpeg():
            return ToolStatus.UNAVAILABLE
        if not self._has_vlm():
            # Shot boundaries still work; only labels are skipped.
            return ToolStatus.DEGRADED
        return ToolStatus.AVAILABLE

    def _has_ffmpeg(self) -> bool:
        import shutil

        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    def _has_vlm(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        input_path = Path(inputs["input_path"])
        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        if not self._has_ffmpeg():
            return ToolResult(
                success=False,
                error="ffmpeg/ffprobe not found on PATH. Install FFmpeg to use segment_shots.",
            )

        start = time.time()
        tool_registry.ensure_discovered()

        output_dir = Path(inputs.get("output_dir", str(input_path.parent / f"{input_path.stem}_shots")))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Boundaries via scene_detect.
        shots = self._detect_shots(inputs, input_path)
        if isinstance(shots, str):  # error path
            return ToolResult(success=False, error=shots)

        max_clips = int(inputs.get("max_clips", 50))
        if len(shots) > max_clips:
            shots = shots[:max_clips]

        # 2) Optional enrichment (shot_size + description).
        enrich = inputs.get("enrich", True)
        enriched = False
        if enrich and shots:
            enriched = self._enrich(shots, input_path, output_dir)

        # 3) Optional physical extraction.
        extracted = False
        if inputs.get("extract_clips", False) and shots:
            clips_subdir = Path(output_dir) / inputs.get("clips_subdir", "clips")
            clips_subdir.mkdir(parents=True, exist_ok=True)
            extracted = self._extract_clips(shots, input_path, clips_subdir)

        elapsed = time.time() - start

        manifest_path = output_dir / "shots.json"
        payload = {
            "method": "pyscenedetect" if self._has_pyscenedetect() else "ffmpeg-scene",
            "shot_count": len(shots),
            "shots": shots,
            "enriched": enriched,
            "extracted": extracted,
            "output_dir": str(output_dir),
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return ToolResult(
            success=True,
            data=payload,
            artifacts=[str(manifest_path)],
            duration_seconds=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Step 1: boundaries
    # ------------------------------------------------------------------
    def _detect_shots(self, inputs: dict[str, Any], input_path: Path) -> list[dict] | str:
        scene_detect = tool_registry.get("scene_detect")
        if scene_detect is None:
            return "scene_detect tool not found in registry."

        sd_inputs: dict[str, Any] = {
            "input_path": str(input_path),
            "method": inputs.get("method", "content"),
        }
        if "threshold" in inputs:
            sd_inputs["threshold"] = inputs["threshold"]
        sd_inputs["min_scene_length_seconds"] = inputs.get("min_scene_length_seconds", 1.0)

        sd_result = scene_detect.execute(sd_inputs)
        if not sd_result.success:
            return f"scene_detect failed: {sd_result.error}"

        raw_scenes = sd_result.data.get("scenes", [])
        shots: list[dict] = []
        for idx, sc in enumerate(raw_scenes):
            start_s = float(sc["start_seconds"])
            end_s = float(sc["end_seconds"])
            shots.append(
                {
                    "id": idx,
                    "start_seconds": round(start_s, 3),
                    "end_seconds": round(end_s, 3),
                    "duration_seconds": round(end_s - start_s, 3),
                    "shot_size": None,
                    "description": None,
                    "clip_path": None,
                }
            )
        return shots

    # ------------------------------------------------------------------
    # Step 2: enrichment via video_understand classify
    # ------------------------------------------------------------------
    def _enrich(self, shots: list[dict], input_path: Path, output_dir: Path) -> bool:
        vlm = tool_registry.get("video_understand")
        if vlm is None or vlm.get_status() == ToolStatus.UNAVAILABLE:
            return False

        for shot in shots:
            mid = (shot["start_seconds"] + shot["end_seconds"]) / 2.0
            try:
                res = vlm.execute(
                    {
                        "input_path": str(input_path),
                        "mode": "classify",
                        "model": "clip",
                        "frame_indices": None,
                        "max_frames": 1,
                        "query": (
                            "Classify the camera framing of this shot as one of: "
                            "extreme_close-up, close-up, medium, wide, establishing. "
                            "Also describe the shot in one short sentence."
                        ),
                    }
                )
            except Exception:
                continue
            if not res.success:
                continue
            label, desc = self._parse_classify(res.data)
            if label:
                shot["shot_size"] = label
            if desc:
                shot["description"] = desc
        return True

    def _parse_classify(self, data: dict[str, Any]) -> tuple[str | None, str | None]:
        """Best-effort extraction of a shot-size label and a description from
        video_understand's free-form classify output. Normalizes the label to the
        canonical vocabulary; anything unrecognized becomes None (left for the
        caller to see the raw description)."""
        summary = (data.get("summary") or "").strip()
        frames = data.get("frames") or []
        text = summary
        if not text and frames:
            text = (frames[0].get("description") or frames[0].get("label") or "").strip()

        label: str | None = None
        low = text.lower()
        for canonical in SHOT_SIZE_LABELS:
            token = canonical.replace("-", " ")
            if canonical in low or token in low:
                label = canonical
                break
        return label, (text or None)

    # ------------------------------------------------------------------
    # Step 3: physical extraction via video_trimmer
    # ------------------------------------------------------------------
    def _extract_clips(
        self, shots: list[dict], input_path: Path, clips_dir: Path
    ) -> bool:
        trimmer = tool_registry.get("video_trimmer")
        if trimmer is None:
            return False

        any_ok = False
        for shot in shots:
            clip_path = clips_dir / f"shot_{shot['id']:04d}.mp4"
            try:
                res = trimmer.execute(
                    {
                        "operation": "cut",
                        "input_path": str(input_path),
                        "output_path": str(clip_path),
                        "start_seconds": shot["start_seconds"],
                        "end_seconds": shot["end_seconds"],
                    }
                )
            except Exception:
                continue
            if res.success:
                shot["clip_path"] = str(clip_path)
                any_ok = True
        return any_ok

    # ------------------------------------------------------------------
    def _has_pyscenedetect(self) -> bool:
        try:
            import scenedetect  # noqa: F401

            return True
        except ImportError:
            return False
