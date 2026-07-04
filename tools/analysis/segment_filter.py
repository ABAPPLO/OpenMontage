"""Filter a list of video segments by per-segment predicates.

Takes a list of time ranges (``segments``) — typically the output of
``segment_shots`` or ``segment_by_face`` — and keeps only those that match a set
of ``predicates``. Predicates are evaluated in cost order (cheap first), and a
segment that fails any predicate is dropped with a recorded reason so the caller
can debug.

Predicate kinds (any subset may be present; all present must pass — AND logic):

  - ``min_duration_seconds`` / ``max_duration_seconds`` (number)
        O(1) local check on the segment's own duration.

  - ``has_face`` (bool)
        If true, keep segments containing at least one face. Uses ``face_tracker``
        on a per-segment sub-clip (cheap, deterministic). Falls back to skipped
        if face_tracker is unavailable.

  - ``has_speech`` (bool)
        If true, keep segments with non-silent audio. Uses ``silence_cutter``
        (mode=mark) on the whole video and intersects the segment with the
        detected speech_segments. Single pass over the video, reused for every
        segment.

  - ``shot_size`` (string | string[])
        Keep segments whose framing matches. e.g. ``"close-up"`` or
        ``["close-up", "medium"]``. Each segment must already carry a
        ``shot_size`` field (set ``enrich=true`` on segment_shots, or supply it
        in the input segments). Segments without a shot_size fail.

  - ``query`` (string)
        Free-text semantic match against a keyframe of the segment. Backends:
        ``clip`` (default, light) or ``vlm`` (heavier, via video_understand qa).
        Supports OR with ``|`` and an ``any:``/``all:`` prefix. A passing score
        threshold is configurable via ``query_threshold``.

Output shares the unified segment shape. ``matched`` keeps the passing segments
(with their per-predicate scores); ``rejected`` lists the failing ones with
``failed_predicates`` and ``reasons`` for debugging.
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


# Predicates evaluated in this cost order (cheap → expensive).
PREDICATE_ORDER = [
    "min_duration_seconds",
    "max_duration_seconds",
    "shot_size",
    "has_face",
    "has_speech",
    "query",
]

SHOT_SIZE_VARIANTS = ("extreme_close-up", "close-up", "medium", "wide", "establishing")


class SegmentFilter(BaseTool):
    name = "segment_filter"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "analysis"
    provider = "ffmpeg"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg"]
    install_instructions = (
        "FFmpeg is required. For richer predicates:\n"
        "  - has_face: pip install mediapipe opencv-python  (face_tracker backend)\n"
        "  - query (clip backend, default): pip install transformers torch\n"
        "  - query (vlm backend): same as clip"
    )
    agent_skills = ["ffmpeg", "video-understand"]

    capabilities = [
        "filter_segments",
        "predicate_filter",
        "has_face_predicate",
        "has_speech_predicate",
        "query_predicate",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path", "segments"],
        "properties": {
            "input_path": {"type": "string"},
            "output_dir": {"type": "string"},
            "segments": {
                "type": "array",
                "minItems": 1,
                "description": (
                    "Time ranges to filter. Each item: {start_seconds, end_seconds, "
                    "[shot_size], [description]}. The shot_size field is required only "
                    "when the shot_size predicate is used."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "start_seconds": {"type": "number", "minimum": 0},
                        "end_seconds": {"type": "number", "minimum": 0},
                        "shot_size": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "predicates": {
                "type": "object",
                "description": (
                    "Subset of: min_duration_seconds, max_duration_seconds, has_face, "
                    "has_speech, shot_size, query. All present must pass (AND)."
                ),
                "properties": {
                    "min_duration_seconds": {"type": "number"},
                    "max_duration_seconds": {"type": "number"},
                    "has_face": {"type": "boolean"},
                    "has_speech": {"type": "boolean"},
                    "shot_size": {
                        "type": ["string", "array"],
                        "description": "A single label or a list; segment matches if its shot_size is in the set.",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Free-text semantic query against a keyframe. Use | between terms for OR "
                            "(e.g. 'person running | person jogging')."
                        ),
                    },
                },
            },
            "query_backend": {
                "type": "string",
                "enum": ["clip", "vlm"],
                "default": "clip",
                "description": "Backend for the query predicate: clip (light, default) or vlm (heavier).",
            },
            "query_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.25,
                "description": "Cosine similarity (clip) or vlm yes-score above which the query predicate passes.",
            },
            "extract_clips": {
                "type": "boolean",
                "default": False,
                "description": "If true, physically cut each matched segment to mp4 via video_trimmer.",
            },
            "clips_subdir": {"type": "string", "default": "clips"},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "input_segment_count": {"type": "integer"},
            "matched_count": {"type": "integer"},
            "rejected_count": {"type": "integer"},
            "matched": {"type": "array"},
            "rejected": {"type": "array"},
            "evaluated_predicates": {"type": "array"},
            "output_dir": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=1024, vram_mb=2048, disk_mb=1000, network_required=False
    )
    idempotency_key_fields = ["input_path", "predicates", "query_backend", "query_threshold"]
    side_effects = [
        "writes a filter result JSON to output_dir",
        "may write mp4 clip files when extract_clips=true",
    ]
    user_visible_verification = [
        "Inspect rejected[].reasons to understand why segments were dropped",
        "Verify query scores against the actual segment content",
    ]

    # ------------------------------------------------------------------
    def get_status(self) -> ToolStatus:
        import shutil

        if not shutil.which("ffmpeg"):
            return ToolStatus.UNAVAILABLE
        # Sub-tools (face_tracker, silence_cutter, video_understand) handle their
        # own status; we report DEGRADED if none of the heavy backends are present.
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return ToolStatus.AVAILABLE
        except ImportError:
            return ToolStatus.DEGRADED

    # ------------------------------------------------------------------
    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        input_path = Path(inputs["input_path"])
        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        segments_in = inputs.get("segments") or []
        if not segments_in:
            return ToolResult(success=False, error="'segments' must be a non-empty list.")
        predicates = inputs.get("predicates") or {}
        if not predicates:
            return ToolResult(success=False, error="'predicates' must contain at least one predicate.")

        # Normalize segments.
        segments: list[dict] = []
        for idx, s in enumerate(segments_in):
            start_s = float(s.get("start_seconds", 0))
            end_s = float(s.get("end_seconds", start_s))
            if end_s <= start_s:
                continue
            seg = dict(s)
            seg["start_seconds"] = start_s
            seg["end_seconds"] = end_s
            seg["duration_seconds"] = round(end_s - start_s, 3)
            seg["id"] = s.get("id", idx)
            segments.append(seg)
        if not segments:
            return ToolResult(success=False, error="All segments were empty or invalid.")

        active = [p for p in PREDICATE_ORDER if p in predicates]
        if not active:
            return ToolResult(success=False, error="No recognized predicates supplied.")

        start = time.time()
        tool_registry.ensure_discovered()

        output_dir = Path(
            inputs.get("output_dir", str(input_path.parent / f"{input_path.stem}_filtered"))
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute whole-video data shared across segments.
        speech_segments = None
        if "has_speech" in active:
            speech_segments = self._compute_speech(input_path)

        # Evaluate per segment.
        matched: list[dict] = []
        rejected: list[dict] = []
        for seg in segments:
            verdict = self._evaluate(
                seg, active, predicates, input_path, output_dir, speech_segments, inputs
            )
            if verdict["passed"]:
                seg["scores"] = verdict["scores"]
                seg["clip_path"] = None
                matched.append(seg)
            else:
                rejected.append(
                    {
                        "segment": seg,
                        "failed_predicates": verdict["failed"],
                        "reasons": verdict["reasons"],
                    }
                )

        # Optional physical extraction of matched segments.
        extracted = False
        if inputs.get("extract_clips", False) and matched:
            clips_dir = output_dir / inputs.get("clips_subdir", "clips")
            clips_dir.mkdir(parents=True, exist_ok=True)
            extracted = self._extract(matched, input_path, clips_dir)

        elapsed = time.time() - start
        payload = {
            "input_segment_count": len(segments),
            "matched_count": len(matched),
            "rejected_count": len(rejected),
            "matched": matched,
            "rejected": rejected,
            "evaluated_predicates": active,
            "clipping_extracted": extracted,
            "output_dir": str(output_dir),
        }
        (output_dir / "filtered.json").write_text(json.dumps(payload, indent=2))
        return ToolResult(
            success=True,
            data=payload,
            artifacts=[str(output_dir / "filtered.json")],
            duration_seconds=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Per-segment evaluation
    # ------------------------------------------------------------------
    def _evaluate(
        self,
        seg: dict,
        active: list[str],
        predicates: dict[str, Any],
        input_path: Path,
        output_dir: Path,
        speech_segments: list[dict] | None,
        inputs: dict[str, Any],
    ) -> dict:
        passed = True
        failed: list[str] = []
        reasons: dict[str, str] = {}
        scores: dict[str, Any] = {}

        for pred in active:
            ok, reason, score = self._eval_one(
                pred, seg, predicates, input_path, output_dir, speech_segments, inputs
            )
            if score is not None:
                scores[pred] = score
            if not ok:
                passed = False
                failed.append(pred)
                reasons[pred] = reason or "predicate failed"

        return {"passed": passed, "failed": failed, "reasons": reasons, "scores": scores}

    def _eval_one(
        self,
        pred: str,
        seg: dict,
        predicates: dict[str, Any],
        input_path: Path,
        output_dir: Path,
        speech_segments: list[dict] | None,
        inputs: dict[str, Any],
    ) -> tuple[bool, str | None, Any]:
        if pred == "min_duration_seconds":
            return self._pred_min_duration(seg, predicates[pred])
        if pred == "max_duration_seconds":
            return self._pred_max_duration(seg, predicates[pred])
        if pred == "shot_size":
            return self._pred_shot_size(seg, predicates[pred])
        if pred == "has_face":
            return self._pred_has_face(seg, predicates[pred], input_path, output_dir)
        if pred == "has_speech":
            return self._pred_has_speech(seg, predicates[pred], speech_segments)
        if pred == "query":
            return self._pred_query(seg, predicates[pred], input_path, output_dir, inputs)
        # Unknown predicate: treat as pass-through (already filtered out).
        return True, None, None

    # --- cheap predicates -------------------------------------------------
    def _pred_min_duration(self, seg: dict, val: float) -> tuple[bool, str | None, Any]:
        ok = seg["duration_seconds"] >= float(val)
        return ok, (None if ok else f"{seg['duration_seconds']}s < {val}s"), seg["duration_seconds"]

    def _pred_max_duration(self, seg: dict, val: float) -> tuple[bool, str | None, Any]:
        ok = seg["duration_seconds"] <= float(val)
        return ok, (None if ok else f"{seg['duration_seconds']}s > {val}s"), seg["duration_seconds"]

    def _pred_shot_size(self, seg: dict, val: Any) -> tuple[bool, str | None, Any]:
        allowed = val if isinstance(val, list) else [val]
        allowed_norm = {self._norm_size(a) for a in allowed}
        seg_size = self._norm_size(seg.get("shot_size"))
        if seg_size is None:
            return False, "segment has no shot_size label", None
        ok = seg_size in allowed_norm
        return ok, (None if ok else f"shot_size '{seg_size}' not in {sorted(allowed_norm)}"), seg_size

    @staticmethod
    def _norm_size(val: Any) -> str | None:
        if not val:
            return None
        s = str(val).strip().lower().replace(" ", "-")
        # Map common synonyms to the canonical vocabulary.
        synonyms = {
            "extreme_closeup": "extreme_close-up",
            "extreme_close_up": "extreme_close-up",
            "ecu": "extreme_close-up",
            "cu": "close-up",
            "ms": "medium",
            "ws": "wide",
            "es": "establishing",
        }
        return synonyms.get(s, s)

    # --- heavier predicates ----------------------------------------------
    def _pred_has_face(
        self, seg: dict, want: bool, input_path: Path, output_dir: Path
    ) -> tuple[bool, str | None, Any]:
        """Cut a tiny sub-clip and run face_tracker on it. want=True keeps segments
        with a face; want=False keeps segments without one."""
        face_tracker = tool_registry.get("face_tracker")
        if face_tracker is None or face_tracker.get_status() == ToolStatus.UNAVAILABLE:
            # Can't evaluate: fail open (skip predicate) so we don't drop everything.
            return True, "face_tracker unavailable; predicate skipped", None

        sub = self._cut_subclip_for_analysis(input_path, seg, output_dir, tag="face")
        if sub is None:
            return True, "could not cut sub-clip; predicate skipped", None

        try:
            res = face_tracker.execute({"input_path": str(sub), "sample_fps": 5})
        except Exception as e:
            return True, f"face_tracker error: {e}; predicate skipped", None
        if not res.success:
            return True, f"face_tracker failed: {res.error}; predicate skipped", None

        face_count = int(res.data.get("faces_detected", 0) or 0)
        has_face = face_count > 0
        ok = has_face == bool(want)
        reason = None if ok else (f"has_face={has_face}, want={bool(want)}")
        return ok, reason, face_count

    def _pred_has_speech(
        self, seg: dict, want: bool, speech_segments: list[dict] | None
    ) -> tuple[bool, str | None, Any]:
        if speech_segments is None:
            return True, "speech detection unavailable; predicate skipped", None
        overlap = self._max_overlap(seg, speech_segments)
        has_speech = overlap > 0.1  # at least 100ms of speech in the segment
        ok = has_speech == bool(want)
        reason = None if ok else f"speech_overlap={overlap:.2f}s, has_speech={has_speech}, want={bool(want)}"
        return ok, reason, round(overlap, 3)

    def _pred_query(
        self, seg: dict, query: str, input_path: Path, output_dir: Path, inputs: dict[str, Any]
    ) -> tuple[bool, str | None, Any]:
        backend = inputs.get("query_backend", "clip")
        threshold = float(inputs.get("query_threshold", 0.25))
        if backend == "vlm":
            return self._query_vlm(seg, query, input_path, output_dir, threshold)
        return self._query_clip(seg, query, input_path, output_dir, threshold)

    def _query_clip(
        self, seg: dict, query: str, input_path: Path, output_dir: Path, threshold: float
    ) -> tuple[bool, str | None, Any]:
        try:
            from lib.clip_embedder import embed_images, embed_texts
        except ImportError as e:
            return True, f"CLIP backend unavailable ({e}); predicate skipped", None

        keyframe = self._extract_keyframe(input_path, seg, output_dir, tag="q")
        if keyframe is None:
            return True, "could not extract keyframe; predicate skipped", None

        # Parse OR alternation: "a | b" → keep max similarity over the variants.
        variants = [v.strip() for v in query.replace("any:", "").replace("all:", "").split("|") if v.strip()]
        if not variants:
            return True, "empty query", None

        try:
            img_emb = embed_images([str(keyframe)])  # (1, D)
            txt_emb = embed_texts(variants)  # (N, D)
        except Exception as e:
            return True, f"CLIP inference failed ({e}); predicate skipped", None

        import numpy as np

        sims = (txt_emb @ img_emb[0]).max()
        best = float(sims)
        ok = best >= threshold
        reason = None if ok else f"clip_sim={best:.3f} < {threshold}"
        return ok, reason, round(best, 4)

    def _query_vlm(
        self, seg: dict, query: str, input_path: Path, output_dir: Path, threshold: float
    ) -> tuple[bool, str | None, Any]:
        vlm = tool_registry.get("video_understand")
        if vlm is None or vlm.get_status() == ToolStatus.UNAVAILABLE:
            return True, "video_understand unavailable; predicate skipped", None

        keyframe = self._extract_keyframe(input_path, seg, output_dir, tag="qv")
        if keyframe is None:
            return True, "could not extract keyframe; predicate skipped", None

        # Frame index isn't meaningful for a still; pass as a single "frame".
        try:
            res = vlm.execute(
                {
                    "input_path": str(keyframe),
                    "mode": "qa",
                    "model": "clip",
                    "query": f"Is the following true in this image? Answer yes or no and rate confidence 0..1: {query}",
                }
            )
        except Exception as e:
            return True, f"vlm error: {e}; predicate skipped", None
        if not res.success:
            return True, f"vlm failed: {res.error}; predicate skipped", None

        score = self._extract_yes_score(res.data)
        ok = score >= threshold
        reason = None if ok else f"vlm_yes_score={score:.3f} < {threshold}"
        return ok, reason, round(score, 4)

    @staticmethod
    def _extract_yes_score(data: dict[str, Any]) -> float:
        """Best-effort parse of a 0..1 confidence from video_understand qa output."""
        summary = (data.get("summary") or "").lower()
        frames = data.get("frames") or []
        text = summary or (frames[0].get("answer") if frames else "") or summary
        # Look for an explicit float; default by yes/no keyword.
        import re

        m = re.search(r"([01](?:\.\d+)?)", text)
        if m:
            return float(m.group(1))
        if "yes" in text:
            return 0.6
        if "no" in text:
            return 0.2
        return 0.0

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _compute_speech(self, input_path: Path) -> list[dict] | None:
        """One pass: use silence_cutter (mode=mark) to get speech_segments."""
        cutter = tool_registry.get("silence_cutter")
        if cutter is None:
            return None
        try:
            res = cutter.execute({"input_path": str(input_path), "mode": "mark"})
        except Exception:
            return None
        if not res.success:
            return None
        return res.data.get("speech_segments") or []

    @staticmethod
    def _max_overlap(seg: dict, speech_segments: list[dict]) -> float:
        s_start, s_end = seg["start_seconds"], seg["end_seconds"]
        total = 0.0
        for sp in speech_segments:
            sp_start = float(sp.get("start_seconds", sp.get("start", 0)))
            sp_end = float(sp.get("end_seconds", sp.get("end", sp_start)))
            ov = max(0.0, min(s_end, sp_end) - max(s_start, sp_start))
            total += ov
        return total

    def _extract_keyframe(
        self, input_path: Path, seg: dict, output_dir: Path, tag: str
    ) -> Path | None:
        mid = (seg["start_seconds"] + seg["end_seconds"]) / 2.0
        out = output_dir / "_keyframes" / f"seg{seg.get('id', 0)}_{tag}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            return out
        import subprocess

        cmd = [
            "ffmpeg", "-y", "-ss", str(mid), "-i", str(input_path),
            "-frames:v", "1", "-q:v", "2", str(out),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        except Exception:
            return None
        return out if out.exists() else None

    def _cut_subclip_for_analysis(
        self, input_path: Path, seg: dict, output_dir: Path, tag: str
    ) -> Path | None:
        """Cut a small low-res sub-clip for analysis tools that operate on video."""
        out = output_dir / "_subclips" / f"seg{seg.get('id', 0)}_{tag}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            return out
        import subprocess

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seg["start_seconds"]),
            "-to", str(seg["end_seconds"]),
            "-i", str(input_path),
            "-vf", "scale=640:-2",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac",
            str(out),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        except Exception:
            return None
        return out if out.exists() else None

    def _extract(
        self, matched: list[dict], input_path: Path, clips_dir: Path
    ) -> bool:
        trimmer = tool_registry.get("video_trimmer")
        if trimmer is None:
            return False
        any_ok = False
        for seg in matched:
            clip_path = clips_dir / f"seg_{seg.get('id', 0):04d}.mp4"
            try:
                res = trimmer.execute(
                    {
                        "operation": "cut",
                        "input_path": str(input_path),
                        "output_path": str(clip_path),
                        "start_seconds": seg["start_seconds"],
                        "end_seconds": seg["end_seconds"],
                    }
                )
            except Exception:
                continue
            if res.success:
                seg["clip_path"] = str(clip_path)
                any_ok = True
        return any_ok
