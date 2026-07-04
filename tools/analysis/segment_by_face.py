"""Identity-based face segmentation — split a video by WHO appears, not where.

Unlike :mod:`face_tracker` (which detects one face per frame with no identity),
this tool performs true face *recognition* across the whole video:

  1. Sample frames at ``sample_fps`` via ffmpeg.
  2. Run InsightFace (ArcFace, buffalo_l) on each frame → detect every face and
     embed each into a 512-d identity space.
  3. Cluster all embeddings with agglomerative clustering on cosine distance
     (``distance_threshold`` ≈ 0.42 is the ArcFace convention for "same person").
  4. Each cluster is one *identity*. Per identity, merge temporally contiguous
     frame hits (within ``max_gap_seconds``) into *segments*.
  5. Save one representative face thumbnail per identity (highest-det-score
     crop) and, optionally, physically cut each segment to mp4.

This is the right tool for "give me all the parts where person X is on screen",
"split this interview into per-speaker sections", or "find every appearance of
this face". For single-frame detection without identity, use ``face_tracker``.

Dependencies (lazy-imported, not in requirements.txt):
  - insightface + onnxruntime  — face detection + ArcFace embeddings
  - scikit-learn               — agglomerative clustering
  - numpy, Pillow              — already in requirements.txt
The buffalo_l model (~250MB) downloads to ~/.insightface on first run.

Output shares the unified ``{segments:[...]}`` shape with segment_shots and
segment_filter so the three tools compose.
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


class SegmentByFace(BaseTool):
    name = "segment_by_face"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "analysis"
    provider = "insightface"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["cmd:ffmpeg", "python:insightface", "python:sklearn"]
    install_instructions = (
        "Install the face-recognition backend and clustering library:\n"
        "  pip install insightface onnxruntime scikit-learn\n"
        "The buffalo_l model (~250MB) auto-downloads to ~/.insightface on first run. "
        "FFmpeg is also required (https://ffmpeg.org)."
    )
    agent_skills = ["ffmpeg"]

    capabilities = [
        "segment_by_face",
        "face_recognition",
        "face_clustering",
        "identity_tracking",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string"},
            "output_dir": {
                "type": "string",
                "description": "Directory for identity JSON, face thumbnails, and (if extracted) clips.",
            },
            "sample_fps": {
                "type": "number",
                "minimum": 0.1,
                "default": 2.0,
                "description": (
                    "Frames per second to sample. Lower = faster, less precise timing. "
                    "For videos > 10 min, 1.0 is recommended."
                ),
            },
            "cluster_threshold": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 1.0,
                "default": 0.42,
                "description": (
                    "Cosine distance below which two face embeddings are considered the same "
                    "identity. 0.42 is the ArcFace convention; lower = stricter (more identities), "
                    "higher = looser (fewer identities)."
                ),
            },
            "min_face_size": {
                "type": "integer",
                "minimum": 8,
                "default": 48,
                "description": "Minimum face size in pixels; smaller faces are ignored.",
            },
            "max_gap_seconds": {
                "type": "number",
                "minimum": 0.0,
                "default": 2.0,
                "description": (
                    "When merging same-identity frame hits into segments, hits separated by more "
                    "than this gap start a new segment. accounts for brief face turns/occlusions."
                ),
            },
            "min_track_seconds": {
                "type": "number",
                "minimum": 0.0,
                "default": 1.0,
                "description": "Identities whose total on-screen time is below this are dropped as noise.",
            },
            "extract_clips": {
                "type": "boolean",
                "default": False,
                "description": "If true, physically cut each segment to mp4 via video_trimmer.",
            },
            "clips_subdir": {
                "type": "string",
                "default": "clips",
            },
            "max_identities": {
                "type": "integer",
                "minimum": 1,
                "default": 50,
                "description": "Hard cap on identities kept (by total screen time).",
            },
            "device": {
                "type": "string",
                "enum": ["auto", "cpu", "gpu"],
                "default": "auto",
                "description": (
                    "Compute device for InsightFace inference. 'cpu' = force CPU "
                    "(default fallback, works everywhere, slower). 'gpu' = force GPU "
                    "(requires onnxruntime-gpu + CUDA). 'auto' = use GPU if available "
                    "else CPU. No third-party API is ever called — inference is local."
                ),
            },
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "method": {"type": "string"},
            "identities_count": {"type": "integer"},
            "frames_analyzed": {"type": "integer"},
            "identities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string"},
                        "total_duration_seconds": {"type": "number"},
                        "segment_count": {"type": "integer"},
                        "segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_seconds": {"type": "number"},
                                    "end_seconds": {"type": "number"},
                                    "duration_seconds": {"type": "number"},
                                },
                            },
                        },
                        "representative_face_path": {"type": ["string", "null"]},
                    },
                },
            },
            "clustering_metrics": {"type": "object"},
            "output_dir": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=2048, vram_mb=512, disk_mb=1500, network_required=False
    )
    idempotency_key_fields = [
        "input_path",
        "sample_fps",
        "cluster_threshold",
        "min_face_size",
    ]
    side_effects = [
        "writes identity JSON to output_dir",
        "writes face thumbnail JPGs to output_dir",
        "may write mp4 clip files when extract_clips=true",
    ]
    user_visible_verification = [
        "Inspect the representative face thumbnail for each identity",
        "Spot-check identity segment boundaries against the video",
    ]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> ToolStatus:
        if not self._has_ffmpeg():
            return ToolStatus.UNAVAILABLE
        if not self._has_backend():
            return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE

    def _has_ffmpeg(self) -> bool:
        import shutil

        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    def _has_backend(self) -> bool:
        try:
            import insightface  # noqa: F401
            import sklearn.cluster  # noqa: F401

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
            return ToolResult(success=False, error="ffmpeg/ffprobe not found on PATH.")
        if not self._has_backend():
            return ToolResult(
                success=False,
                error=(
                    "insightface/sklearn not installed. "
                    "Install with: pip install insightface onnxruntime scikit-learn"
                ),
            )

        start = time.time()
        tool_registry.ensure_discovered()

        output_dir = Path(
            inputs.get("output_dir", str(input_path.parent / f"{input_path.stem}_faces"))
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        sample_fps = float(inputs.get("sample_fps", 2.0))
        cluster_threshold = float(inputs.get("cluster_threshold", 0.42))
        min_face_size = int(inputs.get("min_face_size", 48))
        max_gap = float(inputs.get("max_gap_seconds", 2.0))
        min_track = float(inputs.get("min_track_seconds", 1.0))
        max_identities = int(inputs.get("max_identities", 50))
        device = str(inputs.get("device", "auto"))

        # 1) Probe duration.
        duration = self._probe_duration(input_path)
        if duration <= 0:
            return ToolResult(success=False, error=f"Could not determine duration for {input_path}")

        # 2) Sample frames → (frame_path, timestamp) list.
        frames = self._sample_frames(input_path, output_dir / "_frames", sample_fps, duration)
        if not frames:
            return ToolResult(success=False, error="No frames could be extracted from the video.")

        # 3) Detect + embed every face in every sampled frame.
        detections = self._embed_faces(frames, min_face_size, device)
        if isinstance(detections, str):
            return ToolResult(success=False, error=detections)
        if not detections:
            payload = self._empty_payload(output_dir, len(frames))
            (output_dir / "identities.json").write_text(json.dumps(payload, indent=2))
            return ToolResult(success=True, data=payload, duration_seconds=round(time.time() - start, 2))

        # 4) Cluster embeddings into identities.
        identities, metrics = self._cluster(
            detections,
            threshold=cluster_threshold,
            max_gap=max_gap,
            min_track=min_track,
            sample_fps=sample_fps,
            duration=duration,
        )

        # 5) Rank + cap identities, save thumbnails.
        identities.sort(key=lambda i: i["total_duration_seconds"], reverse=True)
        identities = identities[:max_identities]
        for new_id, identity in enumerate(identities):
            identity["id"] = new_id
            identity["label"] = f"face_{new_id}"

        self._save_thumbnails(identities, output_dir)

        # 6) Optional physical extraction.
        extracted = False
        if inputs.get("extract_clips", False):
            clips_dir = output_dir / inputs.get("clips_subdir", "clips")
            clips_dir.mkdir(parents=True, exist_ok=True)
            extracted = self._extract_clips(identities, input_path, clips_dir)

        elapsed = time.time() - start
        payload = {
            "method": "insightface+agglomerative",
            "identities_count": len(identities),
            "frames_analyzed": len(frames),
            "identities": identities,
            "clipping_extracted": extracted,
            "clustering_metrics": metrics,
            "output_dir": str(output_dir),
        }
        (output_dir / "identities.json").write_text(json.dumps(payload, indent=2))
        return ToolResult(
            success=True,
            data=payload,
            artifacts=[str(output_dir / "identities.json")],
            duration_seconds=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _probe_duration(self, input_path: Path) -> float:
        try:
            probe = tool_registry.get("audio_probe")
            if probe is not None:
                res = probe.execute({"input_path": str(input_path)})
                if res.success:
                    return float(res.data.get("duration_seconds", 0)) or 0.0
        except Exception:
            pass
        # ffprobe fallback
        try:
            import subprocess

            out = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "json", str(input_path),
                ],
                capture_output=True, text=True, timeout=30, check=True,
            )
            return float(json.loads(out.stdout)["format"]["duration"])
        except Exception:
            return 0.0

    def _sample_frames(
        self, input_path: Path, frames_dir: Path, sample_fps: float, duration: float
    ) -> list[tuple[Path, float]]:
        """Sample frames at a fixed fps via ffmpeg. Returns [(path, timestamp)]."""
        frames_dir.mkdir(parents=True, exist_ok=True)
        import subprocess

        # ffmpeg -i in -vf fps=FPS -q:v 2 out/%06d.jpg
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", f"fps={sample_fps}",
            "-q:v", "3",
            str(frames_dir / "%06d.jpg"),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
        except Exception:
            return []

        files = sorted(frames_dir.glob("*.jpg"))
        # timestamp = (index) / sample_fps  (1-based -> 0-based)
        return [(f, (i) / sample_fps) for i, f in enumerate(files)]

    def _embed_faces(
        self, frames: list[tuple[Path, float]], min_face_size: int, device: str = "auto"
    ) -> list[dict] | str:
        """For each frame, detect faces and embed each. Returns list of detection
        dicts: {frame_path, timestamp, bbox, det_score, embedding}.

        ``device`` resolves the InsightFace ``ctx_id``: gpu=0 when forced or
        auto-detected, cpu=-1 otherwise. Inference is always local."""
        try:
            import numpy as np
            from PIL import Image
            from insightface.app import FaceApp
        except ImportError as e:
            return f"Required dependency missing: {e}"

        ctx_id = self._resolve_ctx_id(device)
        try:
            app = FaceApp(name="buffalo_l")
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        except Exception as e:
            # If GPU was requested/autodetected but onnxruntime-gpu isn't set up,
            # fall back to CPU rather than failing the whole job.
            if ctx_id >= 0:
                try:
                    app = FaceApp(name="buffalo_l")
                    app.prepare(ctx_id=-1, det_size=(640, 640))
                except Exception as e2:
                    return f"Could not initialize InsightFace (gpu failed: {e}; cpu fallback failed: {e2})"
            else:
                return f"Could not initialize InsightFace: {e}"

        detections: list[dict] = []
        for frame_path, ts in frames:
            try:
                img = np.array(Image.open(frame_path).convert("RGB"))
                # RGB -> BGR for insightface
                img_bgr = img[:, :, ::-1]
                faces = app.get(img_bgr)
            except Exception:
                continue
            for face in faces:
                bbox = face.bbox.tolist() if hasattr(face.bbox, "tolist") else list(face.bbox)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < min_face_size or h < min_face_size:
                    continue
                emb = face.normed_embedding
                if emb is None:
                    continue
                detections.append(
                    {
                        "frame_path": str(frame_path),
                        "timestamp": ts,
                        "bbox": [round(v, 2) for v in bbox],
                        "det_score": float(face.det_score),
                        "embedding": emb.tolist(),
                    }
                )
        return detections

    @staticmethod
    def _resolve_ctx_id(device: str) -> int:
        """Map the user-facing device string to InsightFace's ctx_id.

        - 'cpu'  -> -1 (force CPU)
        - 'gpu'  ->  0 (force first GPU; caller catches if unavailable)
        - 'auto' ->  0 if a CUDA GPU + onnxruntime-gpu are importable, else -1
        Inference is local either way — no third-party API is ever called.
        """
        device = (device or "auto").lower()
        if device == "cpu":
            return -1
        if device == "gpu":
            return 0
        # auto: probe for a usable CUDA provider via onnxruntime.
        try:
            import onnxruntime as ort

            if any("CUDA" in p for p in ort.get_available_providers()):
                return 0
        except Exception:
            pass
        return -1

    def _cluster(
        self,
        detections: list[dict],
        *,
        threshold: float,
        max_gap: float,
        min_track: float,
        sample_fps: float,
        duration: float,
    ) -> tuple[list[dict], dict]:
        """Cluster embeddings, then turn clusters into identities with segments."""
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        X = np.array([d["embedding"] for d in detections], dtype=np.float32)
        # Normalize (already normalized by ArcFace, but be defensive).
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=threshold,
        )
        labels = clustering.fit_predict(X)

        # Group detections by cluster label.
        by_cluster: dict[int, list[dict]] = {}
        for det, label in zip(detections, labels):
            by_cluster.setdefault(int(label), []).append(det)

        # Silhouette (best-effort; needs >= 2 clusters and >= 2 samples each).
        metrics: dict[str, Any] = {"threshold": threshold, "n_clusters": len(by_cluster)}
        try:
            if len(by_cluster) >= 2:
                from sklearn.metrics import silhouette_score

                sil = silhouette_score(X, labels, metric="cosine")
                metrics["silhouette"] = round(float(sil), 4)
        except Exception:
            pass

        identities: list[dict] = []
        for label, dets in by_cluster.items():
            dets.sort(key=lambda d: d["timestamp"])
            segments = self._merge_hits_to_segments(dets, max_gap, duration)
            total = sum(s["duration_seconds"] for s in segments)
            if total < min_track:
                continue
            # Representative face = highest det_score.
            rep = max(dets, key=lambda d: d["det_score"])
            identities.append(
                {
                    "id": -1,  # reassigned after sorting
                    "label": "",
                    "total_duration_seconds": round(total, 3),
                    "segment_count": len(segments),
                    "segments": segments,
                    "representative_face_path": None,
                    "_representative": rep,  # internal, stripped before output
                    "frame_hits": len(dets),
                }
            )
        return identities, metrics

    def _merge_hits_to_segments(
        self, dets: list[dict], max_gap: float, duration: float
    ) -> list[dict]:
        """Merge a sorted list of same-identity frame hits into contiguous segments.
        Each hit covers roughly [t, t+1/sample_fps); extend until there's a gap > max_gap."""
        if not dets:
            return []
        sample_interval = None
        if len(dets) >= 2:
            # crude estimate from the actual spacing
            diffs = [
                dets[i + 1]["timestamp"] - dets[i]["timestamp"]
                for i in range(len(dets) - 1)
                if dets[i + 1]["timestamp"] > dets[i]["timestamp"]
            ]
            if diffs:
                sample_interval = sum(diffs) / len(diffs)

        segments: list[dict] = []
        seg_start = dets[0]["timestamp"]
        seg_end = dets[0]["timestamp"] + (sample_interval or 0.0)
        for d in dets[1:]:
            t = d["timestamp"]
            if t - seg_end <= max_gap:
                seg_end = t + (sample_interval or 0.0)
            else:
                segments.append(self._seg(seg_start, min(seg_end, duration)))
                seg_start = t
                seg_end = t + (sample_interval or 0.0)
        segments.append(self._seg(seg_start, min(seg_end, duration)))
        return [s for s in segments if s["duration_seconds"] > 0]

    @staticmethod
    def _seg(start: float, end: float) -> dict:
        start = max(0.0, start)
        end = max(start, end)
        return {
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(end - start, 3),
        }

    def _save_thumbnails(self, identities: list[dict], output_dir: Path) -> None:
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return
        for identity in identities:
            rep = identity.get("_representative")
            if not rep:
                continue
            try:
                img = Image.open(rep["frame_path"]).convert("RGB")
                x1, y1, x2, y2 = rep["bbox"]
                # Pad the crop a little for a more recognizable thumbnail.
                pad = 0.25
                w, h = x2 - x1, y2 - y1
                x1 = max(0, x1 - w * pad)
                y1 = max(0, y1 - h * pad)
                x2 = min(img.width, x2 + w * pad)
                y2 = min(img.height, y2 + h * pad)
                crop = img.crop((x1, y1, x2, y2))
                out_path = output_dir / f"{identity['label']}.jpg"
                crop.save(out_path, quality=90)
                identity["representative_face_path"] = str(out_path)
            except Exception:
                identity["representative_face_path"] = None
            finally:
                identity.pop("_representative", None)

    def _extract_clips(
        self, identities: list[dict], input_path: Path, clips_dir: Path
    ) -> bool:
        trimmer = tool_registry.get("video_trimmer")
        if trimmer is None:
            return False
        any_ok = False
        for identity in identities:
            for sidx, seg in enumerate(identity["segments"]):
                clip_path = clips_dir / f"{identity['label']}_seg{sidx:02d}.mp4"
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

    def _empty_payload(self, output_dir: Path, frames_analyzed: int) -> dict:
        return {
            "method": "insightface+agglomerative",
            "identities_count": 0,
            "frames_analyzed": frames_analyzed,
            "identities": [],
            "clipping_extracted": False,
            "clustering_metrics": {"threshold": None, "n_clusters": 0},
            "output_dir": str(output_dir),
        }
