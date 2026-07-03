"""MCP tool handlers — thin wrappers over OpenMontage's existing libraries.

Each handler is a pure async function registered with FastMCP in server.py.
They call the real ``registry`` singleton, ``lib.pipeline_loader``, and
``lib.checkpoint`` — no business logic is duplicated here. Results are plain
dicts/lists so the MCP wire format can serialize them.

Handler groups:
  - Discovery & execution: discover_tools, provider_menu_summary, get_tool_info,
    execute_tool
  - Orchestration primitives: list_pipelines, get_pipeline_manifest,
    read_checkpoint, write_checkpoint
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import Context

from tools.base_tool import BaseTool, ToolTier
from tools.tool_registry import registry
from mcp_server.execution import execute_tool_async, serialize_result


# A tool is treated as "externally publishing" (irreversible, out-facing) if it
# declares tier=publish, capability=publish, or has a side_effect that smells of
# pushing content to an external platform. The publishers/ tool subpackage is
# currently an empty shell, so this guard is forward-looking: when real
# publishing tools land, the guard activates automatically without code changes.
_PUBLISH_INTENT_RE = re.compile(
    r"\b(publish|upload|post_?to|youtube|tiktok|instagram|twitter|x\.com|"
    r"social|schedule|broadcast)\b",
    re.IGNORECASE,
)


def _requires_publish_confirmation(tool: BaseTool) -> Optional[str]:
    """Return a reason string if the tool is a publish-style action, else None.

    A publish action is one that pushes content to an external platform —
    irreversible and out-facing. Local file writes and generation-API calls are
    NOT publish actions (they're reversible/controllable). We trigger on
    tier=PUBLISH, capability=publish, or a matching side_effect.
    """
    if tool.tier == ToolTier.PUBLISH:
        return f"tool tier is 'publish'"
    if tool.capability == "publish":
        return f"tool capability is 'publish'"
    for effect in getattr(tool, "side_effects", []) or []:
        if _PUBLISH_INTENT_RE.search(str(effect)):
            return f"tool has publishing side_effect: {effect!r}"
    return None


def _ensure_discovered() -> None:
    """Idempotently discover tools.

    Only triggers discovery if the registry is empty — so a registry that's
    already populated (e.g. pre-seeded in tests, or a prior call in the same
    server process) is left untouched. ``ensure_discovered`` would otherwise
    re-import the tools package and re-populate even a hand-built registry.
    """
    if not registry.list_all():
        registry.ensure_discovered()


# ---------------------------------------------------------------------------
# Discovery & execution
# ---------------------------------------------------------------------------

async def discover_tools() -> dict[str, Any]:
    """Load all OpenMontage tools and return their names grouped by capability.

    Call this first (or provider_menu_summary) so the registry is populated.
    Returns ``{capabilities: {cap: [names]}, total: N}``.
    """
    _ensure_discovered()
    by_cap: dict[str, list[str]] = {}
    for name in sorted(registry.list_all()):
        tool = registry.get(name)
        if tool is None:
            continue
        by_cap.setdefault(tool.capability, []).append(name)
    return {"capabilities": by_cap, "total": len(registry.list_all())}


async def provider_menu_summary() -> dict[str, Any]:
    """Return the compact, human-ready capability menu (the preflight rollup).

    Mirrors ``make preflight`` output shape: composition_runtimes, capabilities
    (with configured/total counts + provider lists), setup_offers, runtime_warnings.
    """
    _ensure_discovered()
    return registry.provider_menu_summary()


async def get_tool_info(tool_name: str) -> dict[str, Any]:
    """Return the full self-describing contract for one tool.

    Includes input/output schemas, dependencies, install_instructions, best_for,
    agent_skills, runtime, status, etc. Use this to learn a tool's parameters
    before calling execute_tool.
    """
    _ensure_discovered()
    tool = registry.get(tool_name)
    if tool is None:
        raise ValueError(
            f"Tool {tool_name!r} not found. Call discover_tools or "
            f"provider_menu_summary to see available tools."
        )
    info = tool.get_info()
    # Guard: never echo raw dependency env values; get_info() already lists
    # dependency *names* (env:KEY), not values, so this is belt-and-suspenders.
    return info


async def execute_tool(
    tool_name: str,
    inputs: dict[str, Any],
    *,
    confirm: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Execute an OpenMontage tool and return its ToolResult as a dict.

    This is the core call. The tool runs in a worker thread so long-running
    jobs (FFmpeg/Remotion) don't block the MCP message loop. Progress is reported
    via MCP notifications when a Context is available.

    Args:
        tool_name: Registered tool name (e.g. "video_trimmer", "scene_detect").
        inputs: Tool input dict matching the tool's input_schema. Use
            get_tool_info to inspect the schema first.
        confirm: Required (True) for publish-style tools — those that push
            content to an external platform (tier=publish, capability=publish,
            or a publishing side_effect). Mirrors AGENT_GUIDE's "announce before
            execution": an external-facing, irreversible action must be opted
            into explicitly rather than fired off silently.
        ctx: FastMCP Context (injected by the server; pass None when calling
            directly in tests).

    Returns:
        Serialized ToolResult: {success, data, artifacts, error, cost_usd,
        duration_seconds, seed, model}.

    Raises:
        PermissionError: If the tool is publish-style and ``confirm`` is False.
    """
    _ensure_discovered()
    tool = registry.get(tool_name)
    if tool is None:
        raise ValueError(f"Tool {tool_name!r} not found.")

    # Guard: external-facing publish actions require explicit confirmation.
    publish_reason = _requires_publish_confirmation(tool)
    if publish_reason and not confirm:
        raise PermissionError(
            f"Tool {tool_name!r} is a publish action ({publish_reason}). "
            f"Re-call execute_tool with confirm=True to proceed. "
            f"This mirrors AGENT_GUIDE's 'announce before execution' rule."
        )

    status = tool.get_status()
    if ctx is not None:
        await ctx.info(f"Executing {tool_name} (status={status.value})")

    result = await execute_tool_async(tool, inputs)

    if ctx is not None:
        # Report progress as a terminal update. Tools don't emit streaming
        # progress today, so we signal start (above) and completion (here).
        await ctx.report_progress(1, 1)
        if result.success:
            await ctx.info(
                f"{tool_name} succeeded in {result.duration_seconds}s "
                f"(cost ${result.cost_usd:.4f})"
            )
        else:
            await ctx.info(f"{tool_name} failed: {result.error}")

    return serialize_result(result)


async def submit_tool_job(
    tool_name: str,
    inputs: dict[str, Any],
    *,
    confirm: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Submit a tool for async execution; return a job descriptor immediately.

    Use this instead of execute_tool for long-running tools (renders, downloads)
    so the MCP call returns at once with a job_id. Poll with get_job_status.

    Args:
        tool_name: Registered tool name.
        inputs: Tool input dict (same shape execute_tool accepts).
        confirm: Required True for publish-style tools (same guard as execute_tool).
        ctx: FastMCP Context (injected by the server).

    Returns:
        Job snapshot: {job_id, tool_name, status:"pending", progress:0,
        created_at}. Poll via get_job_status(job_id); a terminal snapshot
        includes ``result`` (serialized ToolResult) and ``elapsed_seconds``.
    """
    from mcp_server.job_manager import jobs

    _ensure_discovered()
    tool = registry.get(tool_name)
    if tool is None:
        raise ValueError(f"Tool {tool_name!r} not found.")

    # Same publish-action guard as execute_tool — async submission must not
    # bypass the confirm requirement.
    publish_reason = _requires_publish_confirmation(tool)
    if publish_reason and not confirm:
        raise PermissionError(
            f"Tool {tool_name!r} is a publish action ({publish_reason}). "
            f"Re-call submit_tool_job with confirm=True to proceed."
        )

    def _on_started(job) -> None:
        if ctx is not None:
            # report_progress is async; the hook runs from a worker thread, so
            # schedule the notification rather than awaiting it here.
            try:
                asyncio.create_task(ctx.info(f"Job {job.job_id} started: {tool_name}"))
            except Exception:
                pass

    job = await jobs.submit(tool, inputs, started_hook=_on_started)
    if ctx is not None:
        await ctx.info(f"Submitted {tool_name} as job {job.job_id}")
    return job.to_dict()


async def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll an async job submitted via submit_tool_job.

    Returns the current snapshot: {job_id, tool_name, status, progress, ...}.
    status is one of pending|running|succeeded. A succeeded snapshot includes
    ``result`` (the serialized ToolResult — check result.success for the tool's
    own pass/fail) and ``elapsed_seconds``. Unknown job_id raises ValueError.
    """
    from mcp_server.job_manager import jobs

    job = jobs.get(job_id)
    if job is None:
        raise ValueError(
            f"Job {job_id!r} not found. It may have aged out after a server restart, "
            f"or the id is wrong."
        )
    return job.to_dict()


async def list_jobs() -> dict[str, Any]:
    """List all known async jobs (newest-first), with a status tally.

    Useful for clients that submitted several jobs and want to see the backlog.
    Returns {jobs: [<snapshot>...], counts: {pending, running, succeeded}}.
    """
    from mcp_server.job_manager import jobs, JobStatus

    all_jobs = jobs.list_jobs()
    counts = {"pending": 0, "running": 0, "succeeded": 0}
    for j in all_jobs:
        if j.status.value in counts:
            counts[j.status.value] += 1
    return {"jobs": [j.to_dict() for j in all_jobs], "counts": counts}


# ---------------------------------------------------------------------------
# Orchestration primitives (the client agent does the actual orchestration)
# ---------------------------------------------------------------------------

async def list_pipelines() -> dict[str, Any]:
    """Return all available pipeline manifest names.

    Each name can be passed to get_pipeline_manifest for the full stage/tools
    breakdown. These are the workflows the client agent can orchestrate.
    """
    from lib.pipeline_loader import list_pipelines as _list
    names = sorted(_list())
    return {"pipelines": names, "total": len(names)}


async def get_pipeline_manifest(pipeline_name: str) -> dict[str, Any]:
    """Return a pipeline's manifest plus derived orchestration helpers.

    Loads and validates the YAML manifest, then computes:
      - stage_order: ordered stage names to run
      - stages: per-stage skill path, required/preferred/fallback tools,
        review_focus, human_approval_default
      - required_tools: union of all tools the pipeline may invoke
    The client agent uses this to drive stage-by-stage execution.
    """
    from lib.pipeline_loader import (
        load_pipeline,
        get_stage_order,
        get_required_tools,
        get_stage_skill,
        get_stage_review_focus,
    )

    manifest = load_pipeline(pipeline_name)
    stage_order = get_stage_order(manifest)
    stages_info: list[dict[str, Any]] = []
    for stage_def in manifest.get("stages", []):
        stages_info.append({
            "name": stage_def.get("name"),
            "skill": stage_def.get("skill"),
            "produces": stage_def.get("produces"),
            "tools_available": stage_def.get("tools_available", []),
            "preferred_tools": stage_def.get("preferred_tools", []),
            "fallback_tools": stage_def.get("fallback_tools", []),
            "review_focus": stage_def.get("review_focus", []),
            "human_approval_default": stage_def.get("human_approval_default"),
            "success_criteria": stage_def.get("success_criteria", []),
        })
    return {
        "name": manifest.get("name", pipeline_name),
        "description": manifest.get("description", ""),
        "orchestration_mode": manifest.get("orchestration_mode"),
        "stability": manifest.get("stability"),
        "stage_order": stage_order,
        "stages": stages_info,
        "required_tools": sorted(get_required_tools(manifest)),
        "human_approval_default": manifest.get("human_approval_default"),
    }


async def read_checkpoint(
    project_id: str,
    stage: Optional[str] = None,
    *,
    pipeline_dir: str = "pipeline",
    pipeline_type: Optional[str] = None,
) -> dict[str, Any]:
    """Read a checkpoint (or the latest) and compute the next stage to run.

    If ``stage`` is given, reads that specific checkpoint; otherwise reads the
    latest checkpoint (by mtime). Also returns ``next_stage`` computed from the
    pipeline's stage order, so the client knows where to resume. Returns
    ``{checkpoint: <dict|null>, latest_stage, next_stage, completed_stages}``.
    """
    from lib import checkpoint as cp

    base = Path(pipeline_dir)
    if stage is not None:
        cp_data = cp.read_checkpoint(base, project_id, stage)
    else:
        cp_data = cp.get_latest_checkpoint(base, project_id)

    completed = cp.get_completed_stages(base, project_id, pipeline_type)
    next_stage = cp.get_next_stage(base, project_id, pipeline_type)
    latest_stage = cp_data.get("stage") if cp_data else None

    return {
        "checkpoint": cp_data,
        "latest_stage": latest_stage,
        "next_stage": next_stage,
        "completed_stages": completed,
    }


async def write_checkpoint(
    project_id: str,
    stage: str,
    status: str,
    artifacts: dict[str, Any],
    *,
    pipeline_dir: str = "pipeline",
    pipeline_type: Optional[str] = None,
    style_playbook: Optional[str] = None,
    checkpoint_policy: str = "guided",
    human_approval_required: bool = False,
    human_approved: bool = False,
    review: Optional[dict[str, Any]] = None,
    cost_snapshot: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write a checkpoint for a pipeline stage.

    Validates the checkpoint and its canonical artifact against the project's
    JSON schemas (raises on invalid artifacts). Returns the path written and
    the next stage to run. The client agent calls this after completing each
    stage's work.
    """
    from lib import checkpoint as cp

    path = cp.write_checkpoint(
        Path(pipeline_dir),
        project_id,
        stage,
        status,
        artifacts,
        pipeline_type=pipeline_type,
        style_playbook=style_playbook,
        checkpoint_policy=checkpoint_policy,
        human_approval_required=human_approval_required,
        human_approved=human_approved,
        review=review,
        cost_snapshot=cost_snapshot,
        metadata=metadata,
    )
    next_stage = cp.get_next_stage(Path(pipeline_dir), project_id, pipeline_type)
    return {"path": str(path), "stage": stage, "status": status, "next_stage": next_stage}
