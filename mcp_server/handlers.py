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

from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import Context

from tools.tool_registry import registry
from mcp_server.execution import execute_tool_async, serialize_result


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
        ctx: FastMCP Context (injected by the server; pass None when calling
            directly in tests).

    Returns:
        Serialized ToolResult: {success, data, artifacts, error, cost_usd,
        duration_seconds, seed, model}.
    """
    _ensure_discovered()
    tool = registry.get(tool_name)
    if tool is None:
        raise ValueError(f"Tool {tool_name!r} not found.")

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
