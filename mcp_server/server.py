"""FastMCP server wiring — registers handlers as MCP tools and resources.

Run with ``python -m mcp_server`` (see __main__.py for transport/host/port).
The server exposes 8 tools (discovery, execution, orchestration primitives) and
a set of resources (instruction docs). External agents connect as MCP clients,
discover these, and orchestrate video production themselves — OpenMontage stays
a pure tool+library layer, per its "agent is the orchestrator" architecture.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server import handlers, resources

mcp = FastMCP("openmontage")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# FastMCP derives the tool's JSON schema from the function signature + type
# hints, and the description from the docstring. Keep both accurate.

@mcp.tool()
async def discover_tools() -> dict:
    """Discover and load all OpenMontage tools; return names grouped by capability.

    Call this (or provider_menu_summary) first so the tool registry is populated.
    Returns {capabilities: {<cap>: [<tool_name>...]}, total: <N>}.
    """
    return await handlers.discover_tools()


@mcp.tool()
async def provider_menu_summary() -> dict:
    """Return the compact capability menu (the 'N of M configured' preflight rollup).

    Use this to know which providers are actually configured on this machine
    before planning production. Mirrors `make preflight`. Shape:
    {composition_runtimes, capabilities[], setup_offers[], runtime_warnings[]}.
    """
    return await handlers.provider_menu_summary()


@mcp.tool()
async def get_tool_info(tool_name: str) -> dict:
    """Return the full self-describing contract for one tool.

    Includes input_schema, output_schema, dependencies, install_instructions,
    best_for, agent_skills, runtime, status. Inspect input_schema to learn the
    parameters before calling execute_tool.
    """
    return await handlers.get_tool_info(tool_name)


@mcp.tool()
async def execute_tool(tool_name: str, inputs: dict, confirm: bool = False) -> dict:
    """Execute an OpenMontage tool and return its ToolResult as a dict.

    Runs in a worker thread (long FFmpeg/Remotion jobs won't block the server).
    Returns {success, data, artifacts, error, cost_usd, duration_seconds, seed,
    model}. Check input_schema via get_tool_info first; pass a matching `inputs`.

    Set confirm=True for publish-style tools (those that push content to an
    external platform); the call raises PermissionError otherwise.

    Example: execute_tool("video_trimmer", {"operation":"cut","input_path":"in.mp4",
    "output_path":"out.mp4","start_seconds":0,"end_seconds":5}).
    """
    return await handlers.execute_tool(tool_name, inputs, confirm=confirm, ctx=mcp)


@mcp.tool()
async def list_pipelines() -> dict:
    """List all available pipeline manifest names (the workflows you can orchestrate).

    Returns {pipelines: [<name>...], total: N}. Pass any name to
    get_pipeline_manifest for the stage/tools breakdown.
    """
    return await handlers.list_pipelines()


@mcp.tool()
async def get_pipeline_manifest(pipeline_name: str) -> dict:
    """Return a pipeline's manifest + derived stage/tools/review breakdown.

    Gives stage_order, per-stage skill path + tools_available + review_focus +
    human_approval_default, and the union of required_tools. Use this to drive
    stage-by-stage orchestration (the agent decides ordering and checkpoints).
    """
    return await handlers.get_pipeline_manifest(pipeline_name)


@mcp.tool()
async def read_checkpoint(
    project_id: str,
    stage: str | None = None,
    pipeline_dir: str = "pipeline",
    pipeline_type: str | None = None,
) -> dict:
    """Read a checkpoint (or the latest) and compute the next stage to resume.

    If `stage` is omitted, reads the latest checkpoint. Returns the checkpoint,
    latest_stage, next_stage (where to resume), and completed_stages. Use this
    between stages to decide what to run next.
    """
    return await handlers.read_checkpoint(
        project_id, stage, pipeline_dir=pipeline_dir, pipeline_type=pipeline_type
    )


@mcp.tool()
async def write_checkpoint(
    project_id: str,
    stage: str,
    status: str,
    artifacts: dict,
    pipeline_dir: str = "pipeline",
    pipeline_type: str | None = None,
    style_playbook: str | None = None,
    checkpoint_policy: str = "guided",
    human_approval_required: bool = False,
    human_approved: bool = False,
    review: dict | None = None,
    cost_snapshot: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Write a validated checkpoint for a stage (call after completing a stage).

    Validates the checkpoint + canonical artifact against project JSON schemas
    (raises on invalid artifacts). Returns {path, stage, status, next_stage}.
    `status` is one of: completed, awaiting_human, in_progress, failed.
    """
    return await handlers.write_checkpoint(
        project_id, stage, status, artifacts,
        pipeline_dir=pipeline_dir,
        pipeline_type=pipeline_type,
        style_playbook=style_playbook,
        checkpoint_policy=checkpoint_policy,
        human_approval_required=human_approval_required,
        human_approved=human_approved,
        review=review,
        cost_snapshot=cost_snapshot,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Resources — static guide docs (discoverable via resources/list)
# ---------------------------------------------------------------------------

@mcp.resource("om://guide/agent-guide")
def _guide_agent_guide() -> str:
    """AGENT_GUIDE.md — the central agent operating contract. Read first."""
    text, _ = resources.read("om://guide/agent-guide")
    return text


@mcp.resource("om://guide/project-context")
def _guide_project_context() -> str:
    """PROJECT_CONTEXT.md — architecture, key files, conventions."""
    text, _ = resources.read("om://guide/project-context")
    return text


@mcp.resource("om://guide/agents")
def _guide_agents() -> str:
    """AGENTS.md — the mandatory entry pointer."""
    text, _ = resources.read("om://guide/agents")
    return text


@mcp.resource("om://guide/readme")
def _guide_readme() -> str:
    """README.md — project overview and quick start."""
    text, _ = resources.read("om://guide/readme")
    return text


@mcp.resource("om://pipelines/{name}")
def _pipeline_manifest(name: str) -> str:
    """A pipeline manifest YAML (stage definitions, tools, review focus)."""
    text, _ = resources.read(f"om://pipelines/{name}")
    return text


@mcp.resource("om://skills/{path}")
def _skill(path: str) -> str:
    """A Layer-2 skill doc under skills/ (stage directors, meta skills, core)."""
    text, _ = resources.read(f"om://skills/{path}")
    return text


@mcp.resource("om://agent-skills/{path}")
def _agent_skill(path: str) -> str:
    """A Layer-3 vendor/technology skill under .agents/skills/."""
    text, _ = resources.read(f"om://agent-skills/{path}")
    return text


@mcp.resource("om://styles/{name}")
def _style(name: str) -> str:
    """A visual style playbook YAML under styles/."""
    text, _ = resources.read(f"om://styles/{name}")
    return text
