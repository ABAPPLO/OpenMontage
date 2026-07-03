# OpenMontage MCP Server

Exposes OpenMontage to **external agents** (any language, cross-process) over the
[Model Context Protocol](https://modelcontextprotocol.io). If your agent isn't a
Python process that can `import` the repo, this is the integration surface.

> **Architecture note:** OpenMontage has *no* Python orchestrator — the agent IS
> the orchestrator ("agent-first" design). This server stays faithful to that: it
> exposes **primitives** (tool execution, pipeline/checkpoint helpers, instruction
> docs) and leaves **orchestration to your agent**. Your agent reads the guide +
> pipeline manifests over MCP and drives the stages itself.

## Quick start

The easiest entry point is the launcher script (`mcp-server.sh`, run from the repo
root). It resolves a Python 3.10+ interpreter (venv first), checks the `mcp` SDK
is installed, and dispatches to the right mode:

```bash
./mcp-server.sh              # stdio (default — for Claude Desktop, Cursor, etc.)
./mcp-server.sh http 8765    # streamable-http on port 8765 (for remote/HTTP clients)
./mcp-server.sh sse 8765     # SSE transport
./mcp-server.sh demo         # end-to-end smoke test (generates a test video + slices it)
./mcp-server.sh inspect      # launch the MCP Inspector web GUI (see "Testing" below)
./mcp-server.sh install      # install the `mcp` SDK into the venv
./mcp-server.sh --help       # full usage
```

> First run? Start with `./mcp-server.sh install` (one-time) then `./mcp-server.sh demo`
> to confirm everything works.

Prefer `make` or raw Python? Those still work:

```bash
make mcp-install             # install the mcp SDK
make mcp                     # start (stdio default)
make mcp TRANSPORT=streamable-http PORT=8765

python -m mcp_server --transport stdio
python -m mcp_server --transport streamable-http --host 127.0.0.1 --port 8765
```

Requires Python 3.10+ and a configured OpenMontage repo (`make setup` done, `.env`
present). The server reuses the repo's `.env` for API keys — no separate config.

## Testing

Three ways to verify the server works, no external MCP client required:

**1. Smoke test (CLI, fastest):**
```bash
./mcp-server.sh demo
```
Spawns the server over stdio, generates a 4s test video with ffmpeg, slices it
via `execute_tool` (sync) and `submit_tool_job` (async), and prints each output
path. Exit 0 = everything works.

**2. MCP Inspector (web GUI) — best for interactive exploration:**
```bash
./mcp-server.sh inspect
# or: npx @modelcontextprotocol/inspector .venv/bin/python -m mcp_server
```
Opens a web UI at `http://localhost:6274` (the launcher prints a URL with a
session token). In the GUI: keep **Transport = STDIO**, the command is pre-filled,
click **Connect**, then browse the 11 tools. Click any tool (e.g. `execute_tool`)
and the right pane auto-builds a form from its JSON schema — fill the params and
run. Returns show as formatted JSON. Requires Node.js for `npx`.

**3. Automated test suite:**
```bash
python -m pytest tests/mcp/ -v
```
62 tests covering handlers, async jobs, sandboxed resources, real ffmpeg slicing,
and end-to-end stdio server calls.

## Register with an MCP client

**Claude Desktop / Cursor / any MCP client** — add to your client config:

```json
{
  "mcpServers": {
    "openmontage": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/path/to/OpenMontage"
    }
  }
}
```

For networked transport, point your client at `http://127.0.0.1:8765` (the server
binds to localhost by default; set `host` for remote access).

## What's exposed

### 11 Tools (`tools/list` → `tools/call`)

**Discovery & execution**

| Tool | Purpose |
|---|---|
| `discover_tools` | Load all tools; return names grouped by capability |
| `provider_menu_summary` | The "N of M configured" preflight rollup (runtimes, providers, setup offers) |
| `get_tool_info(name)` | Full contract for one tool — **read `input_schema` before calling execute** |
| `execute_tool(name, inputs, confirm?)` | **Core** — run a tool synchronously; returns `{success, data, artifacts, error, cost_usd, duration_seconds, seed, model}`. Set `confirm=true` for publish-style tools. |

**Async execution (long jobs)**

| Tool | Purpose |
|---|---|
| `submit_tool_job(name, inputs, confirm?)` | Submit a tool for background execution; returns a job snapshot with `job_id` immediately. Use this for renders/downloads that take minutes. |
| `get_job_status(job_id)` | Poll a job: `status` is `pending`→`running`→`succeeded`; a succeeded snapshot includes `result` (serialized ToolResult) + `elapsed_seconds`. |
| `list_jobs()` | All jobs newest-first with a status tally `{pending, running, succeeded}`. |

**Orchestration primitives (your agent drives the pipeline)**

| Tool | Purpose |
|---|---|
| `list_pipelines` | All workflow names (`clip-factory`, `animated-explainer`, ...) |
| `get_pipeline_manifest(name)` | Stage order, per-stage skill + tools + review_focus, required_tools |
| `read_checkpoint(project_id, stage?)` | Read a stage checkpoint + compute `next_stage` to resume |
| `write_checkpoint(...)` | Write a validated checkpoint after completing a stage |

### Resources (`resources/list` → `resources/read`)

Your agent needs the instruction docs to orchestrate correctly. Read them over MCP:

| URI | File |
|---|---|
| `om://guide/agent-guide` | `AGENT_GUIDE.md` — **read first**, the operating contract |
| `om://guide/project-context` | `PROJECT_CONTEXT.md` — architecture & conventions |
| `om://guide/agents` | `AGENTS.md` |
| `om://guide/readme` | `README.md` |
| `om://pipelines/{name}` | `pipeline_defs/{name}.yaml` |
| `om://skills/{path}` | Layer-2 skills (stage directors, meta, core) |
| `om://agent-skills/{path}` | Layer-3 vendor/tech skills (`.agents/skills/`) |
| `om://styles/{name}` | Visual style playbooks |

All `om://` paths are **sandboxed** to the repo's doc/skill directories; path
traversal (`..`) and absolute paths are rejected.

## End-to-end example: slice a video

A minimal flow your agent could run (clip-factory pipeline, single `cut` call):

```
1. provider_menu_summary          → confirm ffmpeg is configured
2. get_pipeline_manifest          → ("clip-factory") learn the stages
3. read om://guide/agent-guide    → learn HOW to orchestrate (Rule Zero, etc.)
4. get_tool_info("video_trimmer") → learn input_schema (operation: cut/speed/concat)
5. execute_tool("video_trimmer", {
       "operation": "cut",
       "input_path": "/abs/source.mp4",
       "output_path": "/abs/clip1.mp4",
       "start_seconds": 12, "end_seconds": 27
   })
6. write_checkpoint(project_id, "compose", "completed", {...})  → persist state
```

## Long-running tools

Tools like `video_compose` / Remotion can run for minutes. Two ways to handle this:

- **`execute_tool`** dispatches to a worker thread so the server's message loop
  never blocks, and reports progress via MCP notifications. **Clients must allow
  generous call timeouts** (≥ 600s for renders).
- **`submit_tool_job` + `get_job_status`** (preferred for long jobs): submit and
  get a `job_id` at once, then poll until `status == "succeeded"`. No long-held
  call. Example:

  ```
  job = submit_tool_job("video_compose", {operation:"render", ...})
  # ...do other work...
  while get_job_status(job.job_id).status != "succeeded": sleep(2)
  result = get_job_status(job.job_id).result
  ```

Jobs live in server memory — a restart loses in-flight and completed jobs. This
is the right trade for a single-process stdio server; durable persistence would
need a backing store.

## Security

- **Sandboxed resources:** `om://` URIs resolve only under allowlisted doc/skill
  directories; traversal and absolute paths are rejected (tested).
- **Secret scrubbing:** `execute_tool` redacts any value whose key looks like a
  secret (`api_key`, `token`, `password`, ...) before returning to the client.
- **Local bind by default:** networked transports bind `127.0.0.1`; set `host`
  explicitly only if you need remote access.
- **Key isolation:** the server reads `.env` for API keys but never echoes key
  *values* to clients.

## Configuration

`config.yaml` → `mcp:` block:

```yaml
mcp:
  transport: stdio        # stdio | sse | streamable-http
  host: 127.0.0.1         # for networked transports
  port: 8765
```

Overrides via CLI flags (`--transport/--host/--port`) or env (`OM_MCP_TRANSPORT`,
`OM_MCP_HOST`, `OM_MCP_PORT`), in that priority order.
