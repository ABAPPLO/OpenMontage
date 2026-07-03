#!/usr/bin/env bash
# OpenMontage MCP server launcher.
#
# Resolves a usable Python (venv preferred) and dispatches to the right mode.
# Usage:
#   ./mcp-server.sh                  # stdio (default — for Claude Desktop etc.)
#   ./mcp-server.sh http [PORT]      # streamable-http, default port 8765
#   ./mcp-server.sh demo             # run the end-to-end smoke test (mcp_server/demo.py)
#   ./mcp-server.sh inspect          # launch the MCP Inspector GUI (web, port 6274)
#   ./mcp-server.sh install          # install the `mcp` SDK dependency into the venv
#
# All remaining args after a mode are forwarded to the underlying command, so you
# can also pass transport flags directly:
#   ./mcp-server.sh --transport sse --port 9000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- resolve a Python 3.10+ interpreter (venv first, then PATH) ----
find_python() {
  if [ -x ".venv/bin/python" ]; then
    echo ".venv/bin/python"
    return
  fi
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    echo "$VIRTUAL_ENV/bin/python"
    return
  fi
  for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      ver="$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 0.0)"
      major="${ver%%.*}"; minor="${ver#*.}"
      if [ "${major:-0}" -gt 3 ] || { [ "${major:-0}" -eq 3 ] && [ "${minor:-0}" -ge 10 ]; }; then
        echo "$cand"
        return
      fi
    fi
  done
  echo ""
}

PYTHON="$(find_python)"
if [ -z "$PYTHON" ]; then
  echo "ERROR: no Python 3.10+ found." >&2
  echo "Create a venv with: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# ---- check the mcp SDK is importable; offer install if not ----
if ! "$PYTHON" -c "import mcp" >/dev/null 2>&1; then
  echo "NOTE: 'mcp' package not importable in $PYTHON." >&2
  echo "      Run '$0 install' first, or: $PYTHON -m pip install 'mcp>=1.0'" >&2
  exit 1
fi

MODE="${1:-stdio}"
case "$MODE" in
  stdio)
    # Default: stdio transport for local MCP clients.
    exec "$PYTHON" -m mcp_server "$@"
    ;;

  http|streamable-http|sse)
    # Map short aliases to the canonical transport name the server accepts.
    case "$MODE" in
      http) TRANSPORT="streamable-http" ;;
      *)    TRANSPORT="$MODE" ;;
    esac
    shift
    PORT=""
    # Collect a leading port number if given positionally: `http 9000`
    if [ "${1:-}" != "" ] && [[ "${1:-}" =~ ^[0-9]+$ ]]; then
      PORT="$1"; shift
    fi
    ARGS=(--transport "$TRANSPORT")
    [ -n "$PORT" ] && ARGS+=(--port "$PORT")
    ARGS+=("$@")
    exec "$PYTHON" -m mcp_server "${ARGS[@]}"
    ;;

  demo|smoke)
    # End-to-end test: generates a test video, slices it (sync + async).
    exec "$PYTHON" mcp_server/demo.py "$@"
    ;;

  inspect|inspector|gui)
    # Launch the MCP Inspector web GUI against this server.
    if ! command -v mcp-inspector >/dev/null 2>&1; then
      echo "ERROR: mcp-inspector not found. Install with:" >&2
      echo "  npm install -g @modelcontextprotocol/inspector" >&2
      exit 1
    fi
    exec mcp-inspector "$PYTHON" -m mcp_server "$@"
    ;;

  install)
    echo "==> Installing MCP SDK into $PYTHON ..."
    "$PYTHON" -m pip install "mcp>=1.0" "$@"
    echo "==> Done. Run '$0' to start the server."
    ;;

  -h|--help|help)
    cat <<EOF
OpenMontage MCP server launcher.

Usage:
  $0                       Start server (stdio, default)
  $0 http [PORT]           Start in streamable-http mode (default port 8765)
  $0 sse [PORT]            Start in SSE mode
  $0 demo                  Run the end-to-end smoke test (auto-generates a test video)
  $0 inspect               Launch MCP Inspector web GUI (port 6274)
  $0 install               Install the 'mcp' SDK into the venv
  $0 --transport stdio ... Pass raw flags through to 'python -m mcp_server'

Python resolved: $PYTHON
EOF
    ;;

  *)
    # Anything else (e.g. --transport ...) is forwarded verbatim.
    exec "$PYTHON" -m mcp_server "$@"
    ;;
esac
