"""Entry point: ``python -m mcp_server``.

Picks transport/host/port from CLI flags, then env (OM_MCP_*), then config.yaml's
``mcp:`` block (default: stdio). Validates the chosen transport against what
FastMCP supports before launching.
"""

from __future__ import annotations

import argparse
import os
import sys

from lib.config_model import OpenMontageConfig


VALID_TRANSPORTS = ("stdio", "sse", "streamable-http")


def _resolve_settings(argv: list[str]) -> argparse.Namespace:
    """Merge CLI flags > env vars > config.yaml into a settings namespace."""
    cfg = OpenMontageConfig.load()

    parser = argparse.ArgumentParser(
        prog="python -m mcp_server",
        description="OpenMontage MCP server — expose tools/pipelines/docs to external agents.",
    )
    parser.add_argument(
        "--transport",
        choices=VALID_TRANSPORTS,
        default=os.environ.get("OM_MCP_TRANSPORT", cfg.mcp.transport),
        help="MCP transport (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OM_MCP_HOST", cfg.mcp.host),
        help="Bind host for networked transports (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("OM_MCP_PORT", cfg.mcp.port)),
        help="Bind port for networked transports (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    settings = _resolve_settings(sys.argv[1:] if argv is None else argv)

    # FastMCP reads host/port from its Settings object. For networked transports
    # we override the singleton before run() so the config/env values win.
    from mcp_server.server import mcp

    if settings.transport in ("sse", "streamable-http"):
        mcp.settings.host = settings.host
        mcp.settings.port = settings.port

    # stdio is the local default — no network surface. Networked transports bind
    # to 127.0.0.1 by default (config.yaml) so they're reachable only locally.
    print(
        f"OpenMontage MCP server starting: transport={settings.transport}"
        + (f" host={settings.host} port={settings.port}" if settings.transport != "stdio" else ""),
        file=sys.stderr,
    )
    mcp.run(transport=settings.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
