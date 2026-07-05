"""OpenMontage MCP Server.

Exposes OpenMontage's tool registry, pipeline/checkpoint libraries, and agent
instruction documents over the Model Context Protocol, so external agents (any
language, cross-process) can drive video production.

See mcp_server/README.md for setup and usage.
"""

from mcp_server.server import mcp

__all__ = ["mcp"]
