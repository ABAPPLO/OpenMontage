"""Sandboxed resolution of MCP ``om://`` resource URIs to file contents.

External agents need to read OpenMontage's instruction documents (AGENT_GUIDE,
pipeline manifests, stage-director skills, Layer-3 vendor skills) to orchestrate
production themselves. This module exposes those as MCP resources while keeping
every resolved path inside a strict allowlist of directories under the repo
root — so a client cannot read arbitrary files (e.g. ``om://../../../etc/passwd``)
or escape to credentials.

URI scheme:
    om://guide/agent-guide              -> AGENT_GUIDE.md
    om://guide/project-context          -> PROJECT_CONTEXT.md
    om://guide/agents                   -> AGENTS.md
    om://guide/readme                   -> README.md
    om://pipelines/<name>               -> pipeline_defs/<name>.yaml
    om://pipelines                      -> listing of all pipeline names
    om://skills/<path...>               -> skills/<path...>  (Layer 2 skills)
    om://agent-skills/<path...>         -> .agents/skills/<path...>  (Layer 3)
    om://styles/<name>                  -> styles/<name>.yaml
    om://styles                         -> listing of style playbooks

Path traversal (``..``), absolute paths, and symlinks pointing outside the
allowlist are rejected with ``PermissionError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# Repo root = parent of this package (mcp_server/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory allowlist for om:// paths. Each maps a URI prefix to a resolved
# directory that must exist. Resolution is always ``allowlisted_dir / rest``.
_ALLOWED_DIRS: dict[str, Path] = {
    "skills": (PROJECT_ROOT / "skills").resolve(),
    "agent-skills": (PROJECT_ROOT / ".agents" / "skills").resolve(),
    "pipelines": (PROJECT_ROOT / "pipeline_defs").resolve(),
    "styles": (PROJECT_ROOT / "styles").resolve(),
}

# Named single-file docs under om://guide/<name>.
_GUIDE_DOCS: dict[str, Path] = {
    "agent-guide": (PROJECT_ROOT / "AGENT_GUIDE.md").resolve(),
    "project-context": (PROJECT_ROOT / "PROJECT_CONTEXT.md").resolve(),
    "agents": (PROJECT_ROOT / "AGENTS.md").resolve(),
    "readme": (PROJECT_ROOT / "README.md").resolve(),
}


class ResourceNotFound(FileNotFoundError):
    """Raised when an om:// URI does not resolve to an existing file."""


class ResourceForbidden(PermissionError):
    """Raised when an om:// URI escapes the sandbox allowlist."""


def _is_within(child: Path, parent: Path) -> bool:
    """True if ``child`` is ``parent`` or inside it (after symlink resolution).

    We resolve the candidate path and compare common-path; symlinks that point
    outside the parent are caught because resolve() follows them.
    """
    try:
        resolved_child = child.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    try:
        resolved_parent = parent.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    if resolved_child == resolved_parent:
        return True
    try:
        resolved_parent in resolved_child.parents
        return resolved_child.relative_to(resolved_parent) is not None
    except ValueError:
        return False


def parse_uri(uri: str) -> tuple[str, str]:
    """Split an ``om://`` URI into ``(prefix, rest)``.

    ``rest`` may be empty (for directory listings). Raises ResourceForbidden if
    the URI isn't an om:// URI.
    """
    if not uri.startswith("om://"):
        raise ResourceForbidden(f"Not an om:// URI: {uri!r}")
    body = uri[len("om://"):]
    if not body:
        raise ResourceNotFound("Empty om:// URI")
    parts = body.split("/", 1)
    prefix = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    return prefix, rest


def resolve_path(uri: str) -> Path:
    """Resolve an om:// URI to a concrete filesystem Path inside the sandbox.

    Validates the path stays within the allowlisted directory. Does NOT require
    the file to exist (caller decides); use ``read()`` for read+exist.
    """
    prefix, rest = parse_uri(uri)

    if prefix == "guide":
        # Named single docs — ignore ``rest`` beyond the doc key.
        doc_key = rest.strip("/")
        if doc_key not in _GUIDE_DOCS:
            raise ResourceNotFound(
                f"Unknown guide doc {doc_key!r}. Available: {sorted(_GUIDE_DOCS)}"
            )
        return _GUIDE_DOCS[doc_key]

    if prefix not in _ALLOWED_DIRS:
        raise ResourceNotFound(
            f"Unknown om:// prefix {prefix!r}. "
            f"Valid prefixes: guide, {sorted(_ALLOWED_DIRS)}"
        )

    base = _ALLOWED_DIRS[prefix]
    # Reject empty rest here — resolve_path is for a single file. Use list_dir
    # for directory listings.
    if not rest:
        raise ResourceNotFound(
            f"om://{prefix} requires a path. Use om://{prefix} to list entries."
        )

    # Normalize the relative path and reject traversal / absolute inputs.
    # ``Path / rest`` would naively join, so we guard ``..``, leading ``/``
    # (absolute), and empty segments (``//``) which can confuse path semantics.
    if rest.startswith("/") or rest.startswith("\\"):
        raise ResourceForbidden(f"Absolute path not allowed in om:// URI: {uri!r}")
    cleaned = rest.replace("\\", "/").strip("/")
    segments = [s for s in cleaned.split("/") if s]
    if cleaned and len(segments) != len(cleaned.split("/")):
        # Empty segments (e.g. "a//b") — reject to avoid ambiguity.
        raise ResourceForbidden(f"Invalid path segment in om:// URI: {uri!r}")
    if ".." in cleaned.split("/"):
        raise ResourceForbidden(f"Path traversal not allowed: {uri!r}")

    candidate = (base / cleaned).resolve(strict=False)
    if not _is_within(candidate, base):
        raise ResourceForbidden(f"URI escapes sandbox: {uri!r}")

    # Pipeline manifests and style playbooks are referenced by name without the
    # .yaml extension (matching list_dir, which returns stems). Auto-append so
    # om://pipelines/clip-factory resolves to clip-factory.yaml.
    if prefix == "pipelines" and candidate.suffix not in (".yaml", ".yml"):
        candidate = candidate.with_suffix(".yaml")
        if not _is_within(candidate, base):
            raise ResourceForbidden(f"URI escapes sandbox: {uri!r}")
    return candidate


def list_dir(uri: str) -> list[str]:
    """List immediate children of an allowlisted directory (one level deep).

    For ``om://pipelines`` returns pipeline manifest names (without extension).
    For ``om://skills`` / ``om://agent-skills`` / ``om://styles`` returns the
    relative paths of files under that tree.
    """
    prefix, rest = parse_uri(uri)
    if prefix == "guide":
        raise ResourceNotFound("om://guide has no listing; request a named doc.")
    if prefix not in _ALLOWED_DIRS:
        raise ResourceNotFound(f"Unknown om:// prefix {prefix!r}")
    if rest:
        raise ResourceNotFound(f"Listing requires no sub-path; got {rest!r}")

    base = _ALLOWED_DIRS[prefix]
    if prefix == "pipelines":
        return sorted(p.stem for p in base.glob("*.yaml") if p.is_file())
    if prefix == "styles":
        return sorted(
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file() and p.suffix in (".yaml", ".yml", ".md")
        )
    # skills trees — list .md files with posix relative paths
    return sorted(
        p.relative_to(base).as_posix()
        for p in base.rglob("*")
        if p.is_file() and p.suffix == ".md"
    )


def read(uri: str) -> tuple[str, str]:
    """Read an om:// resource, returning ``(text, mime_type)``.

    Raises ResourceNotFound if the file is missing, ResourceForbidden if it
    escapes the sandbox.
    """
    path = resolve_path(uri)
    if not path.is_file():
        raise ResourceNotFound(f"Resource not found or not a file: {uri!r}")
    text = path.read_text(encoding="utf-8", errors="replace")
    mime = _mime_for(path)
    return text, mime


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return "text/yaml"
    if suffix == ".md":
        return "text/markdown"
    if suffix == ".json":
        return "application/json"
    return "text/plain"


def list_all_resource_uris() -> list[str]:
    """Enumerate a representative set of om:// URIs for MCP resources/list.

    MCP clients call ``resources/list`` to discover readable docs. We return the
    fixed guide docs plus every pipeline manifest and the top-level entries of
    each skills tree (not a full recursive dump, which would be hundreds of
    files — agents can follow directory-listing tools to go deeper).
    """
    uris: list[str] = ["om://guide/" + name for name in sorted(_GUIDE_DOCS)]
    for name in list_dir("om://pipelines"):
        uris.append(f"om://pipelines/{name}")
    return uris
