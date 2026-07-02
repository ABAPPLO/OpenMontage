"""Tests for mcp_server.resources — sandboxed om:// URI resolution.

The security-critical part: path-traversal, absolute-path, and out-of-sandbox
attempts must be rejected. Valid reads of shipped docs must succeed.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server import resources as R
from mcp_server.resources import ResourceForbidden, ResourceNotFound


# ---------------------------------------------------------------------------
# Valid reads
# ---------------------------------------------------------------------------

def test_read_agent_guide():
    text, mime = R.read("om://guide/agent-guide")
    assert mime == "text/markdown"
    assert "Rule Zero" in text or "AGENT_GUIDE" in text or len(text) > 1000


def test_read_project_context():
    text, mime = R.read("om://guide/project-context")
    assert mime == "text/markdown"
    assert len(text) > 100


def test_read_pipeline_manifest():
    text, mime = R.read("om://pipelines/clip-factory")
    assert mime == "text/yaml"
    assert "stages:" in text or "clip-factory" in text


def test_read_skill():
    text, mime = R.read("om://skills/meta/reviewer.md")
    assert mime == "text/markdown"
    assert len(text) > 50


def test_list_pipelines_includes_known():
    names = R.list_dir("om://pipelines")
    assert "clip-factory" in names
    assert "animated-explainer" in names
    assert all(".yaml" not in n for n in names)  # stems, no extension


def test_list_all_resource_uris():
    uris = R.list_all_resource_uris()
    assert "om://guide/agent-guide" in uris
    assert any(u.startswith("om://pipelines/") for u in uris)


def test_mime_for_json():
    assert R._mime_for(Path("x.json")) == "application/json"
    assert R._mime_for(Path("x.yaml")) == "text/yaml"


# ---------------------------------------------------------------------------
# Security: path traversal / escape must be rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "malicious_uri",
    [
        "om://skills/../../etc/passwd",
        "om://skills/../.env",
        "om://agent-skills/../../../etc/shadow",
        "om://skills/../../../tmp/secret",
        "om://pipelines/../.env",
    ],
)
def test_path_traversal_rejected(malicious_uri):
    with pytest.raises(ResourceForbidden):
        R.resolve_path(malicious_uri)


@pytest.mark.parametrize(
    "bad_uri",
    [
        "https://example.com/foo",   # wrong scheme
        "om://",                      # empty
        "/etc/passwd",                # not om:// at all
    ],
)
def test_non_om_or_empty_uri_rejected(bad_uri):
    with pytest.raises((ResourceForbidden, ResourceNotFound)):
        R.resolve_path(bad_uri)


def test_unknown_prefix_rejected():
    with pytest.raises(ResourceNotFound):
        R.resolve_path("om://bogus/x")


def test_unknown_guide_doc_rejected():
    with pytest.raises(ResourceNotFound):
        R.resolve_path("om://guide/does-not-exist")


def test_missing_file_raises_not_found():
    with pytest.raises(ResourceNotFound):
        R.read("om://skills/meta/does-not-exist.md")


def test_absolute_subpath_rejected():
    with pytest.raises(ResourceForbidden):
        R.resolve_path("om://skills//etc/passwd")


def test_resolve_stays_within_sandbox():
    """A legitimate nested skill resolves under the skills dir."""
    path = R.resolve_path("om://skills/meta/reviewer.md")
    base = R._ALLOWED_DIRS["skills"]
    assert base in path.parents or path == base
