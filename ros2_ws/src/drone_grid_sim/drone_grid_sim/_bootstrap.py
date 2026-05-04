"""Bootstrap shared by every node: make ``closeloop`` and ``agent_diy`` importable
and install the kaiwudrl/common_python/tools stubs.

ROS launches a Python interpreter from the colcon install tree, so the kaiwu
repo root isn't on ``sys.path``. We resolve it via the ``KAIWU_REPO`` env var,
falling back to the value baked at build time (``DEFAULT_REPO_ROOT`` below)."""

from __future__ import annotations

import os
import sys

# Adjust this if the repo lives elsewhere; ``KAIWU_REPO`` env var always wins.
DEFAULT_REPO_ROOT = "/root/workspace/kaiwu-drone"


def bootstrap() -> str:
    repo = os.environ.get("KAIWU_REPO", DEFAULT_REPO_ROOT)
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)

    # Install kaiwudrl / common_python / tools stand-ins so agent_diy imports.
    import closeloop.framework_stub  # noqa: F401
    return repo
