"""Run identity and manifests for the versioned results store.

Every experiment run gets its own directory under
``ARTIFACTS_DIR/runs/<experiment>/<run_id>/`` holding an append-only
``results.jsonl`` and a ``manifest.json`` that records everything needed to
reproduce the run: the git commit (and whether the tree was dirty), the full
prompt texts keyed by ``prompt_hash``, the model list, the sampling spec, and
the serving backend. Rows produced by different code versions or prompt
wordings are therefore always separable after the fact — the failure mode the
legacy single-log store could not prevent.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from . import config


def git_sha(short: bool = True) -> str:
    """Current commit of the code repo, with a ``-dirty`` suffix when the
    working tree has uncommitted changes. Returns ``"nogit"`` outside a repo
    (e.g. a bare deployment on the cluster)."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            cwd=config.REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "nogit"


def new_run_id() -> str:
    """``YYYYMMDD-HHMMSS-<shortsha>`` — sortable, human-readable, and tied to
    the code version that produced the run."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sha = git_sha().replace("-dirty", "d")
    return f"{stamp}-{sha}"


def run_dir(experiment: str, run_id: str) -> Path:
    return config.ARTIFACTS_DIR / "runs" / experiment / run_id


def write_manifest(directory: Path, payload: dict[str, Any],
                   *, filename: str = "manifest.json") -> Path:
    """Write a manifest for a run (or a sidecar manifest for a standalone
    artifact — pass ``filename``). ``payload`` carries the experiment-specific
    fields; the standard provenance block is added here."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_ts": time.time(),
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "backend": config.backend_name(),
        "base_url_host": config.base_url_host(),
        **payload,
    }
    path = directory / filename
    path.write_text(json.dumps(manifest, indent=2, default=str),
                    encoding="utf-8")
    return path
