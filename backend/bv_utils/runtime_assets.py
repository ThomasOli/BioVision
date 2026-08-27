"""Resolve model assets consistently in development and packaged builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_runtime_asset(filename: str) -> str | None:
    """Return an existing model asset path without depending on the CWD.

    Electron installs the frozen backend under ``resources/python`` and model
    checkpoints directly under ``resources``.  Development runs continue to
    support assets in the repository root.  ``BIOVISION_MODEL_ASSET_DIR`` is
    retained as the explicit override used by tests and custom deployments.
    """
    candidates: list[Path] = []
    configured_root = os.environ.get("BIOVISION_MODEL_ASSET_DIR")
    if configured_root:
        candidates.append(Path(configured_root) / filename)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent.parent / filename)

    backend_root = Path(__file__).resolve().parent.parent
    candidates.extend((Path.cwd() / filename, backend_root.parent / filename))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return str(resolved)
    return None
