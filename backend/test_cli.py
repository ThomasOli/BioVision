"""
Tests for backend/cli.py — the PyInstaller dispatcher.

Goal: catch typos in SCRIPT_MAP and regressions in the argv / exit-code
contract. We intentionally do NOT `import backend.cli`, because the module
runs `runpy.run_module()` at import time (no `if __name__ == "__main__"`
guard), which would pull in heavy ML deps.

Strategy:
  - Parse SCRIPT_MAP statically via `ast` to inspect it without executing.
  - Drive the error paths (no args / unknown script) via subprocess, which
    is exactly how Electron invokes the backend in production.
"""

import ast
import os
import subprocess
import sys

import pytest


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BACKEND_DIR)
CLI_PATH = os.path.join(BACKEND_DIR, "cli.py")


def _extract_script_map() -> dict[str, str]:
    """Pull SCRIPT_MAP out of cli.py statically, without executing the module."""
    with open(CLI_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=CLI_PATH)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SCRIPT_MAP" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("SCRIPT_MAP not found in cli.py")


SCRIPT_MAP = _extract_script_map()


class TestScriptMap:
    def test_script_map_is_non_empty(self):
        assert SCRIPT_MAP, "SCRIPT_MAP must not be empty"

    def test_script_names_are_unique(self):
        assert len(SCRIPT_MAP) == len(set(SCRIPT_MAP.keys()))

    def test_targets_are_unique(self):
        # Two script names pointing to the same module would be dead weight.
        targets = list(SCRIPT_MAP.values())
        assert len(targets) == len(set(targets)), (
            f"Duplicate targets in SCRIPT_MAP: {targets}"
        )

    @pytest.mark.parametrize("script_name,dotted", sorted(SCRIPT_MAP.items()))
    def test_every_target_resolves_to_a_real_file(self, script_name, dotted):
        """Each SCRIPT_MAP value must point to an actual .py file under backend/.

        This catches typos like `infernece.predict` before PyInstaller bundles
        a broken binary.
        """
        rel = dotted.replace(".", os.sep) + ".py"
        full = os.path.join(BACKEND_DIR, rel)
        assert os.path.isfile(full), (
            f"SCRIPT_MAP['{script_name}'] -> '{dotted}' "
            f"but no file exists at {full}"
        )


class TestDispatcherErrorPaths:
    """Drive cli.py via subprocess for the paths that exit before runpy."""

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, CLI_PATH, *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )

    def test_no_args_exits_with_usage(self):
        result = self._run_cli()
        assert result.returncode == 1
        assert "Usage:" in result.stderr
        assert "Available:" in result.stderr

    def test_unknown_script_lists_available(self):
        result = self._run_cli("not_a_real_script")
        assert result.returncode == 1
        assert "Unknown script: not_a_real_script" in result.stderr
        assert "Available:" in result.stderr
        # Spot-check that a known good name appears in the availability hint.
        assert "predict" in result.stderr
