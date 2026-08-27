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
import json
import os
import re
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


BUILD_SCRIPT = os.path.join(REPO_ROOT, "scripts", "build-python-backend.js")
ELECTRON_MAIN = os.path.join(REPO_ROOT, "electron", "main.ts")
ELECTRON_BUILDER = os.path.join(REPO_ROOT, "electron-builder.json5")
PACKAGE_JSON = os.path.join(REPO_ROOT, "package.json")
REQUIREMENTS = os.path.join(BACKEND_DIR, "requirements.txt")


def _hidden_imports() -> set[str]:
    """Modules PyInstaller is explicitly told to bundle."""
    with open(BUILD_SCRIPT, "r", encoding="utf-8") as f:
        return set(re.findall(r'"--hidden-import",\s*"([\w.]+)"', f.read()))


def _electron_dev_script_map() -> dict[str, str]:
    """The dev-mode name -> relative source path map in resolveBundledScript."""
    with open(ELECTRON_MAIN, "r", encoding="utf-8") as f:
        source = f.read()
    block = source.split("const scriptMap: Record<string, string> = {", 1)
    if len(block) < 2:
        raise AssertionError("resolveBundledScript scriptMap not found in electron/main.ts")
    return dict(re.findall(r'(\w+):\s*"([^"]+\.py)"', block[1].split("};", 1)[0]))


class TestBundlingContract:
    """Every dispatch target must be reachable in dev *and* in the bundle.

    PyInstaller cannot see modules reached only through `runpy`, so a script
    added to SCRIPT_MAP without a matching `--hidden-import` produces an
    installer that fails only when a user opens that one feature. The dev-mode
    map in electron/main.ts has the same failure mode in reverse.
    """

    @pytest.mark.parametrize("script_name,dotted", sorted(SCRIPT_MAP.items()))
    def test_every_target_is_a_pyinstaller_hidden_import(self, script_name, dotted):
        assert dotted in _hidden_imports(), (
            f"SCRIPT_MAP['{script_name}'] -> '{dotted}' is missing a "
            f'"--hidden-import" in scripts/build-python-backend.js, so it would '
            f"not be bundled into the packaged backend."
        )

    @pytest.mark.parametrize("script_name", sorted(SCRIPT_MAP))
    def test_every_target_has_a_dev_mode_path(self, script_name):
        dev_map = _electron_dev_script_map()
        assert script_name in dev_map, (
            f"'{script_name}' is dispatchable but has no dev-mode entry in "
            f"resolveBundledScript(); dev runs would guess the wrong path."
        )
        full = os.path.join(BACKEND_DIR, dev_map[script_name].replace("/", os.sep))
        assert os.path.isfile(full), f"dev-mode path for '{script_name}' missing: {full}"

    def test_dev_map_has_no_entries_the_cli_cannot_dispatch(self):
        unknown = sorted(set(_electron_dev_script_map()) - set(SCRIPT_MAP))
        assert not unknown, (
            f"electron/main.ts can request {unknown}, but cli.py cannot dispatch "
            f"them; packaged builds would fail with 'Unknown script'."
        )

    def test_selfcheck_imports_every_dispatch_target(self):
        """The check the packaged build runs; proves the map is importable."""
        result = subprocess.run(
            [sys.executable, CLI_PATH, "--selfcheck"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload["ok"], payload["failures"]
        assert set(payload["checked"]) == set(SCRIPT_MAP)

    def test_yolo_world_text_encoder_is_frozen_with_its_data(self):
        with open(BUILD_SCRIPT, "r", encoding="utf-8") as handle:
            build_source = handle.read()
        assert "clip" in _hidden_imports()
        assert re.search(r'"--collect-all",\s*"clip"', build_source)

    def test_release_builds_prepare_and_package_pinned_annotation_assets(self):
        with open(REQUIREMENTS, "r", encoding="utf-8") as handle:
            requirements = handle.read()
        assert (
            "clip @ git+https://github.com/ultralytics/CLIP.git"
            "@68dce32140994dfcb645a1320c4ebdc034fc19fd"
        ) in requirements

        with open(ELECTRON_BUILDER, "r", encoding="utf-8") as handle:
            builder = handle.read()
        assert '"from": "yolov8s-worldv2.pt"' in builder
        assert '"from": "ViT-B-32.pt"' in builder
        assert '"from": "sam2_b.pt"' in builder

        with open(PACKAGE_JSON, "r", encoding="utf-8") as handle:
            scripts = json.load(handle)["scripts"]
        for name in ("dist", "dist:win", "dist:mac", "dist:linux", "publish", "publish:win", "publish:mac", "publish:linux"):
            assert scripts[name].startswith(
                "npm run release:privacy && npm run release:assets &&"
            ), name
