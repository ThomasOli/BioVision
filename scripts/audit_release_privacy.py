#!/usr/bin/env python3
"""Block releases that would publish local app state or trained artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath


FORBIDDEN_TOP_LEVEL = {
    ".claude",
    ".venv",
    "corrected_images",
    "debug",
    "images",
    "labels",
    "locales",
    "model_training",
    "models",
    "resources",
    "runs",
    "sessions",
    "venv",
    "xml",
}
FORBIDDEN_SUFFIXES = {".dat", ".pt", ".pth"}
FORBIDDEN_ROOT_FILES = {
    "BioVision.exe",
    "LICENSE.electron.txt",
    "LICENSES.chromium.html",
    "Uninstall BioVision.exe",
    "resources.pak",
    "snapshot_blob.bin",
    "v8_context_snapshot.bin",
    "vk_swiftshader_icd.json",
}


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [entry.decode("utf-8", errors="surrogateescape") for entry in result.stdout.split(b"\0") if entry]


def forbidden_reason(path_text: str) -> str | None:
    path = PurePosixPath(path_text)
    if path.parts and path.parts[0] in FORBIDDEN_TOP_LEVEL:
        return f"forbidden local/runtime directory: {path.parts[0]}"
    if len(path.parts) == 1 and path.name in FORBIDDEN_ROOT_FILES:
        return "unpacked Electron runtime file"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return f"trained/model artifact: {path.suffix.lower()}"
    return None


def main() -> int:
    violations = [
        (path, reason)
        for path in tracked_paths()
        if (reason := forbidden_reason(path)) is not None
    ]
    if violations:
        print("Release privacy audit failed; tracked local artifacts were found:", file=sys.stderr)
        for path, reason in violations:
            print(f"  - {path} ({reason})", file=sys.stderr)
        return 1
    print("Release privacy audit passed: no local app data or trained artifacts are tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
