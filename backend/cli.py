"""
BioVision CLI dispatcher.

Single entry point for all backend scripts. PyInstaller bundles this once,
sharing all heavy dependencies (torch, ultralytics, dlib, opencv, etc.)
instead of duplicating them across 13 separate executables.

Usage:
    biovision_backend <script_name> [args...]

Example:
    biovision_backend predict /path/to/root model_name image.jpg
    biovision_backend hardware_probe
    biovision_backend super_annotator
"""
import sys
import runpy
import os

SCRIPT_MAP = {
    "prepare_dataset": "data.prepare_dataset",
    "prepare_imported_dlib_dataset": "data.prepare_imported_dlib_dataset",
    "validate_dlib_xml": "data.validate_dlib_xml",
    "audit_dataset": "data.audit_dataset",
    "export_yolo_dataset": "data.export_yolo_dataset",
    "train_shape_model": "training.train_shape_model",
    "train_cnn_model": "training.train_cnn_model",
    "predict": "inference.predict",
    "predict_worker": "inference.predict_worker",
    "shape_tester": "inference.shape_tester",
    "list_cnn_variants": "inference.list_cnn_variants",
    "detect_specimen": "detection.detect_specimen",
    "super_annotator": "annotation.super_annotator",
    "hardware_probe": "hardware_probe",
}

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <script_name> [args...]", file=sys.stderr)
    print(f"Available: {', '.join(sorted(SCRIPT_MAP.keys()))}", file=sys.stderr)
    sys.exit(1)

script_name = sys.argv[1]

# Ensure the backend package is importable
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

if script_name == "--selfcheck":
    # Import every dispatch target and report the result as JSON.
    #
    # PyInstaller cannot see modules reached only through runpy, so each one
    # needs an explicit --hidden-import in scripts/build-python-backend.js. A
    # missing entry produces an executable that looks fine until a user picks
    # that feature. This check turns that into a build-time failure.
    import importlib
    import json

    failures = {}
    for name, module in sorted(SCRIPT_MAP.items()):
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report every failure kind
            failures[name] = f"{type(exc).__name__}: {exc}"
    payload = {
        "ok": not failures,
        "frozen": bool(getattr(sys, "frozen", False)),
        "checked": sorted(SCRIPT_MAP),
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    sys.exit(0 if payload["ok"] else 1)

if script_name not in SCRIPT_MAP:
    print(f"Unknown script: {script_name}", file=sys.stderr)
    print(f"Available: {', '.join(sorted(SCRIPT_MAP.keys()))}", file=sys.stderr)
    sys.exit(1)

# Strip dispatcher + script name so the target sees only its own args
sys.argv = [script_name] + sys.argv[2:]

# Run the module as __main__ so if __name__ == "__main__" blocks execute
runpy.run_module(SCRIPT_MAP[script_name], run_name="__main__", alter_sys=True)
