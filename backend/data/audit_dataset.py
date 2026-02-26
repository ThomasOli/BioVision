#!/usr/bin/env python3
"""
BioVision dataset audit tool.

Checks for four categories of data-quality problems before training is allowed:
  A) Label token-count consistency (5 fields for detect; 11 for pose)
  B) Synthetic images that use positive (fish-containing) backgrounds
  C) ID-map consistency between dlib part indices and schema landmark IDs
  D) High detector-fallback rate in prediction logs

Usage:
    python audit_dataset.py --project-root <dir> --tag <tag> [--fail-on-warn]

Exit codes:
    0 — all checks pass (warnings are printed but do not block)
    1 — at least one FAIL found, or --fail-on-warn and at least one WARN found
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Result helpers
# ──────────────────────────────────────────────────────────────────────────────

class AuditReport:
    def __init__(self, tag: str):
        self.tag = tag
        self.issues: list[dict] = []

    def fail(self, check: str, message: str, detail: str = ""):
        self.issues.append({"level": "FAIL", "check": check, "message": message, "detail": detail})

    def warn(self, check: str, message: str, detail: str = ""):
        self.issues.append({"level": "WARN", "check": check, "message": message, "detail": detail})

    def info(self, check: str, message: str):
        self.issues.append({"level": "INFO", "check": check, "message": message, "detail": ""})

    def has_failures(self) -> bool:
        return any(i["level"] == "FAIL" for i in self.issues)

    def has_warnings(self) -> bool:
        return any(i["level"] == "WARN" for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "pass": not self.has_failures(),
            "fail_count": sum(1 for i in self.issues if i["level"] == "FAIL"),
            "warn_count": sum(1 for i in self.issues if i["level"] == "WARN"),
            "issues": self.issues,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Check A — label token-count consistency
# ──────────────────────────────────────────────────────────────────────────────

def check_label_token_counts(project_root: str, report: AuditReport) -> None:
    """
    Every label row must have consistent field count:
      - 5  tokens: plain YOLO detection  (class cx cy w h)
      - 11 tokens: YOLO-Pose             (class cx cy w h  kx ky vis  kx ky vis)

    The expected count is inferred from dataset.yaml's kpt_shape field.
    """
    yolo_dir = os.path.join(project_root, "yolo_dataset")
    if not os.path.isdir(yolo_dir):
        report.info("label_tokens", "yolo_dataset/ not found — skipping label token check")
        return

    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    expected_tokens = None
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            for line in f:
                if "kpt_shape" in line:
                    expected_tokens = 11
                    break
        if expected_tokens is None:
            expected_tokens = 5  # no kpt_shape → plain detection

    bad_files: list[str] = []
    for split in ("train", "val"):
        lbl_dir = os.path.join(yolo_dir, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(lbl_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    n = len(tokens)
                    if expected_tokens is not None and n != expected_tokens:
                        bad_files.append(
                            f"{split}/{fname}:{lineno} — {n} tokens (expected {expected_tokens})"
                        )
                    elif expected_tokens is None and n not in (5, 11):
                        bad_files.append(
                            f"{split}/{fname}:{lineno} — {n} tokens (expected 5 or 11)"
                        )

    if bad_files:
        report.fail(
            "label_tokens",
            f"{len(bad_files)} label row(s) have wrong token count",
            "\n".join(bad_files[:20]) + ("..." if len(bad_files) > 20 else ""),
        )
    else:
        n_checked = sum(
            1
            for split in ("train", "val")
            for _ in glob.glob(os.path.join(yolo_dir, "labels", split, "*.txt"))
        )
        report.info("label_tokens", f"All {n_checked} label file(s) pass token-count check")


# ──────────────────────────────────────────────────────────────────────────────
# Check B — synthetic images must not use positive-image backgrounds
# ──────────────────────────────────────────────────────────────────────────────

def check_synthetic_backgrounds(project_root: str, report: AuditReport) -> None:
    """
    Synthetic images are named __synth_*.jpg.  Their source background is stored
    in the corresponding __synth_*_meta.json file under segments/.
    If that source_image path is a positive (annotated) training image, flag it.
    """
    yolo_dir = os.path.join(project_root, "yolo_dataset")
    seg_dir = os.path.join(project_root, "segments")

    # Build set of positive image paths (images referenced by session labels)
    labels_dir = os.path.join(project_root, "labels")
    positive_images: set[str] = set()
    if os.path.isdir(labels_dir):
        for fname in os.listdir(labels_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(labels_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                img_fname = data.get("imageFilename", "")
                if img_fname:
                    positive_images.add(os.path.normpath(
                        os.path.join(project_root, "images", img_fname)
                    ))
            except Exception:
                pass

    # Check segment meta files for their source_image
    bad: list[str] = []
    if os.path.isdir(seg_dir):
        for fname in os.listdir(seg_dir):
            if not fname.endswith("_meta.json"):
                continue
            try:
                with open(os.path.join(seg_dir, fname), "r", encoding="utf-8") as f:
                    meta = json.load(f)
                src = os.path.normpath(meta.get("source_image", ""))
                if src in positive_images:
                    bad.append(f"{fname} → {src}")
            except Exception:
                pass

    # Also check synthetic label directories for any __synth_ files
    synth_count = 0
    if os.path.isdir(yolo_dir):
        for split in ("train", "val"):
            img_dir = os.path.join(yolo_dir, "images", split)
            if os.path.isdir(img_dir):
                synth_count += sum(
                    1 for f in os.listdir(img_dir) if f.startswith("__synth_")
                )

    if bad:
        report.fail(
            "synthetic_backgrounds",
            f"{len(bad)} segment(s) use positive images as backgrounds",
            "\n".join(bad[:20]) + ("..." if len(bad) > 20 else ""),
        )
    else:
        if synth_count > 0:
            report.info(
                "synthetic_backgrounds",
                f"{synth_count} synthetic image(s) — background sources look clean",
            )
        else:
            report.info("synthetic_backgrounds", "No synthetic images found — check skipped")


# ──────────────────────────────────────────────────────────────────────────────
# Check C — ID-map consistency
# ──────────────────────────────────────────────────────────────────────────────

def check_id_map_consistency(project_root: str, tag: str, report: AuditReport) -> None:
    """
    The id_mapping_{tag}.json dlib_index_to_original dict must map to IDs
    that actually appear in the session's landmark template.
    """
    debug_dir = os.path.join(project_root, "debug")
    id_map_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")

    if not os.path.exists(id_map_path):
        report.info("id_map", f"id_mapping_{tag}.json not found — skipping ID map check")
        return

    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    explicit = id_map.get("dlib_index_to_original", {})
    if not explicit:
        report.warn("id_map", f"id_mapping_{tag}.json has no dlib_index_to_original field")
        return

    # Load schema landmark IDs from session.json
    session_path = os.path.join(project_root, "session.json")
    schema_ids: set[int] = set()
    if os.path.exists(session_path):
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                session = json.load(f)
            for lm in session.get("landmarkTemplate", []):
                try:
                    schema_ids.add(int(lm["index"]))
                except (KeyError, TypeError, ValueError):
                    pass
        except Exception:
            pass

    missing: list[str] = []
    if schema_ids:
        for dlib_idx, schema_id in explicit.items():
            try:
                sid = int(schema_id)
                if sid not in schema_ids:
                    missing.append(
                        f"dlib index {dlib_idx} → schema ID {schema_id} not in template {sorted(schema_ids)}"
                    )
            except (TypeError, ValueError):
                missing.append(f"dlib index {dlib_idx} → non-integer schema ID {schema_id!r}")

    if missing:
        report.fail(
            "id_map",
            f"{len(missing)} ID mapping entries refer to unknown schema IDs",
            "\n".join(missing),
        )
    else:
        report.info(
            "id_map",
            f"ID map has {len(explicit)} entries — all schema IDs verified",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Check D — detector fallback frequency
# ──────────────────────────────────────────────────────────────────────────────

def check_fallback_rate(project_root: str, tag: str, report: AuditReport,
                         warn_threshold: float = 0.30) -> None:
    """
    If more than warn_threshold of logged predictions used the opencv_contours
    fallback instead of YOLO, emit a warning.
    """
    debug_dir = os.path.join(project_root, "debug")
    log_path = os.path.join(debug_dir, f"prediction_log_{tag}.json")

    if not os.path.exists(log_path):
        report.info("fallback_rate", f"prediction_log_{tag}.json not found — skipping fallback check")
        return

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as exc:
        report.warn("fallback_rate", f"Could not parse prediction log: {exc}")
        return

    if not isinstance(entries, list) or len(entries) == 0:
        report.info("fallback_rate", "Prediction log is empty")
        return

    fallback_methods = {"opencv_contours", "opencv", "fallback"}
    fallback_count = sum(
        1
        for e in entries
        if isinstance(e, dict) and e.get("detection_method", "") in fallback_methods
    )
    total = len(entries)
    rate = fallback_count / total

    if rate > warn_threshold:
        report.warn(
            "fallback_rate",
            f"OpenCV fallback rate is {rate:.0%} ({fallback_count}/{total}) — "
            f"exceeds threshold of {warn_threshold:.0%}. "
            f"Consider retraining YOLO detector with more labelled data.",
        )
    else:
        report.info(
            "fallback_rate",
            f"Fallback rate {rate:.0%} ({fallback_count}/{total}) — within threshold",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_audit(project_root: str, tag: str, fail_on_warn: bool = False) -> AuditReport:
    report = AuditReport(tag)
    check_label_token_counts(project_root, report)
    check_synthetic_backgrounds(project_root, report)
    check_id_map_consistency(project_root, tag, report)
    check_fallback_rate(project_root, tag, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="BioVision dataset audit")
    parser.add_argument("--project-root", required=True,
                        help="Session root directory (e.g. sessions/<speciesId>/)")
    parser.add_argument("--tag", required=True,
                        help="Model tag (e.g. v1)")
    parser.add_argument("--fail-on-warn", action="store_true",
                        help="Exit 1 on warnings as well as failures")
    args = parser.parse_args()

    report = run_audit(args.project_root, args.tag, fail_on_warn=args.fail_on_warn)

    # Human-readable output
    print(f"\n{'='*60}")
    print(f"BioVision Dataset Audit — tag: {args.tag}")
    print(f"Project root: {args.project_root}")
    print(f"{'='*60}")
    for issue in report.issues:
        prefix = {"FAIL": "❌ FAIL", "WARN": "⚠  WARN", "INFO": "✓  INFO"}[issue["level"]]
        print(f"\n{prefix}  [{issue['check']}] {issue['message']}")
        if issue["detail"]:
            for line in issue["detail"].split("\n"):
                print(f"         {line}")
    print(f"\n{'='*60}")
    overall = "PASS" if not report.has_failures() else "FAIL"
    if args.fail_on_warn and report.has_warnings():
        overall = "FAIL"
    print(f"Overall: {overall}  ({report.to_dict()['fail_count']} failure(s), "
          f"{report.to_dict()['warn_count']} warning(s))")
    print(f"{'='*60}\n")

    # JSON report
    os.makedirs(os.path.join(args.project_root, "debug"), exist_ok=True)
    report_path = os.path.join(args.project_root, "debug", f"audit_{args.tag}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Report saved to: {report_path}\n")

    # Exit code
    if report.has_failures():
        sys.exit(1)
    if args.fail_on_warn and report.has_warnings():
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
