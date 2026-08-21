#!/usr/bin/env python3
"""Real, reproducible BioVision landmark-training accuracy smoke run.

This intentionally uses the production dataset preparation, dlib trainer,
locked-cohort evaluation, immutable artifacts, and model-promotion registry.
Only the source images are synthetic and the dlib capacity is reduced so the
entire check remains suitable for a developer smoke run.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.bv_utils import lineage  # noqa: E402
from backend.data.prepare_dataset import json_to_dlib_xml  # noqa: E402
from backend.training.train_shape_model import train_shape_model  # noqa: E402


SEED = 1729
TAG = "synthetic_accuracy_smoke"
INITIAL_POSES = (
    (-17, -11),
    (-14, 8),
    (-10, -4),
    (-7, 13),
    (-3, -14),
    (1, 5),
    (5, -8),
    (8, 15),
    (11, -1),
    (14, 10),
    (17, -12),
    (19, 3),
)
REVIEWED_POSES = (
    (-18, -12), (-18, 0), (-18, 12),
    (-12, -8), (-12, 8),
    (-6, -14), (-6, -4), (-6, 6), (-6, 14),
    (0, -12), (0, 0), (0, 12),
    (6, -14), (6, -4), (6, 6), (6, 14),
    (12, -8), (12, 8),
    (18, -12), (18, 12),
)

FAST_DLIB_OPTIONS = {
    "tree_depth": 2,
    "cascade_depth": 4,
    "nu": 0.12,
    "feature_pool_size": 80,
    "num_trees_per_cascade_level": 24,
    "num_test_splits": 5,
    "oversampling_amount": 2,
    "oversampling_translation_jitter": 0.0,
    "feature_pool_region_padding": 0.05,
    "lambda_param": 0.1,
    "num_threads": 1,
    "be_verbose": False,
    "random_seed": "42",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _landmarks_for_pose(dx: float, dy: float) -> list[dict]:
    return [
        {"id": 1, "x": 30.0 + dx * 0.58, "y": 33.0 + dy * 0.52, "isSkipped": False},
        {"id": 2, "x": 92.0 - dx * 0.43, "y": 38.0 - dy * 0.34, "isSkipped": False},
        {"id": 3, "x": 61.0 + dx * 0.32, "y": 91.0 + dy * 0.47, "isSkipped": False},
    ]


def _render_sample(dx: float, dy: float, variant: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + variant * 7919)
    y_grid, x_grid = np.mgrid[0:128, 0:128]
    base = 24.0 + 0.09 * x_grid + 0.06 * y_grid
    texture = rng.normal(0.0, 2.2, size=(128, 128))
    gray = np.clip(base + texture, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    landmarks = _landmarks_for_pose(dx, dy)
    points = [(int(round(item["x"])), int(round(item["y"]))) for item in landmarks]
    cv2.fillConvexPoly(image, np.asarray(points, dtype=np.int32), (48, 60, 72))
    cv2.polylines(image, [np.asarray(points, dtype=np.int32)], True, (115, 125, 140), 2)
    colors = ((245, 245, 245), (185, 235, 250), (250, 205, 175))
    for point, color in zip(points, colors):
        cv2.circle(image, point, 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, point, 7, (18, 18, 18), 1, lineType=cv2.LINE_AA)

    # Variant-specific marks make near-neighbour reviewed samples independent
    # source images without changing the landmark-generating relationship.
    cv2.line(image, (6, 117 - variant % 9), (20, 117 - variant % 9), (60 + variant % 70,) * 3, 1)
    image[0, 0] = (variant % 251, (variant * 3) % 251, (variant * 7) % 251)
    return image


def _write_sample(
    session_root: Path,
    name: str,
    dx: float,
    dy: float,
    variant: int,
    *,
    provenance: dict | None = None,
) -> None:
    images_dir = session_root / "images"
    labels_dir = session_root / "labels"
    image_name = f"{name}.png"
    image_path = images_dir / image_name
    if not cv2.imwrite(str(image_path), _render_sample(dx, dy, variant)):
        raise RuntimeError(f"Could not write synthetic image {image_path}")

    landmarks = _landmarks_for_pose(dx, dy)
    box = {
        "left": 8,
        "top": 8,
        "width": 112,
        "height": 112,
        "obbCorners": [[8, 8], [120, 8], [120, 120], [8, 120]],
        "angle": 0.0,
        "class_id": 0,
        "landmarks": landmarks,
    }
    payload = {
        "imageFilename": image_name,
        "boxes": [box],
        "provenance": provenance or {"source": "manual_annotation"},
        "finalizedDetection": {
            "isFinalized": True,
            "acceptedBoxes": [box],
            "boxSignature": f"synthetic:{name}",
        },
    }
    if provenance:
        payload["reviewHistory"] = [provenance]
    with (labels_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _create_session(session_root: Path) -> None:
    (session_root / "images").mkdir(parents=True)
    (session_root / "labels").mkdir(parents=True)
    with (session_root / "session.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schemaSemanticFingerprint": "v2-synthetic-smoke",
                "schemaSemanticVersion": 2,
                "orientationPolicyConfigured": True,
                "orientationPolicy": {
                    "mode": "invariant",
                    "trainingPrepBoxJitter": False,
                },
                "augmentationPolicy": {},
                "landmarkTemplate": [
                    {"index": 1, "name": "left marker", "category": "marker", "required": True},
                    {"index": 2, "name": "right marker", "category": "marker", "required": True},
                    {"index": 3, "name": "lower marker", "category": "marker", "required": True},
                ],
            },
            handle,
            indent=2,
        )

    for index, (dx, dy) in enumerate(INITIAL_POSES):
        _write_sample(session_root, f"initial_{index:02d}", dx, dy, index + 1)


def _prepare(session_root: Path, *, verbose: bool) -> tuple[dict, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        json_to_dlib_xml(str(session_root), TAG, test_split=0.30, seed=SEED)
    logs = stderr.getvalue()
    if verbose and logs:
        sys.stderr.write(logs)
    split_path = session_root / "debug" / f"split_info_{TAG}.json"
    with split_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), logs


def _train(session_root: Path, *, verbose: bool) -> tuple[dict, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = train_shape_model(
            str(session_root),
            TAG,
            custom_options=FAST_DLIB_OPTIONS,
            aug_angles=[],
            aug_flip=False,
        )
    logs = stderr.getvalue()
    if verbose and logs:
        sys.stderr.write(logs)
    return result, logs


def _add_reviewed_samples(session_root: Path, model_id: str) -> int:
    """Add a fixed dense pose grid independent of the locked test membership."""
    review_events = []
    for review_index, (dx, dy) in enumerate(REVIEWED_POSES):
        name = f"hitl_{review_index:02d}"
        event = {
            "eventId": f"smoke-review-{review_index:02d}",
            "commitId": f"smoke-review-{review_index:02d}",
            "source": "hitl_review",
            "speciesId": "synthetic-smoke",
            "inferenceSessionId": "synthetic-inference-v1",
            "imageFilename": f"{name}.png",
            "landmarkModelKey": model_id,
            "landmarkPredictorType": "dlib",
            "originalPredictionHash": f"v1-error-{review_index:02d}",
            "reviewedPredictionHash": f"reviewed-{review_index:02d}",
            "reviewOutcome": "corrected",
            "reviewer": "synthetic-smoke-harness",
            "reviewedAt": "2026-01-02T03:04:05Z",
            "acceptedSpecimens": 1,
            "rejectedDetections": 0,
        }
        _write_sample(
            session_root,
            name,
            dx,
            dy,
            1000 + review_index,
            provenance=event,
        )
        image_path = session_root / "images" / f"{name}.png"
        event["sourceImageSha256"] = _sha256_file(image_path)
        label_path = session_root / "labels" / f"{name}.json"
        with label_path.open("r", encoding="utf-8") as handle:
            label = json.load(handle)
        label["provenance"] = event
        label["reviewHistory"] = [event]
        with label_path.open("w", encoding="utf-8") as handle:
            json.dump(label, handle, indent=2)
        review_events.append(event)
    lineage.atomic_write_json(str(session_root / "review_events.json"), review_events)
    return len(review_events)


def _finite_float(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Expected finite evaluation metric, got {value!r}")
    return number


def run_smoke(*, verbose: bool = False, max_seconds: float = 60.0) -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="biovision-landmark-smoke-") as temp_root:
        session_root = Path(temp_root)
        _create_session(session_root)
        split_v1, prep_v1_logs = _prepare(session_root, verbose=verbose)
        assignments_path = session_root / "debug" / "cohorts" / "landmark_benchmark_v1.json"
        with assignments_path.open("r", encoding="utf-8") as handle:
            cohort_v1 = json.load(handle)
        assignments_v1 = dict(cohort_v1["assignments"])

        run_v1, train_v1_logs = _train(session_root, verbose=verbose)
        reviewed_count = _add_reviewed_samples(session_root, run_v1["model_id"])
        split_v2, prep_v2_logs = _prepare(session_root, verbose=verbose)
        with assignments_path.open("r", encoding="utf-8") as handle:
            cohort_v2 = json.load(handle)

        original_assignments_frozen = all(
            cohort_v2["assignments"].get(source_id) == assignment
            for source_id, assignment in assignments_v1.items()
        )
        adaptive_ids = [
            source_id
            for source_id, metadata in cohort_v2["sources"].items()
            if metadata.get("adaptiveTrainingSample")
        ]
        adaptive_train_only = bool(adaptive_ids) and all(
            cohort_v2["assignments"].get(source_id) == "train"
            for source_id in adaptive_ids
        )
        test_ids_frozen = split_v1["test_source_ids"] == split_v2["test_source_ids"]
        validation_ids_frozen = (
            split_v1["validation_source_ids"] == split_v2["validation_source_ids"]
        )

        run_v2, train_v2_logs = _train(session_root, verbose=verbose)
        v1_error = _finite_float(run_v1["test_median_error"])
        v2_error = _finite_float(run_v2["test_median_error"])
        v1_validation_error = _finite_float(run_v1["validation_median_error"])
        v2_validation_error = _finite_float(run_v2["validation_median_error"])
        improvement = v1_error - v2_error
        validation_improvement = v1_validation_error - v2_validation_error
        promotion = run_v2["registry"]["promotion"]
        elapsed = time.perf_counter() - started
        improved = improvement > 1e-9
        validation_improved = validation_improvement > 1e-9
        promoted = bool(promotion.get("promoted")) and run_v2["registry"].get("status") == "active"
        within_budget = elapsed <= float(max_seconds)
        ok = all(
            (
                original_assignments_frozen,
                test_ids_frozen,
                validation_ids_frozen,
                adaptive_train_only,
                validation_improved,
                promoted,
                within_budget,
            )
        )

        result = {
            "ok": ok,
            "seed": SEED,
            "elapsedSeconds": round(elapsed, 3),
            "maxSeconds": float(max_seconds),
            "temporarySessionCleaned": True,
            "cohort": {
                "initialTrainSources": len(split_v1["train_source_ids"]),
                "lockedTestSources": len(split_v1["test_source_ids"]),
                "lockedValidationSources": len(split_v1["validation_source_ids"]),
                "reviewedTrainSourcesAdded": reviewed_count,
                "originalAssignmentsFrozen": original_assignments_frozen,
                "testIdsFrozen": test_ids_frozen,
                "validationIdsFrozen": validation_ids_frozen,
                "adaptiveSourcesTrainOnly": adaptive_train_only,
                "assignmentDigestV1": lineage.sha256_json(assignments_v1),
                "assignmentDigestV2": lineage.sha256_json(cohort_v2["assignments"]),
                "lockedTestIdDigest": lineage.sha256_json(split_v1["test_source_ids"]),
                "lockedValidationIdDigest": lineage.sha256_json(
                    split_v1["validation_source_ids"]
                ),
            },
            "v1": {
                "modelId": run_v1["model_id"],
                "status": run_v1["registry"]["status"],
                "trainMedianNormalizedError": _finite_float(run_v1["train_median_error"]),
                "lockedValidationMedianNormalizedError": v1_validation_error,
                "lockedTestNativeError": _finite_float(run_v1["test_error"]),
                "lockedTestMedianNormalizedError": v1_error,
            },
            "v2": {
                "modelId": run_v2["model_id"],
                "status": run_v2["registry"]["status"],
                "trainMedianNormalizedError": _finite_float(run_v2["train_median_error"]),
                "lockedValidationMedianNormalizedError": v2_validation_error,
                "lockedTestNativeError": _finite_float(run_v2["test_error"]),
                "lockedTestMedianNormalizedError": v2_error,
            },
            "accuracy": {
                "lockedCohortPromotionMetric": "validationMedianError",
                "absoluteValidationMedianNormalizedErrorImprovement": validation_improvement,
                "relativeValidationMedianNormalizedErrorImprovement": (
                    validation_improvement / v1_validation_error
                    if v1_validation_error > 0
                    else None
                ),
                "validationNumericallyImproved": validation_improved,
                "reportOnlyTestAbsoluteMedianNormalizedErrorImprovement": improvement,
                "reportOnlyTestRelativeMedianNormalizedErrorImprovement": (
                    improvement / v1_error if v1_error > 0 else None
                ),
                # Informational only: test never participates in smoke success
                # or model promotion, so repeated developer runs cannot select
                # implementations against the locked report cohort.
                "reportOnlyTestNumericallyImproved": improved,
            },
            "promotion": promotion,
            "checks": {
                "promoted": promoted,
                "withinRuntimeBudget": within_budget,
            },
        }
        if not ok:
            result["diagnostics"] = {
                "preparationV1LogTail": prep_v1_logs[-2000:],
                "trainingV1LogTail": train_v1_logs[-2000:],
                "preparationV2LogTail": prep_v2_logs[-2000:],
                "trainingV2LogTail": train_v2_logs[-2000:],
            }
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Forward preparation/training logs to stderr.")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Fail if runtime exceeds this budget.")
    args = parser.parse_args()
    try:
        result = run_smoke(verbose=args.verbose, max_seconds=args.max_seconds)
    except Exception as exc:
        result = {
            "ok": False,
            "seed": SEED,
            "error": f"{type(exc).__name__}: {exc}",
            "temporarySessionCleaned": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
