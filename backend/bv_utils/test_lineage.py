import json
import os
import tempfile
import unittest
from unittest.mock import patch

from backend.bv_utils import lineage


PROMOTION_METRICS = ("validationMedianError", "validationError")


def evaluator_protocol(model_type="dlib", metric_names=PROMOTION_METRICS):
    """A valid, self-verifying evaluator protocol for promotion fixtures.

    Promotion requires this fingerprint on every model, including the first, so
    a fixture without one can never reach "active".
    """
    return lineage.build_validation_evaluator_protocol(
        {
            "modelType": model_type,
            "landmarkOrder": [0, 1],
            "evaluator": {"name": "landmark_validation", "version": 1},
            "preprocessing": {"standardSize": 512, "colorSpace": "bgr"},
            "metricDefinitions": {
                name: {"reduction": "median", "space": "normalized"}
                for name in metric_names
            },
        }
    )


def promotable_manifest(
    *,
    run_id="run",
    predictor_type="dlib",
    schema_fingerprint="v2-shared-schema",
    cohort_revision="locked-validation-cohort",
    validation_source_count=4,
    single_source_overlap=False,
    protocol=True,
):
    """The minimum manifest that satisfies the full promotion contract."""
    split_contract = {
        "testCohortRevision": cohort_revision,
        "validationCohortRevision": cohort_revision,
        "validationSourceCount": validation_source_count,
    }
    if single_source_overlap is not None:
        split_contract.update(
            {
                "singleSourceOverlap": single_source_overlap,
                "validationSourceOverlap": single_source_overlap,
            }
        )
    manifest = {
        "runId": run_id,
        "predictorType": predictor_type,
        "lineage": {
            "schema": {"semanticFingerprint": schema_fingerprint},
            "dataset": {"splits": [split_contract]},
        },
    }
    if protocol:
        manifest["validationEvaluatorProtocol"] = (
            protocol
            if isinstance(protocol, dict)
            else evaluator_protocol(predictor_type)
        )
    return manifest


class PromotionFixtureMixin:
    def _publish(
        self,
        root,
        run_id,
        score,
        *,
        metric_name="validationMedianError",
        schema_fingerprint="v2-shared-schema",
        cohort_revision="locked-validation-cohort",
        single_source_overlap=False,
        extra_metrics=None,
        predictor_type="dlib",
        promotion_policy=None,
        validation_source_count=4,
        protocol=True,
    ):
        model_type = "cnn" if predictor_type == "cnn" else "dlib"
        artifact_dir = lineage.create_model_artifact_dir(root, model_type, run_id)
        artifact = os.path.join(artifact_dir, "predictor.dat")
        alias = os.path.join(root, "models", f"predictor_model__{run_id}.dat")
        manifest = os.path.join(artifact_dir, "manifest.json")
        with open(artifact, "wb") as handle:
            handle.write(run_id.encode("ascii"))
        lineage.atomic_copy_file(artifact, alias)
        lineage.atomic_write_json(
            manifest,
            promotable_manifest(
                run_id=run_id,
                predictor_type=predictor_type,
                schema_fingerprint=schema_fingerprint,
                cohort_revision=cohort_revision,
                validation_source_count=validation_source_count,
                single_source_overlap=single_source_overlap,
                protocol=protocol,
            ),
        )
        return lineage.publish_model_run(
            root,
            model_type=model_type,
            predictor_type=predictor_type,
            run_id=run_id,
            display_name="model",
            artifact_tag=f"model__{run_id}",
            artifact_path=artifact,
            legacy_path=alias,
            run_manifest_path=manifest,
            metrics={metric_name: score, **dict(extra_metrics or {})},
            promotion_policy=promotion_policy,
        )


class ModelPromotionTests(PromotionFixtureMixin, unittest.TestCase):
    def test_regressing_run_is_candidate_and_better_run_promotes(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._publish(root, "run1", 0.20)
            worse = self._publish(root, "run2", 0.30)
            better = self._publish(root, "run3", 0.10)

            self.assertEqual(first["status"], "active")
            self.assertEqual(worse["status"], "candidate")
            self.assertFalse(worse["promotion"]["promoted"])
            self.assertEqual(better["status"], "active")
            self.assertTrue(better["promotion"]["promoted"])

            with open(os.path.join(root, "models", "model_registry.json"), encoding="utf-8") as handle:
                registry = json.load(handle)
            statuses = {entry["runId"]: entry["status"] for entry in registry["models"]}
            self.assertEqual(statuses, {"run1": "deprecated", "run2": "deprecated", "run3": "active"})

    def test_missing_metric_does_not_displace_measured_active_model(self):
        with tempfile.TemporaryDirectory() as root:
            self._publish(root, "run1", 0.20)
            artifact_dir = lineage.create_model_artifact_dir(root, "dlib", "run2")
            artifact = os.path.join(artifact_dir, "predictor.dat")
            manifest = os.path.join(artifact_dir, "manifest.json")
            with open(artifact, "wb") as handle:
                handle.write(b"dlib")
            lineage.atomic_write_json(manifest, {})
            record = lineage.publish_model_run(
                root,
                model_type="dlib",
                predictor_type="dlib",
                run_id="run2",
                display_name="model",
                artifact_tag="model__run2",
                artifact_path=artifact,
                legacy_path=artifact,
                run_manifest_path=manifest,
                metrics={},
            )
            self.assertEqual(record["status"], "candidate")

    def test_unvalidated_first_model_stays_candidate_until_a_measured_run_exists(self):
        with tempfile.TemporaryDirectory() as root:
            artifact_dir = lineage.create_model_artifact_dir(root, "dlib", "run1")
            artifact = os.path.join(artifact_dir, "predictor.dat")
            manifest = os.path.join(artifact_dir, "manifest.json")
            with open(artifact, "wb") as handle:
                handle.write(b"unvalidated")
            lineage.atomic_write_json(manifest, {"lineage": {}})
            first = lineage.publish_model_run(
                root,
                model_type="dlib",
                predictor_type="dlib",
                run_id="run1",
                display_name="model",
                artifact_tag="model__run1",
                artifact_path=artifact,
                legacy_path=artifact,
                run_manifest_path=manifest,
                metrics={},
            )
            self.assertEqual(first["status"], "candidate")
            self.assertEqual(
                first["promotion"]["reason"],
                "first_model_missing_locked_cohort_metric",
            )

            validated = self._publish(root, "run2", 0.20)
            self.assertEqual(validated["status"], "active")
            self.assertEqual(
                validated["promotion"]["reason"],
                "first_validated_model_of_type",
            )

    def test_single_validation_source_is_candidate_only(self):
        with tempfile.TemporaryDirectory() as root:
            record = self._publish(
                root,
                "run1",
                0.20,
                validation_source_count=1,
            )
            self.assertEqual(record["status"], "candidate")
            self.assertEqual(
                record["promotion"]["reason"],
                "first_model_insufficient_validation_sources",
            )
            self.assertEqual(record["promotion"]["minimumValidationSources"], 2)

    def test_missing_disjointness_evidence_cannot_activate_a_model(self):
        with tempfile.TemporaryDirectory() as root:
            unverified = self._publish(
                root,
                "run1",
                0.20,
                single_source_overlap=None,
            )
            self.assertEqual(unverified["status"], "candidate")
            self.assertEqual(
                unverified["promotion"]["reason"],
                "first_model_missing_cohort_disjointness_contract",
            )

            verified = self._publish(root, "run2", 0.20)
            self.assertEqual(verified["status"], "active")

    def test_promotion_uses_the_best_common_identically_named_metric(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._publish(
                root,
                "run1",
                0.20,
                metric_name="validationError",
            )
            self.assertEqual(first["status"], "active")

            artifact_dir = lineage.create_model_artifact_dir(root, "dlib", "run2")
            artifact = os.path.join(artifact_dir, "predictor.dat")
            manifest = os.path.join(artifact_dir, "manifest.json")
            with open(artifact, "wb") as handle:
                handle.write(b"run2")
            lineage.atomic_write_json(manifest, promotable_manifest(run_id="run2"))
            second = lineage.publish_model_run(
                root,
                model_type="dlib",
                predictor_type="dlib",
                run_id="run2",
                display_name="model",
                artifact_tag="model__run2",
                artifact_path=artifact,
                legacy_path=artifact,
                run_manifest_path=manifest,
                metrics={"validationMedianError": 0.50, "validationError": 0.10},
            )
            self.assertEqual(second["status"], "active")
            self.assertEqual(second["promotion"]["metric"], "validationError")
            self.assertEqual(second["promotion"]["candidateMetric"], "validationError")
            self.assertEqual(second["promotion"]["baselineMetric"], "validationError")

    def test_schema_or_cohort_mismatch_cannot_promote(self):
        with tempfile.TemporaryDirectory() as root:
            self._publish(root, "run1", 0.20)
            wrong_schema = self._publish(
                root,
                "run2",
                0.10,
                schema_fingerprint="v2-other-schema",
            )
            self.assertEqual(wrong_schema["status"], "candidate")
            self.assertEqual(
                wrong_schema["promotion"]["reason"],
                "incomparable_schema_semantic_fingerprint",
            )
            wrong_cohort = self._publish(
                root,
                "run3",
                0.05,
                cohort_revision="different-test-cohort",
            )
            self.assertEqual(wrong_cohort["status"], "candidate")
            self.assertEqual(
                wrong_cohort["promotion"]["reason"],
                "incomparable_frozen_cohort_revision",
            )

    def test_tie_and_training_overlap_remain_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            self._publish(root, "run1", 0.20)
            tied = self._publish(root, "run2", 0.20)
            self.assertEqual(tied["status"], "candidate")
            self.assertEqual(tied["promotion"]["reason"], "locked_cohort_not_improved")

        with tempfile.TemporaryDirectory() as root:
            self._publish(root, "run1", 0.20, single_source_overlap=True)
            leaked = self._publish(root, "run2", 0.10, single_source_overlap=True)
            self.assertEqual(leaked["status"], "candidate")
            self.assertEqual(
                leaked["promotion"]["reason"],
                "candidate_benchmark_overlaps_training",
            )

    def test_optional_effect_size_policy_blocks_noise_but_allows_material_gain(self):
        policy = {
            "policyVersion": "cnn_validation_effect_v1",
            "minimumAbsoluteImprovement": 1e-4,
            "minimumRelativeImprovement": 0.005,
        }
        with tempfile.TemporaryDirectory() as root:
            first = self._publish(
                root,
                "run1",
                0.05,
                predictor_type="cnn",
                promotion_policy=policy,
            )
            noisy = self._publish(
                root,
                "run2",
                0.05 - 1e-9,
                predictor_type="cnn",
                promotion_policy=policy,
            )
            material = self._publish(
                root,
                "run3",
                0.049,
                predictor_type="cnn",
                promotion_policy=policy,
            )

            self.assertEqual(first["status"], "active")
            self.assertEqual(noisy["status"], "candidate")
            self.assertEqual(
                noisy["promotion"]["reason"],
                "locked_cohort_improvement_below_minimum",
            )
            self.assertAlmostEqual(noisy["promotion"]["observedImprovement"], 1e-9)
            self.assertAlmostEqual(noisy["promotion"]["requiredImprovement"], 0.00025)
            self.assertEqual(noisy["promotion"]["improvementPolicy"], policy)
            self.assertEqual(material["status"], "active")
            self.assertEqual(material["promotion"]["reason"], "locked_cohort_improved")

    def test_test_metric_is_report_only_when_validation_is_present(self):
        with tempfile.TemporaryDirectory() as root:
            self._publish(
                root,
                "run1",
                0.20,
                metric_name="validationMedianError",
                cohort_revision="locked-validation-cohort",
                extra_metrics={"testMedianError": 0.10},
            )
            second = self._publish(
                root,
                "run2",
                0.10,
                metric_name="validationMedianError",
                cohort_revision="locked-validation-cohort",
                extra_metrics={"testMedianError": 0.50},
            )
            self.assertEqual(second["status"], "active")
            self.assertEqual(second["promotion"]["metric"], "validationMedianError")

    def test_dlib_test_only_improvement_cannot_promote(self):
        with tempfile.TemporaryDirectory() as root:
            self._publish(
                root,
                "run1",
                0.20,
                extra_metrics={"testMedianError": 0.40},
            )
            candidate = self._publish(
                root,
                "run2",
                0.01,
                metric_name="testMedianError",
            )
            self.assertEqual(candidate["status"], "candidate")
            self.assertEqual(
                candidate["promotion"]["reason"],
                "candidate_missing_locked_cohort_metric",
            )

    def test_cnn_test_only_improvement_cannot_promote(self):
        with tempfile.TemporaryDirectory() as root:
            self._publish(
                root,
                "run1",
                0.20,
                predictor_type="cnn",
                extra_metrics={"testMedianError": 0.40},
            )
            candidate = self._publish(
                root,
                "run2",
                0.01,
                metric_name="testMedianError",
                predictor_type="cnn",
            )
            self.assertEqual(candidate["status"], "candidate")
            self.assertEqual(
                candidate["promotion"]["reason"],
                "candidate_missing_locked_cohort_metric",
            )

    def test_catastrophic_validation_tail_blocks_better_median(self):
        with tempfile.TemporaryDirectory() as root:
            self._publish(root, "run1", 0.20)
            unstable = self._publish(
                root,
                "run2",
                0.10,
                extra_metrics={
                    "stabilityGatePassed": False,
                    "unstable": True,
                    "validationMaxError": 4.0,
                },
            )
            self.assertEqual(unstable["status"], "candidate")
            self.assertEqual(
                unstable["promotion"]["reason"],
                "candidate_failed_validation_stability_gate",
            )

    def test_alias_failure_rolls_back_aliases_and_registry(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._publish(root, "run1", 0.20)
            old_artifact = first["path"]
            current_alias = os.path.join(root, "models", "predictor_model.dat")
            lineage.atomic_copy_file(old_artifact, current_alias)
            registry_path = os.path.join(root, "models", "model_registry.json")
            registry_before = lineage.sha256_file(registry_path)
            alias_before = lineage.sha256_file(current_alias)

            artifact_dir = lineage.create_model_artifact_dir(root, "dlib", "run2")
            artifact = os.path.join(artifact_dir, "predictor.dat")
            manifest = os.path.join(artifact_dir, "manifest.json")
            with open(artifact, "wb") as handle:
                handle.write(b"better-run")
            lineage.atomic_write_json(manifest, promotable_manifest(run_id="run2"))
            invalid_destination = os.path.join(root, "models", "cannot-replace-directory")
            os.makedirs(invalid_destination)
            with self.assertRaises(OSError):
                lineage.publish_model_run(
                    root,
                    model_type="dlib",
                    predictor_type="dlib",
                    run_id="run2",
                    display_name="model",
                    artifact_tag="model__run2",
                    artifact_path=artifact,
                    legacy_path=artifact,
                    run_manifest_path=manifest,
                    current_alias_path=current_alias,
                    metrics={"validationMedianError": 0.10},
                    active_aliases=[
                        (artifact, current_alias),
                        (artifact, invalid_destination),
                    ],
                )
            self.assertEqual(lineage.sha256_file(registry_path), registry_before)
            self.assertEqual(lineage.sha256_file(current_alias), alias_before)

    def test_registry_write_failure_restores_previously_committed_alias(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._publish(root, "run1", 0.20)
            current_alias = os.path.join(root, "models", "predictor_model.dat")
            lineage.atomic_copy_file(first["path"], current_alias)
            registry_path = os.path.join(root, "models", "model_registry.json")
            registry_before = lineage.sha256_file(registry_path)
            alias_before = lineage.sha256_file(current_alias)

            artifact_dir = lineage.create_model_artifact_dir(root, "dlib", "run2")
            artifact = os.path.join(artifact_dir, "predictor.dat")
            manifest = os.path.join(artifact_dir, "manifest.json")
            with open(artifact, "wb") as handle:
                handle.write(b"better-run")
            with open(manifest, "w", encoding="utf-8") as handle:
                json.dump(promotable_manifest(run_id="run2"), handle)
            with (
                patch.object(lineage, "atomic_write_json", side_effect=OSError("registry locked")),
                self.assertRaisesRegex(OSError, "registry locked"),
            ):
                lineage.publish_model_run(
                    root,
                    model_type="dlib",
                    predictor_type="dlib",
                    run_id="run2",
                    display_name="model",
                    artifact_tag="model__run2",
                    artifact_path=artifact,
                    legacy_path=artifact,
                    run_manifest_path=manifest,
                    current_alias_path=current_alias,
                    metrics={"validationMedianError": 0.10},
                    active_aliases=[(artifact, current_alias)],
                )
            self.assertEqual(lineage.sha256_file(registry_path), registry_before)
            self.assertEqual(lineage.sha256_file(current_alias), alias_before)


class EvaluatorProtocolPromotionTests(PromotionFixtureMixin, unittest.TestCase):
    """Promotion requires a verifiable evaluator protocol on every model.

    Frozen cohort revisions bind the evaluator's pixels and targets; this
    fingerprint independently binds how those targets are mapped, preprocessed,
    and reduced to the metric.  Without it, two models' scores are not
    comparable even on identical data.
    """

    def test_first_model_without_an_evaluator_protocol_stays_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            record = self._publish(root, "run1", 0.20, protocol=False)
            self.assertEqual(record["status"], "candidate")
            self.assertEqual(
                record["promotion"]["reason"],
                "first_model_missing_evaluator_protocol_fingerprint",
            )

    def test_tampered_protocol_fingerprint_is_refused(self):
        tampered = dict(evaluator_protocol())
        tampered["preprocessing"] = {"standardSize": 256, "colorSpace": "bgr"}
        # fingerprint still describes the original 512px preprocessing
        with tempfile.TemporaryDirectory() as root:
            record = self._publish(root, "run1", 0.20, protocol=tampered)
            self.assertEqual(record["status"], "candidate")
            self.assertEqual(
                record["promotion"]["reason"],
                "first_model_missing_evaluator_protocol_fingerprint",
            )

    def test_protocol_from_a_different_model_type_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            record = self._publish(
                root, "run1", 0.20, protocol=evaluator_protocol("cnn")
            )
            self.assertEqual(record["status"], "candidate")
            self.assertEqual(
                record["promotion"]["reason"],
                "first_model_missing_evaluator_protocol_fingerprint",
            )

    def test_protocol_not_defining_the_promotion_metric_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            record = self._publish(
                root,
                "run1",
                0.20,
                protocol=evaluator_protocol(metric_names=("testMedianError",)),
            )
            self.assertEqual(record["status"], "candidate")
            self.assertEqual(
                record["promotion"]["reason"],
                "first_model_missing_evaluator_protocol_fingerprint",
            )

    def test_a_better_score_under_a_different_protocol_cannot_promote(self):
        """The core guarantee: a changed evaluator invalidates the comparison."""
        with tempfile.TemporaryDirectory() as root:
            baseline = self._publish(root, "run1", 0.20)
            self.assertEqual(baseline["status"], "active")

            divergent = evaluator_protocol()
            divergent = lineage.build_validation_evaluator_protocol(
                {
                    **{
                        key: value
                        for key, value in divergent.items()
                        if key != "fingerprint"
                    },
                    "preprocessing": {"standardSize": 256, "colorSpace": "rgb"},
                }
            )
            candidate = self._publish(root, "run2", 0.01, protocol=divergent)

            self.assertEqual(candidate["status"], "candidate")
            self.assertEqual(
                candidate["promotion"]["reason"],
                "incomparable_evaluator_protocol_fingerprint",
            )
            # The far better score must not have displaced the active model.
            self.assertEqual(baseline["runId"], "run1")

    def test_candidate_without_a_protocol_cannot_displace_an_active_model(self):
        with tempfile.TemporaryDirectory() as root:
            self._publish(root, "run1", 0.20)
            candidate = self._publish(root, "run2", 0.01, protocol=False)
            self.assertEqual(candidate["status"], "candidate")
            self.assertEqual(
                candidate["promotion"]["reason"],
                "candidate_missing_evaluator_protocol_fingerprint",
            )

    def test_protocol_builder_rejects_incomplete_contracts(self):
        complete = {
            "modelType": "dlib",
            "landmarkOrder": [0, 1],
            "evaluator": {"name": "landmark_validation", "version": 1},
            "preprocessing": {"standardSize": 512},
            "metricDefinitions": {"validationMedianError": {"reduction": "median"}},
        }
        lineage.build_validation_evaluator_protocol(complete)

        for field in ("evaluator", "preprocessing", "metricDefinitions", "landmarkOrder"):
            with self.subTest(missing=field):
                broken = {key: value for key, value in complete.items() if key != field}
                with self.assertRaises(ValueError):
                    lineage.build_validation_evaluator_protocol(broken)

        with self.subTest(field="modelType"):
            with self.assertRaises(ValueError):
                lineage.build_validation_evaluator_protocol(
                    {**complete, "modelType": "yolo"}
                )

    def test_protocol_fingerprint_is_stable_and_order_independent(self):
        first = evaluator_protocol()
        second = lineage.build_validation_evaluator_protocol(
            {
                "metricDefinitions": {
                    name: {"space": "normalized", "reduction": "median"}
                    for name in reversed(PROMOTION_METRICS)
                },
                "preprocessing": {"colorSpace": "bgr", "standardSize": 512},
                "evaluator": {"version": 1, "name": "landmark_validation"},
                "landmarkOrder": [0, 1],
                "modelType": "dlib",
            }
        )
        self.assertEqual(first["fingerprint"], second["fingerprint"])


class SchemaLineageTests(unittest.TestCase):
    def test_cnn_comparison_uses_cnn_validation_revision_not_dlib_split(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = os.path.join(root, "manifest.json")
            lineage.atomic_write_json(
                manifest,
                {
                    "modelType": "cnn",
                    "predictorType": "cnn",
                    "lineage": {
                        "schema": {"semanticFingerprint": "v2-shared-schema"},
                        "dataset": {
                            "splits": [
                                {
                                    "name": "split_info_model.json",
                                    "validationCohortRevision": "dlib-validation",
                                    "validationSourceOverlap": False,
                                },
                                {
                                    "name": "cnn_validation_v1.json",
                                    "validationCohortRevision": "cnn-validation",
                                    "validationSourceOverlap": False,
                                },
                            ]
                        },
                    },
                },
            )
            comparison = lineage._comparison_context(
                manifest,
                "validationMedianError",
            )
            self.assertEqual(comparison["cohortRevision"], "cnn-validation")
            self.assertTrue(comparison["cohortDisjoint"])

    def test_compatibility_fingerprint_is_distinct_from_integrity_hash(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "session.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schemaSemanticFingerprint": "v2-deadbeefdeadbeef",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [{"index": 1, "name": "Tip"}],
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                    },
                    handle,
                )
            snapshot = lineage.build_schema_snapshot(root)
            self.assertEqual(snapshot["semanticFingerprint"], "v2-deadbeefdeadbeef")
            self.assertEqual(snapshot["semanticVersion"], 2)
            self.assertEqual(len(snapshot["lineageSchemaHash"]), 64)

    def test_optional_semantics_are_preserved_in_lineage_hash_input(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "session.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "landmarkTemplate": [
                            {"index": 1, "name": "Required"},
                            {"index": 2, "name": "Optional", "optional": True},
                        ]
                    },
                    handle,
                )
            landmarks = lineage.build_schema_snapshot(root)["semantics"]["landmarks"]
            self.assertTrue(landmarks[0]["required"])
            self.assertFalse(landmarks[0]["optional"])
            self.assertFalse(landmarks[1]["required"])
            self.assertTrue(landmarks[1]["optional"])

    def test_dirty_diff_hash_includes_tracked_diff_state(self):
        responses = [
            "mock-commit\n",
            " M backend/example.py\n",
            b"mock-binary-diff",
            b"",
        ]
        with patch.object(lineage.subprocess, "check_output", side_effect=responses):
            state = lineage.collect_code_state()
        self.assertTrue(state["dirty"])
        self.assertEqual(
            state["trackedDiffSha256"],
            lineage.sha256_bytes(b"mock-binary-diff"),
        )
        self.assertTrue(state["diffSha256"])

    def test_code_state_hashes_backend_dependency_lock(self):
        with tempfile.TemporaryDirectory() as root:
            backend_dir = os.path.join(root, "backend")
            os.makedirs(backend_dir)
            with open(os.path.join(root, "package-lock.json"), "wb") as handle:
                handle.write(b"node-lock")
            with open(os.path.join(backend_dir, "requirements.txt"), "wb") as handle:
                handle.write(b"python-lock")
            with patch.object(lineage, "_code_root", return_value=root):
                state = lineage.collect_code_state()
            self.assertEqual(
                set(state["dependencyLocks"]),
                {"package-lock.json", "backend/requirements.txt"},
            )

    def test_runtime_state_includes_environment_and_accelerator_contract(self):
        state = lineage.collect_runtime_state()
        self.assertIn("installedDistributions", state)
        self.assertIn("pythonExecutable", state)
        self.assertIn("accelerator", state)
        self.assertIn("cudaAvailable", state["accelerator"])


class StrictJsonReadTests(unittest.TestCase):
    """`read_json` fails open by design; frozen cohort readers must not."""

    def test_missing_path_yields_the_declared_default(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(
                lineage.read_json_strict(
                    os.path.join(root, "absent.json"), missing_default={}
                ),
                {},
            )
            self.assertIsNone(lineage.read_json_strict(""))

    def test_present_but_unreadable_paths_raise(self):
        with tempfile.TemporaryDirectory() as root:
            empty = os.path.join(root, "empty.json")
            open(empty, "w", encoding="utf-8").close()
            malformed = os.path.join(root, "malformed.json")
            with open(malformed, "w", encoding="utf-8") as handle:
                handle.write('{"assignments": {')

            for path in (empty, malformed, root):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(RuntimeError, "unreadable or malformed"):
                        lineage.read_json_strict(path, missing_default={})

    def test_valid_payload_round_trips(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "manifest.json")
            lineage.atomic_write_json(path, {"version": 3})
            self.assertEqual(lineage.read_json_strict(path), {"version": 3})


if __name__ == "__main__":
    unittest.main()


class PackagedCodeStateTests(unittest.TestCase):
    """A packaged build has no source tree, but must still identify itself."""

    def test_frozen_build_records_executable_identity_instead_of_a_temp_path(self):
        with tempfile.TemporaryDirectory() as root:
            executable = os.path.join(root, "biovision_backend.exe")
            with open(executable, "wb") as handle:
                handle.write(b"pretend-bundle")

            with (
                patch.object(lineage.sys, "frozen", True, create=True),
                patch.object(lineage.sys, "executable", executable),
            ):
                state = lineage.collect_code_state()

            self.assertTrue(state["packaged"])
            self.assertEqual(state["executable"], os.path.abspath(executable))
            self.assertEqual(state["root"], os.path.abspath(root))
            self.assertEqual(state["executableSize"], len(b"pretend-bundle"))
            self.assertTrue(state["buildRevision"])
            # Never claim a git identity the packaged build does not have.
            self.assertIsNone(state["commit"])
            self.assertFalse(state["dirty"])

    def test_frozen_build_revision_changes_when_the_executable_changes(self):
        def revision_for(payload):
            with tempfile.TemporaryDirectory() as root:
                executable = os.path.join(root, "biovision_backend.exe")
                with open(executable, "wb") as handle:
                    handle.write(payload)
                with (
                    patch.object(lineage.sys, "frozen", True, create=True),
                    patch.object(lineage.sys, "executable", executable),
                ):
                    return lineage.collect_code_state()["buildRevision"]

        self.assertNotEqual(revision_for(b"build-one"), revision_for(b"build-two-longer"))

    def test_unfrozen_build_is_not_marked_packaged(self):
        state = lineage.collect_code_state()
        self.assertFalse(state["packaged"])
