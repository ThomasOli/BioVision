import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.bv_utils import lineage
from backend.bv_utils.landmark_artifacts import (
    ImmutableLandmarkArtifactError,
    bundle_id_mapping,
    resolve_landmark_runtime,
)


class ImmutableLandmarkArtifactTests(unittest.TestCase):
    def _mapping_payload(self, mode="directional"):
        policy = {"mode": mode, "obbLevelingMode": "on"}
        if mode == "directional":
            policy["targetOrientation"] = "left"
        return {
            "dlib_index_to_original": {"0": 3, "1": 12},
            "landmark_template": {
                "3": {"x_mean": 100.0, "y_mean": 220.0, "count": 2},
                "12": {"x_mean": 410.0, "y_mean": 230.0, "count": 2},
            },
            "training_config": {
                "orientation_mode": mode,
                "orientation_policy": policy,
                "target_orientation": "left" if mode == "directional" else None,
                "head_landmark_id": 3,
                "tail_landmark_id": 12,
            },
        }

    def _publish(self, root, predictor_type="dlib", run_id="run-1", tag="fish"):
        artifact_dir = lineage.create_model_artifact_dir(root, predictor_type, run_id)
        source_mapping = os.path.join(root, "debug", f"id_mapping_{tag}.json")
        lineage.atomic_write_json(source_mapping, self._mapping_payload())
        _, mapping_descriptor, _ = bundle_id_mapping(source_mapping, artifact_dir)

        if predictor_type == "dlib":
            model_name = "predictor.dat"
            config_path = None
        else:
            model_name = "model.pth"
            config_path = os.path.join(artifact_dir, "config.json")
            lineage.atomic_write_json(
                config_path,
                {
                    "cnn_format_version": 2,
                    "n_landmarks": 2,
                    "landmark_ids": [3, 12],
                },
            )
        model_path = os.path.join(artifact_dir, model_name)
        Path(model_path).write_bytes(f"{predictor_type}:{run_id}".encode("ascii"))
        model_id = lineage.build_model_id(predictor_type, run_id)
        artifact_tag = lineage.build_artifact_tag(tag, run_id)
        manifest_path = os.path.join(artifact_dir, "manifest.json")
        manifest = {
            "formatVersion": 2,
            "modelId": model_id,
            "modelType": predictor_type,
            "predictorType": predictor_type,
            "displayName": tag,
            "artifactTag": artifact_tag,
            "runId": run_id,
            "artifact": {
                "path": model_path,
                "relativePath": model_name,
                "sha256": lineage.sha256_file(model_path),
            },
            "sidecars": {"idMapping": mapping_descriptor},
            "lineage": {
                "schema": {"semanticFingerprint": "schema-v1"},
                "dataset": {"splits": [{"testCohortRevision": "cohort-v1"}]},
            },
        }
        if config_path:
            manifest["config"] = {
                "path": config_path,
                "relativePath": "config.json",
                "sha256": lineage.sha256_file(config_path),
            }
        lineage.atomic_write_json(manifest_path, manifest)
        record = lineage.publish_model_run(
            root,
            model_type=predictor_type,
            predictor_type=predictor_type,
            run_id=run_id,
            display_name=tag,
            artifact_tag=artifact_tag,
            artifact_path=model_path,
            legacy_path=model_path,
            config_path=config_path,
            run_manifest_path=manifest_path,
            metrics={"testMedianError": 0.2},
        )
        return {
            "artifact_dir": artifact_dir,
            "artifact_tag": artifact_tag,
            "manifest": manifest,
            "record": record,
        }

    def test_dlib_registry_and_resolver_use_verified_artifact_sidecar(self):
        with tempfile.TemporaryDirectory() as root:
            published = self._publish(root)
            runtime = resolve_landmark_runtime(
                root,
                published["artifact_tag"],
                "dlib",
            )

            expected_mapping = os.path.join(published["artifact_dir"], "id_mapping.json")
            self.assertTrue(runtime["immutable"])
            self.assertEqual(runtime["id_mapping_path"], expected_mapping)
            self.assertEqual(runtime["id_mapping"]["dlib_index_to_original"], {"0": 3, "1": 12})
            registry_descriptor = published["record"]["sidecars"]["idMapping"]
            self.assertEqual(
                registry_descriptor["sha256"],
                lineage.sha256_file(expected_mapping),
            )
            self.assertEqual(registry_descriptor["relativePath"], "id_mapping.json")
            self.assertEqual(
                published["record"]["artifact"]["sha256"],
                lineage.sha256_file(runtime["model_path"]),
            )

    def test_missing_or_tampered_immutable_sidecar_never_uses_debug_fallback(self):
        for mutation in ("missing", "tampered"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                published = self._publish(root)
                artifact_mapping = os.path.join(published["artifact_dir"], "id_mapping.json")
                # A valid mutable debug mapping exists, but an immutable record
                # must never fall back to it after resolution succeeds.
                debug_mapping = os.path.join(root, "debug", f"id_mapping_{published['artifact_tag']}.json")
                lineage.atomic_write_json(debug_mapping, self._mapping_payload("invariant"))
                if mutation == "missing":
                    os.unlink(artifact_mapping)
                else:
                    lineage.atomic_write_json(artifact_mapping, self._mapping_payload("bilateral"))

                with self.assertRaisesRegex(
                    ImmutableLandmarkArtifactError,
                    "missing|SHA-256 mismatch",
                ):
                    resolve_landmark_runtime(
                        root,
                        published["artifact_tag"],
                        "dlib",
                        allow_legacy=True,
                    )

    def test_legacy_alias_fallback_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "models", "predictor_legacy.dat")
            mapping_path = os.path.join(root, "debug", "id_mapping_legacy.json")
            Path(model_path).parent.mkdir(parents=True)
            Path(mapping_path).parent.mkdir(parents=True)
            Path(model_path).write_bytes(b"legacy")
            Path(mapping_path).write_text(
                json.dumps(self._mapping_payload()),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ImmutableLandmarkArtifactError,
                "No immutable dlib landmark artifact",
            ):
                resolve_landmark_runtime(root, "legacy", "dlib")
            runtime = resolve_landmark_runtime(
                root,
                "legacy",
                "dlib",
                allow_legacy=True,
            )
            self.assertFalse(runtime["immutable"])
            self.assertEqual(runtime["model_path"], model_path)
            self.assertEqual(runtime["id_mapping_path"], mapping_path)

            lineage.atomic_write_json(
                os.path.join(root, "models", "model_registry.json"),
                {
                    "version": 2,
                    "models": [
                        {
                            "modelId": "legacy-dlib:registered",
                            "key": "legacy-dlib:registered",
                            "name": "legacy",
                            "displayName": "legacy",
                            "artifactTag": "legacy",
                            "predictorType": "dlib",
                            "modelKind": "landmark",
                            "path": model_path,
                            "legacyPath": model_path,
                            "status": "active",
                        }
                    ],
                },
            )
            registered = resolve_landmark_runtime(
                root,
                "legacy-dlib:registered",
                "dlib",
                allow_legacy=True,
            )
            self.assertFalse(registered["immutable"])
            self.assertEqual(registered["record"]["modelId"], "legacy-dlib:registered")

    def test_deleted_v2_registry_record_cannot_fall_back_to_mutable_alias(self):
        with tempfile.TemporaryDirectory() as root:
            published = self._publish(root)
            alias_tag = published["artifact_tag"]
            alias_path = os.path.join(root, "models", f"predictor_{alias_tag}.dat")
            mapping_path = os.path.join(root, "debug", f"id_mapping_{alias_tag}.json")
            Path(alias_path).write_bytes(b"mutable-alias")
            lineage.atomic_write_json(mapping_path, self._mapping_payload("invariant"))
            lineage.atomic_write_json(
                os.path.join(root, "models", "model_registry.json"),
                {"version": 2, "models": []},
            )

            with self.assertRaisesRegex(
                ImmutableLandmarkArtifactError,
                "no registered dlib identity|refusing mutable alias fallback",
            ):
                resolve_landmark_runtime(
                    root,
                    alias_tag,
                    "dlib",
                    allow_legacy=True,
                )

    def test_malformed_v2_registry_cannot_fall_back_to_mutable_alias(self):
        malformed_registries = (
            '{"version": 2, "models": ',
            json.dumps({"version": 2, "models": {"not": "a list"}}),
            json.dumps({"models": []}),
        )
        for registry_text in malformed_registries:
            with self.subTest(registry=registry_text), tempfile.TemporaryDirectory() as root:
                alias_path = os.path.join(root, "models", "predictor_fish__run-1.dat")
                mapping_path = os.path.join(root, "debug", "id_mapping_fish__run-1.json")
                registry_path = os.path.join(root, "models", "model_registry.json")
                Path(alias_path).parent.mkdir(parents=True)
                Path(mapping_path).parent.mkdir(parents=True)
                Path(alias_path).write_bytes(b"mutable-alias")
                Path(mapping_path).write_text(
                    json.dumps(self._mapping_payload()),
                    encoding="utf-8",
                )
                Path(registry_path).write_text(registry_text, encoding="utf-8")

                with self.assertRaisesRegex(
                    ImmutableLandmarkArtifactError,
                    "malformed|invalid models collection|missing or unsupported version",
                ):
                    resolve_landmark_runtime(
                        root,
                        "fish__run-1",
                        "dlib",
                        allow_legacy=True,
                    )

    def test_cnn_config_and_mapping_are_resolved_from_same_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            published = self._publish(root, predictor_type="cnn")
            runtime = resolve_landmark_runtime(root, published["artifact_tag"], "cnn")

            self.assertTrue(runtime["immutable"])
            self.assertEqual(
                runtime["config_path"],
                os.path.join(published["artifact_dir"], "config.json"),
            )
            self.assertEqual(
                published["record"]["config"]["sha256"],
                lineage.sha256_file(runtime["config_path"]),
            )
            self.assertEqual(
                published["record"]["sidecars"]["idMapping"]["sha256"],
                lineage.sha256_file(runtime["id_mapping_path"]),
            )

    def test_hitl_annotator_loads_artifact_mapping_and_orientation(self):
        from backend.annotation.super_annotator import SuperAnnotator

        with tempfile.TemporaryDirectory() as root:
            published = self._publish(root)
            model_path = os.path.join(published["artifact_dir"], "predictor.dat")
            # Mutable session/debug metadata deliberately disagrees with the
            # directional training contract bundled in the artifact.
            lineage.atomic_write_json(
                os.path.join(root, "session.json"),
                {"orientationPolicy": {"mode": "invariant"}},
            )
            annotator = SuperAnnotator()
            with patch("dlib.shape_predictor", return_value=object()) as loader:
                annotator.load_dlib_model(model_path)

            loader.assert_called_once_with(model_path)
            self.assertTrue(annotator.dlib_runtime_immutable)
            self.assertEqual(annotator.dlib_id_mapping, {0: 3, 1: 12})
            self.assertEqual(annotator.dlib_orientation_policy["mode"], "directional")

    def test_prediction_runtime_ignores_mutated_session_metadata(self):
        from backend.inference.predict import _load_landmark_runtime_metadata

        with tempfile.TemporaryDirectory() as root:
            published = self._publish(root)
            lineage.atomic_write_json(
                os.path.join(root, "session.json"),
                {"orientationPolicy": {"mode": "invariant"}},
            )

            metadata = _load_landmark_runtime_metadata(
                root,
                published["artifact_tag"],
                "dlib",
            )
            self.assertTrue(metadata["runtime"]["immutable"])
            self.assertEqual(metadata["orientation_policy"]["mode"], "directional")
            self.assertEqual(metadata["index_to_original"], {0: 3, 1: 12})
            self.assertEqual(metadata["head_landmark_id"], 3)
            self.assertEqual(metadata["tail_landmark_id"], 12)

    def test_semantically_invalid_but_hash_matching_sidecar_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            published = self._publish(root)
            mapping_path = os.path.join(published["artifact_dir"], "id_mapping.json")
            invalid = self._mapping_payload()
            invalid["training_config"]["orientation_policy"] = {"mode": "invariant"}
            lineage.atomic_write_json(mapping_path, invalid)
            invalid_sha = lineage.sha256_file(mapping_path)

            manifest_path = os.path.join(published["artifact_dir"], "manifest.json")
            manifest = lineage.read_json(manifest_path)
            manifest["sidecars"]["idMapping"]["sha256"] = invalid_sha
            lineage.atomic_write_json(manifest_path, manifest)
            registry_path = os.path.join(root, "models", "model_registry.json")
            registry = lineage.read_json(registry_path)
            registry["models"][0]["sidecars"]["idMapping"]["sha256"] = invalid_sha
            lineage.atomic_write_json(registry_path, registry)

            with self.assertRaisesRegex(
                ImmutableLandmarkArtifactError,
                "sidecar is invalid",
            ):
                resolve_landmark_runtime(root, published["artifact_tag"], "dlib")

    def test_registry_run_id_cannot_escape_immutable_run_root(self):
        unsafe_run_ids = (
            "../outside",
            "..\\outside",
            "nested/run",
            "nested\\run",
            os.path.abspath(os.path.join(os.sep, "outside-biovision-run")),
        )
        for unsafe_run_id in unsafe_run_ids:
            with self.subTest(run_id=unsafe_run_id), tempfile.TemporaryDirectory() as root:
                published = self._publish(root)
                registry_path = os.path.join(root, "models", "model_registry.json")
                registry = lineage.read_json(registry_path)
                registry["models"][0]["runId"] = unsafe_run_id
                lineage.atomic_write_json(registry_path, registry)

                with self.assertRaisesRegex(
                    ImmutableLandmarkArtifactError,
                    "unsafe run ID|outside its model run root",
                ):
                    resolve_landmark_runtime(
                        root,
                        published["artifact_tag"],
                        "dlib",
                        allow_legacy=True,
                    )


if __name__ == "__main__":
    unittest.main()
