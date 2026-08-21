import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.bv_utils import lineage
from backend.training import train_cnn_model


class CnnValidationCohortTests(unittest.TestCase):
    @staticmethod
    def _write_record(root, index):
        image_path = os.path.join(root, f"crop-{index}.bin")
        with open(image_path, "wb") as handle:
            handle.write(f"deterministic-image-{index}".encode("ascii"))
        return image_path, {"00": (10 + index, 20 + index)}

    def test_fallback_source_identity_uses_content_not_filename(self):
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, "first.bin")
            renamed = os.path.join(root, "renamed.bin")
            for path in (first, renamed):
                with open(path, "wb") as handle:
                    handle.write(b"identical-source")
            self.assertEqual(
                train_cnn_model._record_source_key(first),
                train_cnn_model._record_source_key(renamed),
            )

    def test_no_validation_cohort_never_restores_an_early_checkpoint(self):
        epoch_one_state = {"weight": object()}
        self.assertFalse(
            train_cnn_model._should_restore_validation_checkpoint(False, epoch_one_state)
        )
        self.assertTrue(
            train_cnn_model._should_restore_validation_checkpoint(True, epoch_one_state)
        )

    def test_cnn_validation_tail_gate_blocks_catastrophic_outliers(self):
        stable = train_cnn_model._cnn_validation_stability_gate(
            {"mean": 0.04, "median": 0.04, "p95": 0.06, "max": 0.07}
        )
        unstable = train_cnn_model._cnn_validation_stability_gate(
            {"mean": 0.09, "median": 0.01, "p95": 0.18, "max": 0.30}
        )
        unavailable = train_cnn_model._cnn_validation_stability_gate(None)

        self.assertTrue(stable["passed"])
        self.assertFalse(unstable["passed"])
        self.assertEqual(unstable["reason"], "catastrophic_validation_outliers")
        self.assertIsNone(unavailable["passed"])

    def test_reproducibility_protocol_seeds_all_rngs_and_disables_cudnn_benchmark(self):
        fake_cudnn = SimpleNamespace(
            benchmark=True,
            deterministic=False,
            allow_tf32=True,
        )
        fake_matmul = SimpleNamespace(allow_tf32=True)
        fake_torch = MagicMock()
        fake_torch.backends = SimpleNamespace(
            cudnn=fake_cudnn,
            cuda=SimpleNamespace(matmul=fake_matmul),
        )

        with (
            patch.object(train_cnn_model, "torch", fake_torch),
            patch.object(train_cnn_model.random, "seed") as python_seed,
            patch.object(train_cnn_model.np.random, "seed") as numpy_seed,
            patch.dict(os.environ, {}, clear=True),
        ):
            protocol = train_cnn_model._configure_reproducibility(1234)

        python_seed.assert_called_once_with(1234)
        numpy_seed.assert_called_once_with(1234)
        fake_torch.manual_seed.assert_called_once_with(1234)
        fake_torch.cuda.manual_seed_all.assert_called_once_with(1234)
        fake_torch.use_deterministic_algorithms.assert_called_once_with(
            True,
            warn_only=True,
        )
        self.assertFalse(fake_cudnn.benchmark)
        self.assertTrue(fake_cudnn.deterministic)
        self.assertFalse(fake_cudnn.allow_tf32)
        self.assertFalse(fake_matmul.allow_tf32)
        self.assertEqual(protocol["seed"], 1234)
        self.assertEqual(protocol["protocolVersion"], "cnn_reproducibility_v1")
        self.assertTrue(protocol["dataLoaderShuffleGeneratorSeeded"])

    def test_dataloader_workers_receive_torch_derived_independent_rng(self):
        dataset = SimpleNamespace(_rng=None)
        fake_torch = MagicMock()
        fake_torch.initial_seed.return_value = (2 ** 32) + 17
        fake_torch.utils.data.get_worker_info.return_value = SimpleNamespace(
            dataset=dataset
        )
        seeded_rng = object()
        with (
            patch.object(train_cnn_model, "torch", fake_torch),
            patch.object(train_cnn_model.random, "seed") as python_seed,
            patch.object(train_cnn_model.np.random, "seed") as numpy_seed,
            patch.object(
                train_cnn_model.np.random,
                "default_rng",
                return_value=seeded_rng,
            ) as dataset_seed,
        ):
            train_cnn_model._seed_dataloader_worker(3)

        python_seed.assert_called_once_with(17)
        numpy_seed.assert_called_once_with(17)
        dataset_seed.assert_called_once_with(17)
        self.assertIs(dataset._rng, seeded_rng)

    def test_dataloader_shuffle_stream_replays_for_same_seed(self):
        first, first_seed = train_cnn_model._make_dataloader_generator(77, 0)
        replay, replay_seed = train_cnn_model._make_dataloader_generator(77, 0)
        other, other_seed = train_cnn_model._make_dataloader_generator(77, 1)

        first_order = train_cnn_model.torch.randperm(32, generator=first).tolist()
        replay_order = train_cnn_model.torch.randperm(32, generator=replay).tolist()
        other_order = train_cnn_model.torch.randperm(32, generator=other).tolist()
        self.assertEqual(first_seed, replay_seed)
        self.assertEqual(first_order, replay_order)
        self.assertNotEqual(first_seed, other_seed)
        self.assertNotEqual(first_order, other_order)

    def test_training_protocol_and_promotion_policy_are_revisioned(self):
        first = train_cnn_model._build_cnn_training_protocol(
            {"seed": 42, "optimizer": {"lr": 0.001, "epochs": 10}}
        )
        reordered = train_cnn_model._build_cnn_training_protocol(
            {"optimizer": {"epochs": 10, "lr": 0.001}, "seed": 42}
        )
        changed = train_cnn_model._build_cnn_training_protocol(
            {"seed": 43, "optimizer": {"lr": 0.001, "epochs": 10}}
        )

        self.assertEqual(first["revision"], reordered["revision"])
        self.assertNotEqual(first["revision"], changed["revision"])
        self.assertEqual(
            train_cnn_model._cnn_promotion_policy(),
            {
                "policyVersion": "cnn_validation_effect_v1",
                "minimumAbsoluteImprovement": 1e-4,
                "minimumRelativeImprovement": 0.005,
            },
        )

    def test_effective_dataset_revision_binds_crop_geometry_source_and_split(self):
        with tempfile.TemporaryDirectory() as root:
            first_path = os.path.join(root, "first.bin")
            renamed_path = os.path.join(root, "renamed.bin")
            for path in (first_path, renamed_path):
                with open(path, "wb") as handle:
                    handle.write(b"same-canonical-crop")
            first_record = (first_path, {"00": (10, 20), "01": (30, 40)})
            renamed_record = (renamed_path, {"00": (10, 20), "01": (30, 40)})
            first_sources = {
                os.path.normcase(os.path.abspath(first_path)): "sha256:raw-source"
            }
            renamed_sources = {
                os.path.normcase(os.path.abspath(renamed_path)): "sha256:raw-source"
            }

            first = train_cnn_model._build_effective_cnn_dataset(
                [first_record],
                [],
                [],
                landmark_keys=["00", "01"],
                source_ids_by_path=first_sources,
            )
            renamed = train_cnn_model._build_effective_cnn_dataset(
                [renamed_record],
                [],
                [],
                landmark_keys=["00", "01"],
                source_ids_by_path=renamed_sources,
            )
            changed_geometry = train_cnn_model._build_effective_cnn_dataset(
                [(renamed_path, {"00": (11, 20), "01": (30, 40)})],
                [],
                [],
                landmark_keys=["00", "01"],
                source_ids_by_path=renamed_sources,
            )
            changed_assignment = train_cnn_model._build_effective_cnn_dataset(
                [],
                [renamed_record],
                [],
                landmark_keys=["00", "01"],
                source_ids_by_path=renamed_sources,
            )

            self.assertEqual(first["revision"], renamed["revision"])
            self.assertNotEqual(first["revision"], changed_geometry["revision"])
            self.assertNotEqual(first["revision"], changed_assignment["revision"])
            record = first["splits"]["train"]["records"][0]
            self.assertEqual(record["sourceId"], "sha256:raw-source")
            self.assertEqual(len(record["canonicalCropSha256"]), 64)
            self.assertEqual(record["landmarks"][0]["x"], 10)

            train_cnn_model._assert_effective_cnn_dataset_unchanged(
                [first_record],
                [],
                [],
                landmark_keys=["00", "01"],
                source_ids_by_path=first_sources,
                expected=first,
            )
            with open(first_path, "wb") as handle:
                handle.write(b"mutated-canonical-crop")
            with self.assertRaisesRegex(RuntimeError, "changed while training"):
                train_cnn_model._assert_effective_cnn_dataset_unchanged(
                    [first_record],
                    [],
                    [],
                    landmark_keys=["00", "01"],
                    source_ids_by_path=first_sources,
                    expected=first,
                )

    def test_initializer_state_hash_tracks_exact_tensor_bytes(self):
        first = train_cnn_model.torch.nn.Linear(3, 2)
        clone = train_cnn_model.torch.nn.Linear(3, 2)
        clone.load_state_dict(first.state_dict())
        baseline = train_cnn_model._hash_torch_state_dict(first.state_dict())
        self.assertEqual(
            train_cnn_model._hash_torch_state_dict(clone.state_dict()),
            baseline,
        )
        with train_cnn_model.torch.no_grad():
            clone.weight[0, 0] += 1.0
        self.assertNotEqual(
            train_cnn_model._hash_torch_state_dict(clone.state_dict()),
            baseline,
        )

    def test_preparation_source_map_groups_distinct_augmented_crops(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._write_record(root, 1)[0]
            augmented = self._write_record(root, 2)[0]
            source_ids = {
                os.path.normcase(os.path.abspath(first)): "sha256:raw-source",
                os.path.normcase(os.path.abspath(augmented)): "sha256:raw-source",
            }
            self.assertEqual(
                train_cnn_model._record_source_key(first, source_ids),
                train_cnn_model._record_source_key(augmented, source_ids),
            )

    def test_legacy_filename_manifest_migrates_to_content_identity(self):
        with tempfile.TemporaryDirectory() as root:
            alpha_path = os.path.join(root, "alpha_crop.bin")
            beta_path = os.path.join(root, "beta_crop.bin")
            with open(alpha_path, "wb") as handle:
                handle.write(b"alpha-content")
            with open(beta_path, "wb") as handle:
                handle.write(b"beta-content")
            records = [
                (alpha_path, {"00": (10, 20)}),
                (beta_path, {"00": (30, 40)}),
            ]
            assignment_path = os.path.join(root, "cnn_validation_v1.json")
            lineage.atomic_write_json(
                assignment_path,
                {
                    "version": 1,
                    "seed": 42,
                    "validationFraction": 0.2,
                    "updatedAt": "2026-01-01T00:00:00Z",
                    "assignments": {"alpha": "val", "beta": "train"},
                },
            )
            train_records, val_records, _meta = train_cnn_model._split_train_val_records(
                records,
                assignment_path=assignment_path,
            )
            self.assertEqual([record[0] for record in val_records], [alpha_path])
            self.assertEqual([record[0] for record in train_records], [beta_path])
            migrated = lineage.read_json(assignment_path)
            self.assertEqual(migrated["version"], 3)
            self.assertNotIn("alpha", migrated["assignments"])
            self.assertIn(
                train_cnn_model._record_source_key(alpha_path),
                migrated["assignments"],
            )

    def test_first_validation_bootstrap_keeps_hitl_sources_train_only(self):
        with tempfile.TemporaryDirectory() as root:
            records = [self._write_record(root, index) for index in range(1, 7)]
            source_ids = {
                os.path.normcase(os.path.abspath(path)): f"sha256:source-{index}"
                for index, (path, _parts) in enumerate(records, start=1)
            }
            adaptive_id = "sha256:source-1"
            assignment_path = os.path.join(root, "cnn_validation_v1.json")

            train_records, val_records, meta = train_cnn_model._split_train_val_records(
                records,
                val_fraction=0.9,
                seed=42,
                assignment_path=assignment_path,
                source_ids_by_path=source_ids,
                adaptive_source_ids={adaptive_id},
            )

            self.assertIn(records[0], train_records)
            self.assertNotIn(records[0], val_records)
            self.assertIn(adaptive_id, meta["adaptive_train_sources"])
            manifest = lineage.read_json(assignment_path)
            self.assertEqual(manifest["assignments"][adaptive_id], "train")
            self.assertEqual(manifest["adaptiveSourcePolicy"], "train_only")

    def test_one_to_two_growth_keeps_original_train_and_uses_only_new_source_for_validation(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._write_record(root, 1)
            second = self._write_record(root, 2)
            assignment_path = os.path.join(root, "cnn_validation_v1.json")

            train_records, val_records, meta = train_cnn_model._split_train_val_records(
                [first],
                assignment_path=assignment_path,
                seed=42,
            )
            first_source = train_cnn_model._record_source_key(first[0])
            second_source = train_cnn_model._record_source_key(second[0])
            self.assertEqual(train_records, [first])
            self.assertEqual(val_records, [])
            self.assertIsNone(meta["validation_cohort_revision"])
            initial_manifest = lineage.read_json(assignment_path)
            self.assertEqual(initial_manifest["assignments"], {first_source: "train"})

            train_records, val_records, meta = train_cnn_model._split_train_val_records(
                [first, second],
                assignment_path=assignment_path,
                val_fraction=0.99,
                seed=999,
            )
            grown_manifest = lineage.read_json(assignment_path)
            self.assertEqual(train_records, [first])
            self.assertEqual(val_records, [second])
            self.assertEqual(grown_manifest["assignments"][first_source], "train")
            self.assertEqual(grown_manifest["assignments"][second_source], "val")
            self.assertEqual(grown_manifest["seed"], 42)
            self.assertIsNotNone(meta["validation_cohort_revision"])

    def test_one_to_three_growth_never_moves_prior_train_and_excludes_new_adaptive_source(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._write_record(root, 1)
            second = self._write_record(root, 2)
            reviewed = self._write_record(root, 3)
            assignment_path = os.path.join(root, "cnn_validation_v1.json")
            source_ids = {
                os.path.normcase(os.path.abspath(first[0])): "sha256:first",
                os.path.normcase(os.path.abspath(second[0])): "sha256:second",
                os.path.normcase(os.path.abspath(reviewed[0])): "sha256:reviewed",
            }

            train_cnn_model._split_train_val_records(
                [first],
                assignment_path=assignment_path,
                source_ids_by_path=source_ids,
                seed=73,
            )
            train_records, val_records, _meta = train_cnn_model._split_train_val_records(
                [first, second, reviewed],
                assignment_path=assignment_path,
                source_ids_by_path=source_ids,
                adaptive_source_ids={"sha256:reviewed"},
                seed=999,
            )

            manifest = lineage.read_json(assignment_path)
            self.assertEqual(manifest["assignments"]["sha256:first"], "train")
            self.assertEqual(manifest["assignments"]["sha256:second"], "val")
            self.assertEqual(manifest["assignments"]["sha256:reviewed"], "train")
            self.assertEqual(train_records, [first, reviewed])
            self.assertEqual(val_records, [second])

    def test_clean_growth_bootstraps_two_source_validation_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._write_record(root, 1)
            second = self._write_record(root, 2)
            third = self._write_record(root, 3)
            assignment_path = os.path.join(root, "cnn_validation_v1.json")

            train_cnn_model._split_train_val_records(
                [first],
                assignment_path=assignment_path,
                seed=73,
            )
            train_records, val_records, meta = train_cnn_model._split_train_val_records(
                [first, second, third],
                assignment_path=assignment_path,
                seed=999,
            )

            self.assertEqual(train_records, [first])
            self.assertCountEqual(val_records, [second, third])
            self.assertEqual(len(meta["val_sources"]), 2)
            manifest = lineage.read_json(assignment_path)
            self.assertEqual(
                manifest["assignments"][train_cnn_model._record_source_key(first[0])],
                "train",
            )

    def test_adaptive_only_growth_cannot_repurpose_the_original_training_source(self):
        with tempfile.TemporaryDirectory() as root:
            first = self._write_record(root, 1)
            reviewed = self._write_record(root, 2)
            assignment_path = os.path.join(root, "cnn_validation_v1.json")
            source_ids = {
                os.path.normcase(os.path.abspath(first[0])): "sha256:first",
                os.path.normcase(os.path.abspath(reviewed[0])): "sha256:reviewed",
            }

            train_cnn_model._split_train_val_records(
                [first],
                assignment_path=assignment_path,
                source_ids_by_path=source_ids,
            )
            train_records, val_records, meta = train_cnn_model._split_train_val_records(
                [first, reviewed],
                assignment_path=assignment_path,
                source_ids_by_path=source_ids,
                adaptive_source_ids={"sha256:reviewed"},
            )

            manifest = lineage.read_json(assignment_path)
            self.assertEqual(train_records, [first, reviewed])
            self.assertEqual(val_records, [])
            self.assertIsNone(meta["validation_cohort_revision"])
            self.assertEqual(
                manifest["assignments"],
                {"sha256:first": "train", "sha256:reviewed": "train"},
            )

    def test_all_hitl_sources_train_without_fabricating_validation(self):
        with tempfile.TemporaryDirectory() as root:
            records = [self._write_record(root, index) for index in range(1, 4)]
            source_ids = {
                os.path.normcase(os.path.abspath(path)): f"sha256:source-{index}"
                for index, (path, _parts) in enumerate(records, start=1)
            }
            adaptive_ids = set(source_ids.values())
            assignment_path = os.path.join(root, "cnn_validation_v1.json")

            train_records, val_records, meta = train_cnn_model._split_train_val_records(
                records,
                assignment_path=assignment_path,
                source_ids_by_path=source_ids,
                adaptive_source_ids=adaptive_ids,
            )

            self.assertEqual(train_records, records)
            self.assertEqual(val_records, [])
            self.assertIsNone(meta["validation_cohort_revision"])
            self.assertFalse(os.path.exists(assignment_path))

    def test_existing_validation_cannot_be_reclassified_as_hitl(self):
        with tempfile.TemporaryDirectory() as root:
            records = [self._write_record(root, index) for index in range(1, 6)]
            assignment_path = os.path.join(root, "cnn_validation_v1.json")
            train_cnn_model._split_train_val_records(
                records,
                assignment_path=assignment_path,
            )
            manifest = lineage.read_json(assignment_path)
            validation_id = next(
                source
                for source, cohort in manifest["assignments"].items()
                if cohort == "val"
            )

            with self.assertRaisesRegex(RuntimeError, "contains model-assisted/HITL sources"):
                train_cnn_model._split_train_val_records(
                    records,
                    assignment_path=assignment_path,
                    adaptive_source_ids={validation_id},
                )

    def test_adaptive_source_ids_are_loaded_from_landmark_cohort(self):
        with tempfile.TemporaryDirectory() as root:
            cohort_path = os.path.join(root, "landmark_benchmark_v1.json")
            lineage.atomic_write_json(
                cohort_path,
                {
                    "sources": {
                        "sha256:manual": {"adaptiveTrainingSample": False},
                        "sha256:reviewed": {"adaptiveTrainingSample": True},
                    }
                },
            )
            self.assertEqual(
                train_cnn_model._adaptive_source_ids_from_split_info(
                    {"cohort_manifest": cohort_path}
                ),
                {"sha256:reviewed"},
            )

    def test_validation_manifest_is_repeatable_and_rejects_content_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            records = [self._write_record(root, index) for index in range(1, 8)]
            assignment_path = os.path.join(root, "cohorts", "cnn_validation_v1.json")
            train_cnn_model._split_train_val_records(
                records,
                val_fraction=0.25,
                seed=42,
                assignment_path=assignment_path,
            )
            first_manifest = lineage.read_json(assignment_path)
            self.assertIs(first_manifest["validationSourceOverlap"], False)
            first_digest = lineage.sha256_file(assignment_path)
            val_source = next(
                source
                for source, cohort in first_manifest["assignments"].items()
                if cohort == "val"
            )

            train_cnn_model._split_train_val_records(
                records,
                val_fraction=0.9,
                seed=999,
                assignment_path=assignment_path,
            )
            second_manifest = lineage.read_json(assignment_path)
            self.assertEqual(lineage.sha256_file(assignment_path), first_digest)
            self.assertEqual(second_manifest["seed"], 42)
            self.assertEqual(second_manifest["validationFraction"], 0.25)
            self.assertEqual(
                second_manifest["validationCohortRevision"],
                first_manifest["validationCohortRevision"],
            )

            changed_records = []
            for image_path, parts in records:
                copied_parts = dict(parts)
                if train_cnn_model._record_source_key(image_path) == val_source:
                    copied_parts["00"] = (copied_parts["00"][0] + 1, copied_parts["00"][1])
                changed_records.append((image_path, copied_parts))
            with self.assertRaisesRegex(RuntimeError, "validation image/landmark content changed"):
                train_cnn_model._split_train_val_records(
                    changed_records,
                    val_fraction=0.25,
                    seed=42,
                    assignment_path=assignment_path,
                )

            missing_records = [
                record
                for record in records
                if train_cnn_model._record_source_key(record[0]) != val_source
            ]
            with self.assertRaisesRegex(RuntimeError, "validation cohort is missing sources"):
                train_cnn_model._split_train_val_records(
                    missing_records,
                    val_fraction=0.25,
                    seed=42,
                    assignment_path=assignment_path,
                )


class CnnValidationManifestIntegrityTests(unittest.TestCase):
    """The frozen CNN cohort must fail closed on a corrupt or stripped manifest.

    Without these gates a truncated manifest reads as ``{}``, the cohort is
    silently rebootstrapped, and the rebuilt manifest overwrites the evidence.
    """

    @staticmethod
    def _write_record(root, index):
        return CnnValidationCohortTests._write_record(root, index)

    def _bootstrap(self, root, count=7, **kwargs):
        """Create a real v3 manifest and return (records, assignment_path)."""
        records = [self._write_record(root, index) for index in range(1, count + 1)]
        assignment_path = os.path.join(root, "cohorts", "cnn_validation_v1.json")
        train_cnn_model._split_train_val_records(
            records,
            val_fraction=kwargs.pop("val_fraction", 0.25),
            seed=kwargs.pop("seed", 42),
            assignment_path=assignment_path,
            **kwargs,
        )
        return records, assignment_path

    def _resplit(self, records, assignment_path, **kwargs):
        return train_cnn_model._split_train_val_records(
            records,
            val_fraction=kwargs.pop("val_fraction", 0.25),
            seed=kwargs.pop("seed", 42),
            assignment_path=assignment_path,
            **kwargs,
        )

    def _mutate(self, assignment_path, mutator):
        manifest = lineage.read_json(assignment_path)
        mutator(manifest)
        lineage.atomic_write_json(assignment_path, manifest)
        return manifest

    # ---------------------------------------------------------------- corrupt

    def test_truncated_manifest_fails_closed_without_destroying_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            with open(assignment_path, "w", encoding="utf-8") as handle:
                handle.write('{"assignments": {"sha256:abc": "va')
            with open(assignment_path, "rb") as handle:
                before = handle.read()

            with self.assertRaisesRegex(RuntimeError, "unreadable or malformed"):
                self._resplit(records, assignment_path)

            # The rebuilt-manifest write at the end of the split must never run.
            with open(assignment_path, "rb") as handle:
                self.assertEqual(handle.read(), before)

    def test_empty_manifest_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            open(assignment_path, "w", encoding="utf-8").close()
            with self.assertRaisesRegex(RuntimeError, "unreadable or malformed"):
                self._resplit(records, assignment_path)

    def test_non_object_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            lineage.atomic_write_json(assignment_path, [])
            with self.assertRaisesRegex(RuntimeError, "must be a JSON object"):
                self._resplit(records, assignment_path)

    def test_malformed_assignments_map_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            self._mutate(assignment_path, lambda m: m.__setitem__("assignments", []))
            with self.assertRaisesRegex(RuntimeError, "malformed assignments map"):
                self._resplit(records, assignment_path)

    # ---------------------------------------------------------- version gating

    def test_unsupported_future_version_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            self._mutate(
                assignment_path,
                lambda m: m.__setitem__(
                    "version", train_cnn_model.CNN_VALIDATION_MANIFEST_VERSION + 1
                ),
            )
            with self.assertRaisesRegex(
                RuntimeError, "Unsupported CNN validation cohort manifest version"
            ):
                self._resplit(records, assignment_path)

    def test_non_integer_version_fails_closed_instead_of_taking_migration_path(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            self._mutate(assignment_path, lambda m: m.__setitem__("version", "three"))
            with self.assertRaisesRegex(RuntimeError, "no recognized version"):
                self._resplit(records, assignment_path)

    def test_legacy_v2_manifest_is_exempt_from_current_version_field_requirements(self):
        with tempfile.TemporaryDirectory() as root:
            records = [self._write_record(root, index) for index in range(1, 8)]
            assignment_path = os.path.join(root, "cohorts", "cnn_validation_v1.json")
            source_keys = sorted(
                train_cnn_model._record_source_key(path) for path, _ in records
            )
            lineage.atomic_write_json(
                assignment_path,
                {
                    "version": 2,
                    "assignments": {
                        source: ("val" if index < 2 else "train")
                        for index, source in enumerate(source_keys)
                    },
                },
            )
            # No assignmentRevision, snapshots, revision, or policy strings: a
            # legacy manifest must still upgrade rather than fail closed.
            self._resplit(records, assignment_path)
            upgraded = lineage.read_json(assignment_path)
            self.assertEqual(
                upgraded["version"], train_cnn_model.CNN_VALIDATION_MANIFEST_VERSION
            )
            self.assertTrue(upgraded["assignmentRevision"])

    # ------------------------------------------- current-version field stripping

    def test_stripping_revision_and_snapshots_fails_closed(self):
        """The reported bypass: keep assignments, delete the integrity fields."""
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)

            def strip(manifest):
                manifest.pop("validationCohortRevision", None)
                manifest.pop("validationSourceSnapshots", None)

            self._mutate(assignment_path, strip)
            with self.assertRaisesRegex(
                RuntimeError, "missing exact snapshot/revision metadata"
            ):
                self._resplit(records, assignment_path)

    def test_missing_assignment_revision_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            self._mutate(assignment_path, lambda m: m.pop("assignmentRevision", None))
            with self.assertRaisesRegex(
                RuntimeError, "assignment revision is missing or corrupt"
            ):
                self._resplit(records, assignment_path)

    def test_tampered_assignments_fail_closed_against_the_stored_revision(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)

            def flip(manifest):
                for source, cohort in manifest["assignments"].items():
                    if cohort == "train":
                        manifest["assignments"][source] = "val"
                        return

            self._mutate(assignment_path, flip)
            with self.assertRaisesRegex(
                RuntimeError, "assignment revision is missing or corrupt"
            ):
                self._resplit(records, assignment_path)

    def test_unknown_cohort_value_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)

            def relabel(manifest):
                source = next(iter(manifest["assignments"]))
                manifest["assignments"][source] = "holdout"

            self._mutate(assignment_path, relabel)
            with self.assertRaisesRegex(
                RuntimeError, "invalid cohort assignment"
            ):
                self._resplit(records, assignment_path)

    def test_removing_one_validation_snapshot_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)

            def drop(manifest):
                manifest["validationSourceSnapshots"].pop(
                    next(iter(manifest["validationSourceSnapshots"]))
                )

            self._mutate(assignment_path, drop)
            with self.assertRaisesRegex(
                RuntimeError, "missing exact snapshot/revision metadata"
            ):
                self._resplit(records, assignment_path)

    def test_malformed_snapshot_map_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            self._mutate(
                assignment_path,
                lambda m: m.__setitem__("validationSourceSnapshots", []),
            )
            with self.assertRaisesRegex(
                RuntimeError, "malformed validationSourceSnapshots"
            ):
                self._resplit(records, assignment_path)

    def test_manifest_declaring_an_adaptive_validation_source_fails_closed(self):
        """The HITL gate must hold from the manifest alone, with no cohort file."""
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)

            def contaminate(manifest):
                val_source = next(
                    source
                    for source, cohort in manifest["assignments"].items()
                    if cohort == "val"
                )
                manifest["adaptiveTrainSources"] = [val_source]

            self._mutate(assignment_path, contaminate)
            with self.assertRaisesRegex(
                RuntimeError, "manifest itself declares as model-assisted/HITL"
            ):
                self._resplit(records, assignment_path)

    def test_malformed_adaptive_train_sources_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            self._mutate(
                assignment_path,
                lambda m: m.__setitem__("adaptiveTrainSources", {"a": 1}),
            )
            with self.assertRaisesRegex(
                RuntimeError, "malformed adaptiveTrainSources"
            ):
                self._resplit(records, assignment_path)

    def test_asserted_source_overlap_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            self._mutate(
                assignment_path,
                lambda m: m.__setitem__("validationSourceOverlap", True),
            )
            with self.assertRaisesRegex(RuntimeError, "disjoint train/validation"):
                self._resplit(records, assignment_path)

    def test_unrecognized_split_policies_fail_closed(self):
        for field, pattern in (
            ("newSourcePolicy", "unrecognized new-source policy"),
            ("adaptiveSourcePolicy", "unrecognized adaptive-source policy"),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as root:
                    records, assignment_path = self._bootstrap(root)
                    self._mutate(
                        assignment_path, lambda m: m.__setitem__(field, "anything_goes")
                    )
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        self._resplit(records, assignment_path)

    def test_malformed_seed_and_validation_fraction_fail_closed_with_context(self):
        cases = (
            ("seed", "not-a-seed", "malformed seed"),
            ("validationFraction", "many", "malformed validationFraction"),
            ("validationFraction", 1.5, r"validationFraction outside \[0, 1\]"),
        )
        for field, value, pattern in cases:
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as root:
                    records, assignment_path = self._bootstrap(root)
                    self._mutate(
                        assignment_path, lambda m: m.__setitem__(field, value)
                    )
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        self._resplit(records, assignment_path)

    # ------------------------------------------------- landmark cohort manifest

    def test_corrupt_landmark_cohort_manifest_cannot_silently_disable_hitl_gate(self):
        with tempfile.TemporaryDirectory() as root:
            cohort_path = os.path.join(root, "landmark_benchmark_v1.json")
            with open(cohort_path, "w", encoding="utf-8") as handle:
                handle.write("{")
            with self.assertRaisesRegex(RuntimeError, "unreadable or malformed"):
                train_cnn_model._adaptive_source_ids_from_split_info(
                    {"cohort_manifest": cohort_path}
                )

    def test_landmark_cohort_manifest_with_non_dict_sources_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            cohort_path = os.path.join(root, "landmark_benchmark_v1.json")
            lineage.atomic_write_json(cohort_path, {"sources": []})
            with self.assertRaisesRegex(RuntimeError, "malformed sources map"):
                train_cnn_model._adaptive_source_ids_from_split_info(
                    {"cohort_manifest": cohort_path}
                )

    def test_missing_landmark_cohort_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            cohort_path = os.path.join(root, "cohorts", "landmark_benchmark_v1.json")
            with self.assertRaisesRegex(
                RuntimeError, "referenced by split_info is missing"
            ):
                train_cnn_model._adaptive_source_ids_from_split_info(
                    {"cohort_manifest": cohort_path}
                )

    def test_relocated_session_resolves_cohort_manifest_from_debug_dir(self):
        """An absolute path baked into split_info goes stale when a session moves."""
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            actual = os.path.join(debug_dir, "cohorts", "landmark_benchmark_v1.json")
            lineage.atomic_write_json(
                actual, {"sources": {"sha256:reviewed": {"adaptiveTrainingSample": True}}}
            )
            stale = os.path.join(
                root, "elsewhere", "debug", "cohorts", "landmark_benchmark_v1.json"
            )
            self.assertEqual(
                train_cnn_model._adaptive_source_ids_from_split_info(
                    {"cohort_manifest": stale}, debug_dir=debug_dir
                ),
                {"sha256:reviewed"},
            )

    def test_cohort_manifest_digest_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            cohort_path = os.path.join(root, "landmark_benchmark_v1.json")
            lineage.atomic_write_json(cohort_path, {"sources": {}})
            with self.assertRaisesRegex(RuntimeError, "does not match the digest"):
                train_cnn_model._adaptive_source_ids_from_split_info(
                    {
                        "cohort_manifest": cohort_path,
                        "cohort_manifest_sha256": "0" * 64,
                    }
                )

    def test_versionless_landmark_cohort_manifest_still_loads_adaptive_ids(self):
        """Regression guard: this reader must stay version-agnostic.

        It consumes both the native landmark manifest and the imported-dlib
        manifest, neither of which is guaranteed to carry a version key.
        """
        with tempfile.TemporaryDirectory() as root:
            cohort_path = os.path.join(root, "landmark_benchmark_v1.json")
            lineage.atomic_write_json(
                cohort_path,
                {"sources": {"sha256:reviewed": {"adaptiveTrainingSample": True}}},
            )
            self.assertEqual(
                train_cnn_model._adaptive_source_ids_from_split_info(
                    {"cohort_manifest": cohort_path}
                ),
                {"sha256:reviewed"},
            )

    def test_imported_dlib_manifest_without_sources_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            cohort_path = os.path.join(root, "imported_dlib_fish.json")
            lineage.atomic_write_json(
                cohort_path, {"source": "imported_dlib_xml", "train": {}}
            )
            self.assertEqual(
                train_cnn_model._adaptive_source_ids_from_split_info(
                    {"cohort_manifest": cohort_path}
                ),
                set(),
            )

    # ------------------------------------------------------------- split_info

    def test_corrupt_split_info_does_not_downgrade_source_identity(self):
        with tempfile.TemporaryDirectory() as root:
            split_path = os.path.join(root, "split_info_fish.json")
            with open(split_path, "w", encoding="utf-8") as handle:
                handle.write("{")
            with self.assertRaisesRegex(RuntimeError, "unreadable or malformed"):
                train_cnn_model._read_split_info_strict(split_path)

    def test_malformed_split_info_source_map_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            split_path = os.path.join(root, "split_info_fish.json")
            lineage.atomic_write_json(split_path, {"train_file_source_ids": []})
            with self.assertRaisesRegex(
                RuntimeError, "malformed train_file_source_ids"
            ):
                train_cnn_model._read_split_info_strict(split_path)

    def test_absent_split_info_is_still_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(
                train_cnn_model._read_split_info_strict(
                    os.path.join(root, "split_info_fish.json")
                ),
                {},
            )

    # --------------------------------------------------- over-strictness guard

    def test_manifest_written_by_this_module_passes_every_strict_check(self):
        """The gates must never reject a manifest the writer itself produced."""
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            digest = lineage.sha256_file(assignment_path)

            # Different fraction and seed: the locked cohort must win, unchanged.
            self._resplit(records, assignment_path, val_fraction=0.9, seed=999)
            self.assertEqual(lineage.sha256_file(assignment_path), digest)

            self._resplit(records, assignment_path)
            self.assertEqual(lineage.sha256_file(assignment_path), digest)

    def test_first_run_with_no_manifest_still_bootstraps(self):
        with tempfile.TemporaryDirectory() as root:
            records, assignment_path = self._bootstrap(root)
            self.assertTrue(os.path.exists(assignment_path))
            manifest = lineage.read_json(assignment_path)
            self.assertEqual(
                manifest["version"], train_cnn_model.CNN_VALIDATION_MANIFEST_VERSION
            )


if __name__ == "__main__":
    unittest.main()
