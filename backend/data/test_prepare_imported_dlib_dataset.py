import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from backend.bv_utils import lineage
from backend.bv_utils import orientation_utils as ou
from backend.bv_utils.landmark_artifacts import bundle_id_mapping
from backend.data.prepare_imported_dlib_dataset import (
    ImportedDlibContractError,
    prepare_or_verify,
)


DEFAULT_POLICY = {
    "mode": "directional",
    "targetOrientation": "left",
    "anteriorAnchorIds": [3],
    "posteriorAnchorIds": [12],
    "headCategories": ["head"],
    "tailCategories": ["tail"],
    "obbLevelingMode": "on",
}
DEFAULT_TEMPLATE = [
    {"index": 3, "name": "snout", "category": "head", "required": True},
    {"index": 12, "name": "tail base", "category": "tail", "required": True},
]


class ImportedDlibFixtureMixin:
    def _write_session(self, root: Path, *, policy=None, template=None):
        (root / "session.json").write_text(
            json.dumps(
                {
                    "orientationPolicyConfigured": True,
                    "orientationPolicy": policy or DEFAULT_POLICY,
                    "landmarkTemplate": template or DEFAULT_TEMPLATE,
                    "schemaSemanticFingerprint": "schema-semantic-test",
                    "schemaSemanticVersion": 2,
                }
            ),
            encoding="utf-8",
        )

    def _image(self, root: Path, name: str, value: int) -> Path:
        path = root / name
        image = np.full((48, 96, 3), value, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def _xml(self, path: Path, image_paths: list[Path], *, slots=(0, 1), names=None):
        """Write a source dlib XML.

        `slots` orders the parts; the first is placed at x=68 and the rest at
        x=28, so a directional import has to flip the crop.  `names` overrides
        the raw part-name strings independently of the slot values, which is how
        alias and multi-digit cases are exercised.
        """
        raw_names = list(names) if names is not None else [str(slot) for slot in slots]
        dataset = ET.Element("dataset")
        images = ET.SubElement(dataset, "images")
        for image_path in image_paths:
            image = ET.SubElement(images, "image", file=str(image_path.resolve()))
            box = ET.SubElement(image, "box", left="8", top="4", width="80", height="40")
            for position, name in enumerate(raw_names):
                # Anterior is initially to the right; directional import must flip it.
                ET.SubElement(
                    box,
                    "part",
                    name=str(name),
                    x="68" if position == 0 else "28",
                    y="22",
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(dataset).write(path, encoding="utf-8", xml_declaration=True)

    def _xml_with_geometry(self, path: Path, image_paths: list[Path], parts):
        """Write a source XML with explicit (name, x, y) triples per box."""
        dataset = ET.Element("dataset")
        images = ET.SubElement(dataset, "images")
        for image_path in image_paths:
            image = ET.SubElement(images, "image", file=str(image_path.resolve()))
            box = ET.SubElement(image, "box", left="8", top="4", width="80", height="40")
            for name, x, y in parts:
                ET.SubElement(box, "part", name=str(name), x=str(x), y=str(y))
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(dataset).write(path, encoding="utf-8", xml_declaration=True)

    def _prepare(self, root: Path, tag: str, **kwargs):
        options = {
            "mode": "prepare",
            "validation_mode": "derive",
            "test_mode": "derive",
            "confirm_template_order": False,
            "validation_fraction": 0.2,
            "seed": 42,
        }
        options.update(kwargs)
        return prepare_or_verify(str(root), tag, **options)

    @staticmethod
    def _canonical_parts(xml_path: Path):
        """Return [{name: (x, y)}] for each box in a prepared XML."""
        boxes = ET.parse(xml_path).getroot().findall("./images/image/box")
        return [
            {
                part.get("name"): (int(part.get("x")), int(part.get("y")))
                for part in box.findall("part")
            }
            for box in boxes
        ]


class ImportedDlibDatasetTests(ImportedDlibFixtureMixin, unittest.TestCase):

    def test_confirmed_mapping_canonicalizes_and_freezes_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            train_images = [self._image(root, f"train-{index}.png", 30 + index) for index in range(3)]
            test_image = self._image(root, "test.png", 90)
            self._xml(root / "xml" / "train_fish.xml", train_images)
            self._xml(root / "xml" / "test_fish.xml", [test_image])

            with self.assertRaises(ImportedDlibContractError) as pending:
                prepare_or_verify(
                    str(root),
                    "fish",
                    mode="prepare",
                    validation_mode="derive",
                    test_mode="explicit",
                    confirm_template_order=False,
                    validation_fraction=0.2,
                    seed=42,
                )
            self.assertEqual(pending.exception.code, "mapping_confirmation_required")
            self.assertTrue(pending.exception.details["requiresMappingConfirmation"])

            prepared = prepare_or_verify(
                str(root),
                "fish",
                mode="prepare",
                validation_mode="derive",
                test_mode="explicit",
                confirm_template_order=True,
                validation_fraction=0.2,
                seed=42,
            )
            self.assertTrue(prepared["ok"])
            self.assertEqual(prepared["mappingMode"], "confirmed_template_order")
            self.assertEqual(prepared["trainImages"], 2)
            self.assertEqual(prepared["validationImages"], 1)
            self.assertEqual(prepared["testImages"], 1)

            mapping = json.loads((root / "debug" / "id_mapping_fish.json").read_text())
            self.assertEqual(mapping["dlib_index_to_original"], {"0": 3, "1": 12})
            self.assertEqual(mapping["landmark_template"]["3"]["count"], 2)
            self.assertEqual(mapping["training_config"]["orientation_policy"]["mode"], "directional")
            self.assertTrue(mapping["training_config"]["canonical_training_enabled"])

            for cohort in ("train", "validation", "test"):
                xml_path = root / "xml" / f"{cohort}_fish.xml"
                boxes = ET.parse(xml_path).getroot().findall("./images/image/box")
                self.assertTrue(boxes)
                for box in boxes:
                    self.assertEqual(
                        (box.get("left"), box.get("top"), box.get("width"), box.get("height")),
                        ("0", "0", "512", "512"),
                    )
                    parts = {part.get("name"): int(part.get("x")) for part in box.findall("part")}
                    self.assertEqual(set(parts), {"00", "01"})
                    self.assertLess(parts["00"], parts["01"])

            verified = prepare_or_verify(
                str(root),
                "fish",
                mode="verify",
                validation_mode="derive",
                test_mode="explicit",
                confirm_template_order=False,
                validation_fraction=0.2,
                seed=42,
            )
            self.assertEqual(
                verified["validationCohortRevision"], prepared["validationCohortRevision"]
            )

    def test_incomplete_slots_do_not_invent_optional_or_required_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"sample-{index}.png", 10 + index) for index in range(2)]
            self._xml(root / "xml" / "train_incomplete.xml", images, slots=(0,))
            with self.assertRaises(ImportedDlibContractError) as raised:
                prepare_or_verify(
                    str(root),
                    "incomplete",
                    mode="prepare",
                    validation_mode="derive",
                    test_mode="none",
                    confirm_template_order=True,
                    validation_fraction=0.2,
                    seed=42,
                )
            self.assertEqual(raised.exception.code, "ambiguous_schema_mapping")

    def test_explicit_schema_ids_derive_disjoint_test_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"direct-{index}.png", 100 + index) for index in range(3)]
            self._xml(root / "xml" / "train_direct.xml", images, slots=(3, 12))
            prepared = prepare_or_verify(
                str(root),
                "direct",
                mode="prepare",
                validation_mode="derive",
                test_mode="derive",
                confirm_template_order=False,
                validation_fraction=0.2,
                seed=42,
            )
            self.assertEqual(prepared["mappingMode"], "explicit_schema_ids")
            self.assertEqual(
                (prepared["trainImages"], prepared["validationImages"], prepared["testImages"]),
                (1, 1, 1),
            )
            mapping = json.loads((root / "debug" / "id_mapping_direct.json").read_text())
            self.assertEqual(mapping["dlib_index_to_original"], {"0": 3, "1": 12})
            split = json.loads((root / "debug" / "split_info_direct.json").read_text())
            self.assertTrue(split["validationCohortRevision"])
            self.assertTrue(split["testCohortRevision"])
            self.assertFalse(split["validationSourceOverlap"])

    def test_content_overlap_between_train_and_test_is_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            shared = self._image(root, "shared.png", 70)
            other = self._image(root, "other.png", 71)
            train_path = root / "xml" / "train_overlap.xml"
            test_path = root / "xml" / "test_overlap.xml"
            self._xml(train_path, [shared, other])
            self._xml(test_path, [shared])
            train_before = train_path.read_bytes()
            test_before = test_path.read_bytes()
            with self.assertRaises(ImportedDlibContractError) as raised:
                prepare_or_verify(
                    str(root),
                    "overlap",
                    mode="prepare",
                    validation_mode="derive",
                    test_mode="explicit",
                    confirm_template_order=True,
                    validation_fraction=0.2,
                    seed=42,
                )
            self.assertEqual(raised.exception.code, "cohort_source_overlap")
            self.assertEqual(train_path.read_bytes(), train_before)
            self.assertEqual(test_path.read_bytes(), test_before)
            self.assertFalse((root / "xml" / "validation_overlap.xml").exists())
            crop_dir = root / "corrected_images" / "imported_dlib_overlap"
            self.assertFalse(crop_dir.exists() and any(crop_dir.iterdir()))
            self.assertFalse((root / "debug" / "id_mapping_overlap.json").exists())
            self.assertFalse((root / "debug" / "split_info_overlap.json").exists())


class ImportedDlibMappingContractTests(ImportedDlibFixtureMixin, unittest.TestCase):
    """How raw XML part names become schema IDs, and how they must fail."""

    def test_multi_digit_part_names_survive_the_lexical_ordering_trap(self):
        """Source names 1, 2, 10 must stay in numeric order after canonicalization.

        dlib indexes parts by lexical sort of their names, so raw names 1/2/10
        would be read as 1/10/2.  Canonical rewriting to 00/01/02 is what keeps
        inference (predict.py) and the trainer's metric sort agreeing.
        """
        template = [
            {"index": 1, "name": "snout", "category": "head", "required": True},
            {"index": 2, "name": "dorsal", "category": "body", "required": True},
            {"index": 10, "name": "tail base", "category": "tail", "required": True},
        ]
        policy = dict(DEFAULT_POLICY, anteriorAnchorIds=[1], posteriorAnchorIds=[10])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root, policy=policy, template=template)
            images = [self._image(root, f"lex-{index}.png", 40 + index) for index in range(3)]
            # Anterior already left of posterior: no directional flip, so the
            # x ordering below is preserved verbatim through canonicalization.
            self._xml_with_geometry(
                root / "xml" / "train_lex.xml",
                images,
                [("1", 28, 22), ("2", 48, 22), ("10", 68, 22)],
            )

            prepared = self._prepare(root, "lex")
            self.assertTrue(prepared["ok"])
            self.assertEqual(prepared["mappingMode"], "explicit_schema_ids")

            mapping = json.loads((root / "debug" / "id_mapping_lex.json").read_text())
            # Canonical slot order follows the numeric slot order 1, 2, 10 -
            # never the lexical order 1, 10, 2.
            self.assertEqual(
                mapping["dlib_index_to_original"], {"0": 1, "1": 2, "2": 10}
            )
            self.assertEqual(mapping["part_names_sorted"], ["00", "01", "02"])
            self.assertEqual(
                mapping["dlib_name_to_original"], {"00": 1, "01": 2, "02": 10}
            )

            for cohort in ("train", "validation", "test"):
                for parts in self._canonical_parts(root / "xml" / f"{cohort}_lex.xml"):
                    self.assertEqual(set(parts), {"00", "01", "02"})
                    self.assertLess(parts["00"][0], parts["01"][0])
                    self.assertLess(parts["01"][0], parts["02"][0])

    def test_explicit_schema_id_maps_are_complete_and_mutually_inverse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"inv-{index}.png", 60 + index) for index in range(3)]
            self._xml(root / "xml" / "train_inv.xml", images, slots=(3, 12))

            self._prepare(root, "inv")
            mapping = json.loads((root / "debug" / "id_mapping_inv.json").read_text())

            index_to_original = mapping["dlib_index_to_original"]
            self.assertEqual(sorted(index_to_original.values()), [3, 12])
            self.assertEqual(
                {int(name): schema for name, schema in mapping["dlib_name_to_original"].items()},
                {int(slot): schema for slot, schema in index_to_original.items()},
            )
            for slot, schema_id in index_to_original.items():
                self.assertEqual(int(mapping["original_to_dlib"][str(schema_id)]), int(slot))
            self.assertEqual(mapping["num_landmarks"], 2)
            self.assertEqual(mapping["original_ids"], [3, 12])
            self.assertEqual(mapping["excluded_ids"], [])
            self.assertEqual(mapping["standard_size"], 512)

    def test_zero_based_slots_require_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"zero-{index}.png", 20 + index) for index in range(3)]
            self._xml(root / "xml" / "train_zero.xml", images, slots=(0, 1))

            with self.assertRaises(ImportedDlibContractError) as pending:
                self._prepare(root, "zero", confirm_template_order=False)
            self.assertEqual(pending.exception.code, "mapping_confirmation_required")
            proposal = pending.exception.details["mappingProposal"]
            self.assertEqual([entry["schemaId"] for entry in proposal], [3, 12])

            prepared = self._prepare(root, "zero", confirm_template_order=True)
            self.assertEqual(prepared["mappingMode"], "confirmed_template_order")
            mapping = json.loads((root / "debug" / "id_mapping_zero.json").read_text())
            self.assertEqual(mapping["dlib_index_to_original"], {"0": 3, "1": 12})

    def test_duplicate_part_names_in_one_box_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"dup-{index}.png", 80 + index) for index in range(2)]
            # "1" and "01" both resolve to dlib slot 1.
            self._xml_with_geometry(
                root / "xml" / "train_dup.xml",
                images,
                [("3", 28, 22), ("1", 48, 22), ("01", 68, 22)],
            )
            with self.assertRaises(ImportedDlibContractError) as raised:
                self._prepare(root, "dup", confirm_template_order=True)
            self.assertEqual(raised.exception.code, "duplicate_part_slot")

    def test_mixed_zero_padding_spellings_are_rejected(self):
        """"12" and "012" are distinct part names to dlib and must not mix.

        dlib indexes parts by lexical name sort, so an XML that spells one slot
        two ways was assembled from inconsistent sources; silently collapsing
        them would hide a real landmark-contract inconsistency.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            image_paths = [self._image(root, f"alias-{index}.png", 81 + index) for index in range(4)]
            dataset = ET.Element("dataset")
            images = ET.SubElement(dataset, "images")
            for index, image_path in enumerate(image_paths):
                image = ET.SubElement(images, "image", file=str(image_path.resolve()))
                box = ET.SubElement(image, "box", left="8", top="4", width="80", height="40")
                ET.SubElement(box, "part", name="3", x="28", y="22")
                ET.SubElement(
                    box, "part", name="12" if index % 2 == 0 else "012", x="68", y="22"
                )
            path = root / "xml" / "train_alias.xml"
            path.parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(dataset).write(path, encoding="utf-8", xml_declaration=True)

            with self.assertRaises(ImportedDlibContractError) as raised:
                self._prepare(root, "alias")
            self.assertEqual(raised.exception.code, "ambiguous_part_slot")
            self.assertFalse((root / "debug" / "id_mapping_alias.json").exists())

    def test_one_consistent_spelling_is_accepted(self):
        """Zero-padded names are fine as long as the dataset is consistent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"pad-{index}.png", 86 + index) for index in range(4)]
            self._xml_with_geometry(
                root / "xml" / "train_pad.xml",
                images,
                [("03", 28, 22), ("12", 68, 22)],
            )
            self._prepare(root, "pad")
            mapping = json.loads((root / "debug" / "id_mapping_pad.json").read_text())
            self.assertEqual(mapping["dlib_index_to_original"], {"0": 3, "1": 12})
            for cohort in ("train", "validation", "test"):
                for parts in self._canonical_parts(root / "xml" / f"{cohort}_pad.xml"):
                    self.assertEqual(set(parts), {"00", "01"})

    def test_non_integer_part_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"named-{index}.png", 85 + index) for index in range(2)]
            self._xml(root / "xml" / "train_named.xml", images, names=("snout", "tail"))
            with self.assertRaises(ImportedDlibContractError) as raised:
                self._prepare(root, "named", confirm_template_order=True)
            self.assertEqual(raised.exception.code, "invalid_part_name")

    def test_boxes_with_inconsistent_part_sets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            complete = self._image(root, "complete.png", 91)
            partial = self._image(root, "partial.png", 92)
            dataset = ET.Element("dataset")
            images = ET.SubElement(dataset, "images")
            image = ET.SubElement(images, "image", file=str(complete.resolve()))
            box = ET.SubElement(image, "box", left="8", top="4", width="80", height="40")
            ET.SubElement(box, "part", name="3", x="28", y="22")
            ET.SubElement(box, "part", name="12", x="68", y="22")
            image = ET.SubElement(images, "image", file=str(partial.resolve()))
            box = ET.SubElement(image, "box", left="8", top="4", width="80", height="40")
            ET.SubElement(box, "part", name="3", x="28", y="22")
            path = root / "xml" / "train_partial.xml"
            path.parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(dataset).write(path, encoding="utf-8", xml_declaration=True)

            with self.assertRaises(ImportedDlibContractError) as raised:
                self._prepare(root, "partial")
            self.assertEqual(raised.exception.code, "inconsistent_box_parts")

    def test_train_and_test_part_vocabularies_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            train_images = [self._image(root, f"voc-{index}.png", 50 + index) for index in range(2)]
            test_image = self._image(root, "voc-test.png", 59)
            self._xml(root / "xml" / "train_voc.xml", train_images, slots=(3, 12))
            self._xml(root / "xml" / "test_voc.xml", [test_image], slots=(0, 1))

            with self.assertRaises(ImportedDlibContractError) as raised:
                self._prepare(root, "voc", test_mode="explicit")
            self.assertEqual(raised.exception.code, "cohort_part_mismatch")

    def test_unknown_schema_ids_cannot_be_positionally_inferred(self):
        """A slot set that is neither schema IDs nor template-sized must fail."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"extra-{index}.png", 70 + index) for index in range(2)]
            self._xml(root / "xml" / "train_extra.xml", images, slots=(3, 12, 99))
            with self.assertRaises(ImportedDlibContractError) as raised:
                self._prepare(root, "extra", confirm_template_order=True)
            self.assertEqual(raised.exception.code, "ambiguous_schema_mapping")

    def test_missing_required_landmarks_are_rejected(self):
        """A partial explicit-ID set is refused rather than positionally guessed.

        Slots {3, 7} are real schema IDs but omit required ID 12, so the
        identity branch does not apply; the count also differs from the template
        size, so the importer refuses to invent which landmark was omitted
        instead of silently mapping 3 -> 3 and 7 -> 12.
        """
        template = DEFAULT_TEMPLATE + [
            {"index": 7, "name": "eye", "category": "head", "required": False}
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root, template=template)
            images = [self._image(root, f"req-{index}.png", 75 + index) for index in range(2)]
            # Optional 7 plus required 3, but required 12 is absent.
            self._xml(root / "xml" / "train_req.xml", images, slots=(3, 7))
            with self.assertRaises(ImportedDlibContractError) as raised:
                self._prepare(root, "req", confirm_template_order=True)
            self.assertEqual(raised.exception.code, "ambiguous_schema_mapping")

    def test_confirmed_positional_mapping_matches_exactly_what_was_proposed(self):
        """Confirmation must commit the shown proposal, not a re-derived guess.

        Slots {3, 7, 99} are not a valid schema-ID set, so the importer falls
        back to positional order.  It refuses without confirmation, and once
        confirmed the persisted mapping must equal the proposal byte for byte -
        the user approved that specific association and nothing else.
        """
        template = DEFAULT_TEMPLATE + [
            {"index": 7, "name": "eye", "category": "head", "required": False}
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root, template=template)
            images = [self._image(root, f"req2-{index}.png", 77 + index) for index in range(4)]
            self._xml(root / "xml" / "train_req2.xml", images, slots=(3, 7, 99))

            with self.assertRaises(ImportedDlibContractError) as pending:
                self._prepare(root, "req2", confirm_template_order=False)
            self.assertEqual(pending.exception.code, "mapping_confirmation_required")
            proposal = pending.exception.details["mappingProposal"]
            # Template order is 3, 12, 7; slot order is 3, 7, 99.
            self.assertEqual(
                [(entry["dlibSlot"], entry["schemaId"]) for entry in proposal],
                [(3, 3), (7, 12), (99, 7)],
            )

            self._prepare(root, "req2", confirm_template_order=True)
            mapping = json.loads((root / "debug" / "id_mapping_req2.json").read_text())
            # Canonical slots 0/1/2 correspond to source slots 3/7/99 in order.
            self.assertEqual(
                mapping["dlib_index_to_original"], {"0": 3, "1": 12, "2": 7}
            )
            self.assertEqual(
                mapping["mapping_confirmation"]["proposal"], proposal
            )
            self.assertTrue(mapping["mapping_confirmation"]["confirmed"])


class ImportedDlibTemplateAndPolicyTests(ImportedDlibFixtureMixin, unittest.TestCase):
    """The mapping sidecar's statistics and orientation snapshot."""

    def test_landmark_template_is_finite_in_512_space_and_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"tpl-{index}.png", 30 + index) for index in range(5)]
            self._xml(root / "xml" / "train_tpl.xml", images, slots=(3, 12))

            prepared = self._prepare(root, "tpl")
            mapping = json.loads((root / "debug" / "id_mapping_tpl.json").read_text())
            template = mapping["landmark_template"]

            # Key set is exactly the mapped output IDs.
            self.assertEqual(
                sorted(int(key) for key in template),
                sorted(mapping["dlib_index_to_original"].values()),
            )
            for key, stats in template.items():
                with self.subTest(landmark=key):
                    for field in ("x_mean", "y_mean", "x_std", "y_std"):
                        value = float(stats[field])
                        self.assertTrue(np.isfinite(value), f"{field} must be finite")
                    self.assertGreaterEqual(float(stats["x_std"]), 0.0)
                    self.assertGreaterEqual(float(stats["y_std"]), 0.0)
                    self.assertTrue(0.0 <= float(stats["x_mean"]) <= 511.0)
                    self.assertTrue(0.0 <= float(stats["y_mean"]) <= 511.0)
                    # Train cohort only: validation and test boxes must not
                    # leak their geometry into runtime orientation scoring.
                    self.assertEqual(stats["count"], prepared["trainImages"])

            self.assertEqual(
                prepared["trainImages"]
                + prepared["validationImages"]
                + prepared["testImages"],
                len(images),
            )
            self.assertGreater(prepared["validationImages"], 0)
            self.assertGreater(prepared["testImages"], 0)

    def test_every_orientation_mode_snapshots_its_exact_policy(self):
        cases = {
            "directional": DEFAULT_POLICY,
            "bilateral": dict(DEFAULT_POLICY, mode="bilateral", bilateralPairs=[[3, 12]]),
            "axial": dict(DEFAULT_POLICY, mode="axial"),
            "invariant": dict(DEFAULT_POLICY, mode="invariant"),
        }
        for mode, policy in cases.items():
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    self._write_session(root, policy=policy)
                    images = [
                        self._image(root, f"{mode}-{index}.png", 30 + index)
                        for index in range(3)
                    ]
                    self._xml(root / "xml" / f"train_{mode}.xml", images, slots=(3, 12))

                    self._prepare(root, mode)
                    mapping = json.loads(
                        (root / "debug" / f"id_mapping_{mode}.json").read_text()
                    )
                    config = mapping["training_config"]

                    self.assertEqual(config["orientation_mode"], mode)
                    # The sidecar validator rejects any drift between these two.
                    self.assertEqual(config["orientation_policy"]["mode"], mode)
                    self.assertEqual(config["orientation_policy"], policy)
                    self.assertEqual(
                        config["target_orientation"], policy.get("targetOrientation")
                    )
                    self.assertTrue(config["imported_dlib_xml"])
                    # canonical_training_enabled must describe actual geometry
                    # transformation, not merely "mode != invariant".
                    self.assertEqual(
                        config["canonical_training_enabled"], mode == "directional"
                    )

    def test_directional_import_normalizes_every_sample_to_the_target(self):
        for target in ("left", "right"):
            with self.subTest(target=target):
                policy = dict(DEFAULT_POLICY, targetOrientation=target)
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    self._write_session(root, policy=policy)
                    images = [
                        self._image(root, f"dir-{index}.png", 30 + index) for index in range(3)
                    ]
                    # Anterior (3) written at x=68, posterior (12) at x=28.
                    self._xml(root / "xml" / f"train_{target}.xml", images, slots=(3, 12))

                    self._prepare(root, target)
                    mapping = json.loads(
                        (root / "debug" / f"id_mapping_{target}.json").read_text()
                    )
                    anterior_slot = mapping["original_to_dlib"]["3"]
                    posterior_slot = mapping["original_to_dlib"]["12"]
                    width = mapping["part_name_width"]
                    anterior_name = f"{int(anterior_slot):0{width}d}"
                    posterior_name = f"{int(posterior_slot):0{width}d}"

                    for cohort in ("train", "validation", "test"):
                        xml_path = root / "xml" / f"{cohort}_{target}.xml"
                        parts_per_box = self._canonical_parts(xml_path)
                        self.assertTrue(parts_per_box)
                        for parts in parts_per_box:
                            anterior_x = parts[anterior_name][0]
                            posterior_x = parts[posterior_name][0]
                            if target == "left":
                                self.assertLess(anterior_x, posterior_x)
                            else:
                                self.assertGreater(anterior_x, posterior_x)

    def test_bilateral_pairs_produce_a_correct_dlib_name_swap_map(self):
        """The flip augmentation in train_shape_model consumes this map."""
        policy = dict(DEFAULT_POLICY, mode="bilateral", bilateralPairs=[[3, 12]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root, policy=policy)
            images = [self._image(root, f"bil-{index}.png", 30 + index) for index in range(3)]
            self._xml(root / "xml" / "train_bil.xml", images, slots=(3, 12))

            self._prepare(root, "bil")
            mapping = json.loads((root / "debug" / "id_mapping_bil.json").read_text())

            name_to_original = mapping["dlib_name_to_original"]
            self.assertEqual(sorted(name_to_original.values()), [3, 12])

            swap = ou.build_dlib_name_swap_map(mapping, {3: 12, 12: 3})
            left_name = mapping["original_to_dlib"]["3"]
            right_name = mapping["original_to_dlib"]["12"]
            width = mapping["part_name_width"]
            left_name = f"{int(left_name):0{width}d}"
            right_name = f"{int(right_name):0{width}d}"

            self.assertEqual(swap.get(left_name), right_name)
            self.assertEqual(swap.get(right_name), left_name)
            # A swap map must be an involution over the paired names.
            for source, destination in swap.items():
                self.assertEqual(swap[destination], source)

    def test_bilateral_pair_ids_must_exist_in_the_imported_xml(self):
        template = DEFAULT_TEMPLATE + [
            {"index": 7, "name": "left eye", "category": "head", "required": False}
        ]
        policy = dict(DEFAULT_POLICY, mode="bilateral", bilateralPairs=[[3, 7]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root, policy=policy, template=template)
            images = [self._image(root, f"bilmiss-{index}.png", 30 + index) for index in range(2)]
            self._xml(root / "xml" / "train_bilmiss.xml", images, slots=(3, 12))
            with self.assertRaises(ImportedDlibContractError) as raised:
                self._prepare(root, "bilmiss")
            self.assertEqual(raised.exception.code, "missing_orientation_landmarks")


class ImportedDlibCohortStabilityTests(ImportedDlibFixtureMixin, unittest.TestCase):
    def test_unchanged_data_yields_a_stable_cohort_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"stab-{index}.png", 30 + index) for index in range(5)]
            self._xml(root / "xml" / "train_stab.xml", images, slots=(3, 12))

            prepared = self._prepare(root, "stab")
            split = json.loads((root / "debug" / "split_info_stab.json").read_text())
            cohort = json.loads(
                (root / "debug" / "cohorts" / "imported_dlib_stab.json").read_text()
            )

            verified = prepare_or_verify(
                str(root),
                "stab",
                mode="verify",
                validation_mode="derive",
                test_mode="derive",
                confirm_template_order=False,
                validation_fraction=0.2,
                seed=42,
            )
            self.assertEqual(
                verified["validationCohortRevision"], prepared["validationCohortRevision"]
            )

            # verify must not rewrite the frozen metadata.
            self.assertEqual(
                json.loads((root / "debug" / "split_info_stab.json").read_text()), split
            )
            self.assertEqual(
                json.loads(
                    (root / "debug" / "cohorts" / "imported_dlib_stab.json").read_text()
                ),
                cohort,
            )
            self.assertTrue(cohort["validationCohortRevision"])
            self.assertTrue(cohort["testCohortRevision"])
            self.assertNotEqual(
                cohort["validationCohortRevision"], cohort["testCohortRevision"]
            )

    def test_verify_rejects_a_tampered_mapping_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"tamper-{index}.png", 30 + index) for index in range(4)]
            self._xml(root / "xml" / "train_tamper.xml", images, slots=(3, 12))
            self._prepare(root, "tamper")

            mapping_path = root / "debug" / "id_mapping_tamper.json"
            mapping = json.loads(mapping_path.read_text())
            mapping["landmark_template"]["3"]["x_mean"] += 25.0
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

            with self.assertRaises(ImportedDlibContractError) as raised:
                prepare_or_verify(
                    str(root),
                    "tamper",
                    mode="verify",
                    validation_mode="derive",
                    test_mode="derive",
                    confirm_template_order=False,
                    validation_fraction=0.2,
                    seed=42,
                )
            self.assertIn(
                raised.exception.code, {"stale_import_mapping", "frozen_cohort_changed"}
            )

    def test_verify_requires_the_frozen_validation_xml(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_session(root)
            images = [self._image(root, f"noval-{index}.png", 30 + index) for index in range(4)]
            self._xml(root / "xml" / "train_noval.xml", images, slots=(3, 12))
            self._prepare(root, "noval")

            (root / "xml" / "validation_noval.xml").unlink()
            with self.assertRaises(ImportedDlibContractError) as raised:
                prepare_or_verify(
                    str(root),
                    "noval",
                    mode="verify",
                    validation_mode="derive",
                    test_mode="derive",
                    confirm_template_order=False,
                    validation_fraction=0.2,
                    seed=42,
                )
            self.assertEqual(raised.exception.code, "missing_validation_xml")


class ImportedDlibSidecarBundlingTests(ImportedDlibFixtureMixin, unittest.TestCase):
    """The importer's sidecar must satisfy the immutable-artifact validator.

    `train_shape_model` and `train_cnn_model` both call `bundle_id_mapping`
    unconditionally before fitting, so an importer sidecar that fails this
    contract makes imported training impossible.
    """

    def _prepared_mapping_path(self, root: Path, tag: str, *, policy=None, template=None):
        self._write_session(root, policy=policy, template=template)
        images = [self._image(root, f"{tag}-{index}.png", 30 + index) for index in range(4)]
        slots = tuple(entry["index"] for entry in (template or DEFAULT_TEMPLATE))
        self._xml(root / "xml" / f"train_{tag}.xml", images, slots=slots)
        self._prepare(root, tag)
        return root / "debug" / f"id_mapping_{tag}.json"

    def test_importer_sidecar_bundles_into_an_immutable_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping_path = self._prepared_mapping_path(root, "bundle")
            artifact_dir = root / "artifacts" / "dlib" / "run-1"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            artifact_path, descriptor, bundled = bundle_id_mapping(
                str(mapping_path), str(artifact_dir)
            )

            self.assertTrue(Path(artifact_path).is_file())
            self.assertEqual(descriptor["sha256"], lineage.sha256_file(artifact_path))
            # Output landmark IDs must be exactly the session schema IDs.
            self.assertEqual(
                sorted(bundled["dlib_index_to_original"].values()),
                sorted(entry["index"] for entry in DEFAULT_TEMPLATE),
            )
            # The validator requires contiguous slots from zero; the importer
            # renumbers non-contiguous source slots to satisfy this.
            self.assertEqual(
                sorted(int(key) for key in bundled["dlib_index_to_original"]),
                list(range(len(DEFAULT_TEMPLATE))),
            )

    def test_noncontiguous_source_slots_still_bundle(self):
        """Source slots 3/12 renumber to 0/1 so the sidecar validator accepts them."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping_path = self._prepared_mapping_path(root, "sparse")
            artifact_dir = root / "artifacts" / "dlib" / "run-1"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            _, _, bundled = bundle_id_mapping(str(mapping_path), str(artifact_dir))
            self.assertEqual(bundled["dlib_index_to_original"], {"0": 3, "1": 12})
            self.assertEqual(
                bundled["training_config"]["orientation_mode"],
                bundled["training_config"]["orientation_policy"]["mode"],
            )

    def test_every_orientation_mode_produces_a_bundleable_sidecar(self):
        cases = {
            "directional": DEFAULT_POLICY,
            "bilateral": dict(DEFAULT_POLICY, mode="bilateral", bilateralPairs=[[3, 12]]),
            "axial": dict(DEFAULT_POLICY, mode="axial"),
            "invariant": dict(DEFAULT_POLICY, mode="invariant"),
        }
        for mode, policy in cases.items():
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    mapping_path = self._prepared_mapping_path(root, mode, policy=policy)
                    artifact_dir = root / "artifacts" / "dlib" / "run-1"
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    _, _, bundled = bundle_id_mapping(
                        str(mapping_path), str(artifact_dir)
                    )
                    self.assertEqual(bundled["training_config"]["orientation_mode"], mode)

    def test_a_tampered_sidecar_is_refused_at_bundle_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping_path = self._prepared_mapping_path(root, "tampered")
            artifact_dir = root / "artifacts" / "dlib" / "run-1"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            mapping = json.loads(mapping_path.read_text())
            mapping["training_config"]["orientation_mode"] = "invariant"
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

            with self.assertRaises(ValueError):
                bundle_id_mapping(str(mapping_path), str(artifact_dir))


if __name__ == "__main__":
    unittest.main()
