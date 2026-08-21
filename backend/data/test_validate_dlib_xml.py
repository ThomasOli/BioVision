"""
Tests for backend/data/validate_dlib_xml.py.

Only touches stdlib + the function under test — no ML deps, no fixtures
from disk beyond the per-test tmp_path.
"""

import os
import xml.etree.ElementTree as ET

import pytest

from backend.data.validate_dlib_xml import validate_and_normalize


def _write_image(path: str) -> None:
    """dlib validator only checks file existence, not image contents."""
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")  # PNG magic; body irrelevant here


def _make_xml(
    tmp_path,
    *,
    image_files: list[str] | None = None,
    box_overrides: dict | None = None,
    parts: list[tuple[str, str, str]] | None = None,
    omit_images_node: bool = False,
    raw_xml: str | None = None,
) -> str:
    """Build a minimal but well-formed dlib XML and write it under tmp_path."""
    xml_path = os.path.join(str(tmp_path), "labels.xml")
    if raw_xml is not None:
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(raw_xml)
        return xml_path

    root = ET.Element("dataset")
    if not omit_images_node:
        images = ET.SubElement(root, "images")
        for img in image_files or []:
            image_node = ET.SubElement(images, "image", {"file": img})
            box_attrs = {"left": "10", "top": "20", "width": "100", "height": "80"}
            if box_overrides:
                box_attrs.update({k: str(v) for k, v in box_overrides.items()})
            box_node = ET.SubElement(image_node, "box", box_attrs)
            for name, x, y in parts or [("1", "50", "60")]:
                ET.SubElement(box_node, "part", {"name": name, "x": x, "y": y})

    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


class TestHappyPath:
    def test_valid_xml_passes_and_counts_are_correct(self, tmp_path):
        img1 = os.path.join(str(tmp_path), "a.png")
        img2 = os.path.join(str(tmp_path), "b.png")
        _write_image(img1)
        _write_image(img2)
        xml_path = _make_xml(
            tmp_path,
            image_files=[img1, img2],
            parts=[("1", "10", "20"), ("2", "30", "40")],
        )

        result = validate_and_normalize(xml_path)

        assert result["ok"] is True
        assert result["errors"] == []
        assert result["num_images"] == 2
        assert result["num_boxes"] == 2
        assert result["num_parts"] == 4

    def test_relative_image_paths_are_normalized_to_absolute_in_output(self, tmp_path):
        img_abs = os.path.join(str(tmp_path), "img.png")
        _write_image(img_abs)
        # Reference it by basename (relative to xml_dir).
        xml_path = _make_xml(tmp_path, image_files=["img.png"])
        out_path = os.path.join(str(tmp_path), "normalized.xml")

        result = validate_and_normalize(xml_path, output_xml=out_path)

        assert result["ok"] is True
        assert os.path.isfile(out_path)
        tree = ET.parse(out_path)
        file_attrs = [img.get("file") for img in tree.getroot().iter("image")]
        assert file_attrs == [os.path.abspath(img_abs)]

    def test_output_xml_not_written_when_input_is_invalid(self, tmp_path):
        xml_path = _make_xml(tmp_path, image_files=[])  # no images -> errors
        out_path = os.path.join(str(tmp_path), "normalized.xml")

        result = validate_and_normalize(xml_path, output_xml=out_path)

        assert result["ok"] is False
        assert not os.path.exists(out_path)


class TestErrorPaths:
    def test_missing_input_file(self, tmp_path):
        result = validate_and_normalize(os.path.join(str(tmp_path), "nope.xml"))
        assert result["ok"] is False
        assert any("not found" in e for e in result["errors"])

    def test_malformed_xml(self, tmp_path):
        xml_path = _make_xml(tmp_path, raw_xml="<dataset><images>")
        result = validate_and_normalize(xml_path)
        assert result["ok"] is False
        assert any("Invalid XML" in e for e in result["errors"])

    def test_missing_images_node(self, tmp_path):
        xml_path = _make_xml(tmp_path, omit_images_node=True)
        result = validate_and_normalize(xml_path)
        assert result["ok"] is False
        assert any("Missing <images>" in e for e in result["errors"])

    def test_referenced_image_does_not_exist(self, tmp_path):
        xml_path = _make_xml(tmp_path, image_files=["missing.png"])
        result = validate_and_normalize(xml_path)
        assert result["ok"] is False
        assert any("Image path does not exist" in e for e in result["errors"])

    def test_non_positive_box_dimension(self, tmp_path):
        img = os.path.join(str(tmp_path), "a.png")
        _write_image(img)
        xml_path = _make_xml(
            tmp_path,
            image_files=[img],
            box_overrides={"width": 0},
        )
        result = validate_and_normalize(xml_path)
        assert result["ok"] is False
        assert any("non-positive width" in e for e in result["errors"])

    def test_non_integer_box_attribute(self, tmp_path):
        img = os.path.join(str(tmp_path), "a.png")
        _write_image(img)
        xml_path = _make_xml(
            tmp_path,
            image_files=[img],
            box_overrides={"left": "abc"},
        )
        result = validate_and_normalize(xml_path)
        assert result["ok"] is False
        assert any("not an integer" in e for e in result["errors"])

    def test_part_with_non_integer_name(self, tmp_path):
        img = os.path.join(str(tmp_path), "a.png")
        _write_image(img)
        xml_path = _make_xml(
            tmp_path,
            image_files=[img],
            parts=[("left_eye", "10", "20")],
        )
        result = validate_and_normalize(xml_path)
        assert result["ok"] is False
        assert any("Part 'name' must be an integer" in e for e in result["errors"])

    @pytest.mark.parametrize("coord", [("bad", "20"), ("10", "bad")])
    def test_part_with_non_integer_coordinates(self, tmp_path, coord):
        img = os.path.join(str(tmp_path), "a.png")
        _write_image(img)
        x, y = coord
        xml_path = _make_xml(
            tmp_path,
            image_files=[img],
            parts=[("1", x, y)],
        )
        result = validate_and_normalize(xml_path)
        assert result["ok"] is False
        assert any("coordinates must be integers" in e for e in result["errors"])


class TestEmptyInputs:
    def test_no_images_fails(self, tmp_path):
        xml_path = _make_xml(tmp_path, image_files=[])
        result = validate_and_normalize(xml_path)
        assert result["ok"] is False
        # Empty dataset triggers all three "no X" errors.
        assert any("No <image>" in e for e in result["errors"])
        assert any("No <box>" in e for e in result["errors"])
        assert any("No <part>" in e for e in result["errors"])
