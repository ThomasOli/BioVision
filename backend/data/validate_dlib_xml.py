#!/usr/bin/env python3
"""
Validate dlib XML annotations and optionally write a normalized copy.

Normalization rewrites image file paths to absolute paths so training is robust
after importing XML from external folders.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET


def validate_and_normalize(input_xml: str, output_xml: str | None = None):
    result = {
        "ok": False,
        "input_xml": input_xml,
        "output_xml": output_xml,
        "num_images": 0,
        "num_boxes": 0,
        "num_parts": 0,
        "warnings": [],
        "errors": [],
    }

    if not os.path.exists(input_xml):
        result["errors"].append(f"XML file not found: {input_xml}")
        return result

    try:
        tree = ET.parse(input_xml)
        root = tree.getroot()
    except Exception as e:
        result["errors"].append(f"Invalid XML: {e}")
        return result

    images_node = root.find("images")
    if images_node is None:
        result["errors"].append("Missing <images> node in XML.")
        return result

    xml_dir = os.path.dirname(os.path.abspath(input_xml))

    for image_node in images_node.findall("image"):
        result["num_images"] += 1
        file_attr = image_node.get("file")
        if not file_attr:
            result["errors"].append("An <image> entry is missing the 'file' attribute.")
            continue

        resolved_path = (
            file_attr
            if os.path.isabs(file_attr)
            else os.path.abspath(os.path.join(xml_dir, file_attr))
        )

        if not os.path.exists(resolved_path):
            result["errors"].append(f"Image path does not exist: {resolved_path}")
        else:
            # Normalize to absolute path to avoid cwd-relative resolution issues.
            image_node.set("file", resolved_path)

        boxes = image_node.findall("box")
        if not boxes:
            result["warnings"].append(
                f"Image has no boxes: {resolved_path if file_attr else file_attr}"
            )

        for box in boxes:
            result["num_boxes"] += 1
            for attr in ("left", "top", "width", "height"):
                raw = box.get(attr)
                if raw is None:
                    result["errors"].append(f"A <box> is missing required '{attr}' attribute.")
                    continue
                try:
                    value = int(raw)
                    if attr in ("width", "height") and value <= 0:
                        result["errors"].append(
                            f"Box has non-positive {attr}: {value} (must be > 0)."
                        )
                except Exception:
                    result["errors"].append(f"Box attribute '{attr}' is not an integer: {raw}")

            parts = box.findall("part")
            if not parts:
                result["warnings"].append("A <box> has no <part> landmarks.")

            for part in parts:
                result["num_parts"] += 1
                name = part.get("name")
                x = part.get("x")
                y = part.get("y")
                if name is None or x is None or y is None:
                    result["errors"].append(
                        "A <part> is missing one of required attributes: name, x, y."
                    )
                    continue
                try:
                    int(name)
                except Exception:
                    result["errors"].append(f"Part 'name' must be an integer id, got: {name}")
                try:
                    int(x)
                    int(y)
                except Exception:
                    result["errors"].append(f"Part coordinates must be integers, got x={x}, y={y}")

    if result["num_images"] == 0:
        result["errors"].append("No <image> entries found in XML.")
    if result["num_boxes"] == 0:
        result["errors"].append("No <box> entries found in XML.")
    if result["num_parts"] == 0:
        result["errors"].append("No <part> entries found in XML.")

    result["ok"] = len(result["errors"]) == 0

    if result["ok"] and output_xml:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_xml)), exist_ok=True)
            tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        except Exception as e:
            result["ok"] = False
            result["errors"].append(f"Failed writing normalized XML: {e}")

    return result


def main():
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    "ok": False,
                    "errors": [
                        "Usage: validate_dlib_xml.py <input_xml> [output_xml]"
                    ],
                }
            )
        )
        return

    input_xml = sys.argv[1]
    output_xml = sys.argv[2] if len(sys.argv) > 2 else None
    result = validate_and_normalize(input_xml, output_xml)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
