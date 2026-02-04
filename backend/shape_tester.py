# backend/shape_tester.py
"""
Detailed model evaluation for dlib shape predictors.
Provides per-landmark error metrics and visualization data.
"""
import os
import sys
import json
import math
import dlib
import xml.etree.ElementTree as ET
from image_utils import load_image


def parse_xml_annotations(xml_path):
    """
    Parse a dlib XML file and extract image paths, boxes, and landmarks.

    Returns:
        List of dicts with 'image_path', 'box', and 'landmarks' keys
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images = root.find("images")
    if images is None:
        return []

    annotations = []
    for image in images.findall("image"):
        image_path = image.get("file")
        for box in image.findall("box"):
            left = int(box.get("left"))
            top = int(box.get("top"))
            width = int(box.get("width"))
            height = int(box.get("height"))

            landmarks = []
            for part in box.findall("part"):
                name = int(part.get("name"))
                x = int(part.get("x"))
                y = int(part.get("y"))
                landmarks.append({"id": name, "x": x, "y": y})

            # Sort by landmark id
            landmarks.sort(key=lambda lm: lm["id"])

            annotations.append({
                "image_path": image_path,
                "box": {
                    "left": left,
                    "top": top,
                    "right": left + width,
                    "bottom": top + height,
                    "width": width,
                    "height": height
                },
                "landmarks": landmarks
            })

    return annotations


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)


def test_model(project_root, tag, xml_path=None):
    """
    Test a trained model and compute detailed error metrics.

    Args:
        project_root: Root directory of the project
        tag: Model tag/name
        xml_path: Optional path to XML file (defaults to test_{tag}.xml)

    Returns:
        dict with detailed test results
    """
    modeldir = os.path.join(project_root, "models")
    xmldir = os.path.join(project_root, "xml")
    debug_dir = os.path.join(project_root, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Model not found at {predictor_path}")

    # Default to test XML if not specified
    if xml_path is None:
        xml_path = os.path.join(xmldir, f"test_{tag}.xml")

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found at {xml_path}")

    # Load the model
    predictor = dlib.shape_predictor(predictor_path)

    # Parse annotations
    annotations = parse_xml_annotations(xml_path)
    if not annotations:
        raise RuntimeError(f"No annotations found in {xml_path}")

    # Get dlib's built-in error metric
    dlib_error = dlib.test_shape_predictor(xml_path, predictor_path)

    # Compute detailed per-landmark errors
    all_errors = []  # List of errors for each image
    per_landmark_errors = {}  # Aggregate errors by landmark ID

    for ann in annotations:
        image_path = ann["image_path"]
        box = ann["box"]
        ground_truth = ann["landmarks"]

        # Load image
        img, w, h = load_image(image_path)
        if img is None:
            print(f"Warning: Could not load {image_path}", file=sys.stderr)
            continue

        # Create dlib rectangle
        rect = dlib.rectangle(
            left=box["left"],
            top=box["top"],
            right=box["right"],
            bottom=box["bottom"]
        )

        # Predict
        shape = predictor(img, rect)

        # Compute errors
        image_errors = {
            "image_path": image_path,
            "box": box,
            "landmarks": []
        }

        box_diagonal = math.sqrt(box["width"]**2 + box["height"]**2)

        for i, gt in enumerate(ground_truth):
            if i >= shape.num_parts:
                break

            pred = shape.part(i)
            pred_point = {"x": pred.x, "y": pred.y}
            gt_point = {"x": gt["x"], "y": gt["y"]}

            # Pixel error
            pixel_error = euclidean_distance(pred_point, gt_point)

            # Normalized error (relative to bounding box diagonal)
            normalized_error = pixel_error / box_diagonal if box_diagonal > 0 else 0

            landmark_result = {
                "id": gt["id"],
                "ground_truth": gt_point,
                "predicted": pred_point,
                "pixel_error": pixel_error,
                "normalized_error": normalized_error
            }

            image_errors["landmarks"].append(landmark_result)

            # Aggregate by landmark ID
            lid = gt["id"]
            if lid not in per_landmark_errors:
                per_landmark_errors[lid] = []
            per_landmark_errors[lid].append(pixel_error)

        all_errors.append(image_errors)

    # Compute summary statistics
    all_pixel_errors = []
    for img_err in all_errors:
        for lm in img_err["landmarks"]:
            all_pixel_errors.append(lm["pixel_error"])

    summary = {
        "dlib_error": dlib_error,
        "num_images": len(annotations),
        "num_landmarks": len(ground_truth) if annotations else 0,
        "total_predictions": len(all_pixel_errors)
    }

    if all_pixel_errors:
        summary["mean_pixel_error"] = sum(all_pixel_errors) / len(all_pixel_errors)
        summary["max_pixel_error"] = max(all_pixel_errors)
        summary["min_pixel_error"] = min(all_pixel_errors)

        # Median
        sorted_errors = sorted(all_pixel_errors)
        mid = len(sorted_errors) // 2
        if len(sorted_errors) % 2 == 0:
            summary["median_pixel_error"] = (sorted_errors[mid-1] + sorted_errors[mid]) / 2
        else:
            summary["median_pixel_error"] = sorted_errors[mid]

        # Per-landmark statistics
        landmark_stats = {}
        for lid, errors in per_landmark_errors.items():
            landmark_stats[lid] = {
                "mean": sum(errors) / len(errors),
                "max": max(errors),
                "min": min(errors),
                "count": len(errors)
            }
        summary["per_landmark_stats"] = landmark_stats

    # Full results
    results = {
        "summary": summary,
        "per_image_errors": all_errors
    }

    # Save results
    results_path = os.path.join(debug_dir, f"test_results_{tag}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Test results saved to: {results_path}", file=sys.stderr)

    # Print summary
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION: {tag}")
    print(f"{'='*50}")
    print(f"Test images: {summary['num_images']}")
    print(f"Landmarks per image: {summary['num_landmarks']}")
    print(f"dlib error (normalized): {summary['dlib_error']:.4f}")

    if "mean_pixel_error" in summary:
        print(f"\nPixel Error Statistics:")
        print(f"  Mean:   {summary['mean_pixel_error']:.2f} px")
        print(f"  Median: {summary['median_pixel_error']:.2f} px")
        print(f"  Min:    {summary['min_pixel_error']:.2f} px")
        print(f"  Max:    {summary['max_pixel_error']:.2f} px")

        if "per_landmark_stats" in summary:
            print(f"\nPer-Landmark Mean Errors:")
            for lid in sorted(summary["per_landmark_stats"].keys()):
                stats = summary["per_landmark_stats"][lid]
                print(f"  Landmark {lid}: {stats['mean']:.2f} px (n={stats['count']})")

    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python shape_tester.py <project_root> <tag> [xml_path]")
        print("  xml_path: Optional path to XML file (defaults to test_{tag}.xml)")
        sys.exit(1)

    project_root = sys.argv[1]
    tag = sys.argv[2]
    xml_path = sys.argv[3] if len(sys.argv) > 3 else None

    results = test_model(project_root, tag, xml_path)
    print(json.dumps(results["summary"], indent=2))
