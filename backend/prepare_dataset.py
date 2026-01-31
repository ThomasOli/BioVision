import json
import os
import glob
import sys
import xml.etree.ElementTree as ET
import cv2

#Generate XML Tree to be used in dlib model generation
'''
XML Tree format

<labels>
    <lmages>
        <image file="images/fossil1.png">
        <box top="..." left="..." width="..." height="...">
            <part name="0" x="123" y="456" />
            <part name="1" x="234" y="321" />
        </box>
        </image>
    </images>
</labels>
'''
def json_to_dlib_xml(project_root, tag, use_full_image=False):
  """
  Convert JSON landmark annotations to dlib XML format.

  Args:
      project_root: Path to project directory
      tag: Model tag name
      use_full_image: If True, use full image dimensions as bounding box.
                      If False (default), use boxes from JSON (or tight box around landmarks for backward compat).
                      Set to True for simpler inference workflow.
  """
  labels_dir = os.path.join(project_root, "labels")
  xmldir = os.path.join(project_root, "xml")
  images_dir = os.path.join(project_root, "images")

  os.makedirs(xmldir, exist_ok=True)
  out_xml_path = os.path.join(xmldir, f"train_{tag}.xml")

  dataset = ET.Element("dataset")
  images_el = ET.SubElement(dataset, "images")

  json_paths = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
  if not json_paths:
    raise RuntimeError(f"No JSON files found in {labels_dir}")

  for jp in json_paths:
    with open(jp, "r") as f:
      data = json.load(f)

    image_filename = data.get("imageFilename")

    image_path = os.path.join(images_dir, image_filename) if image_filename else None

    if not os.path.exists(image_path):
      raise FileNotFoundError(f"Image {image_path} not found for {jp}")

    image_el = ET.SubElement(images_el, "image", file=image_path)

    # Check if new format with explicit boxes exists
    if "boxes" in data and data["boxes"]:
      # New format: boxes with nested landmarks
      for box_data in data["boxes"]:
        landmarks = box_data.get("landmarks", [])

        if use_full_image:
          # Use full image dimensions as bounding box
          img = cv2.imread(image_path)
          if img is None:
            raise RuntimeError(f"Could not read image {image_path}")
          h, w = img.shape[:2]
          left = 0
          top = 0
          width = w
          height = h
        else:
          # Use explicit box coordinates from JSON
          left = int(box_data.get("left", 0))
          top = int(box_data.get("top", 0))
          width = int(box_data.get("width", 0))
          height = int(box_data.get("height", 0))

        box_el = ET.SubElement(image_el, "box",
                               top=str(top), left=str(left),
                               width=str(width), height=str(height))

        for idx, lm in enumerate(sorted(landmarks, key=lambda lm: lm.get("id", 0))):
          ET.SubElement(box_el, "part",
                        name=str(idx),
                        x=str(int(lm["x"])),
                        y=str(int(lm["y"])))
    else:
      # Backward compatibility: old format with flat landmarks array
      landmarks = data.get("landmarks", [])

      if not landmarks:
        continue  # Skip images with no landmarks

      if use_full_image:
        # Use full image dimensions as bounding box
        img = cv2.imread(image_path)
        if img is None:
          raise RuntimeError(f"Could not read image {image_path}")
        h, w = img.shape[:2]
        left = 0
        top = 0
        width = w
        height = h
      else:
        # Use tight bounding box around landmarks with margin
        xs = [lm["x"] for lm in landmarks]
        ys = [lm["y"] for lm in landmarks]
        margin = 10
        left = max(int(min(xs)) - margin, 0)
        top = max(int(min(ys)) - margin, 0)
        right = int(max(xs)) + margin
        bottom = int(max(ys)) + margin
        width = right - left
        height = bottom - top

      box_el = ET.SubElement(image_el, "box",
                             top=str(top), left=str(left),
                             width=str(width), height=str(height))

      for idx, lm in enumerate(sorted(landmarks, key=lambda lm: lm.get("id", 0))):
        ET.SubElement(box_el, "part",
                      name=str(idx),
                      x=str(int(lm["x"])),
                      y=str(int(lm["y"])))

  tree = ET.ElementTree(dataset)
  tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)
  print(out_xml_path)


if __name__ == "__main__":
  # args: project_root tag [--full-image]
  project_root = sys.argv[1]
  tag = sys.argv[2]
  use_full_image = "--full-image" in sys.argv
  json_to_dlib_xml(project_root, tag, use_full_image=use_full_image)
