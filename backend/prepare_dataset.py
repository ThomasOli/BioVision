import json
import os
import glob
import sys
import xml.etree.ElementTree as ET

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
def json_to_dlib_xml(project_root, tag):
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
    landmarks = data["landmarks"]

    # image_path = os.path.join(images_dir, image_filename)
    if not os.path.exists(image_path):
      raise FileNotFoundError(f"Image {image_path} not found for {jp}")

    xs = [lm["x"] for lm in landmarks]
    ys = [lm["y"] for lm in landmarks]
    margin = 10
    left = max(min(xs) - margin, 0)
    top = max(min(ys) - margin, 0)
    right = max(xs) + margin
    bottom = max(ys) + margin
    width = right - left
    height = bottom - top

    image_el = ET.SubElement(images_el, "image", file=image_path)
    box_el = ET.SubElement(image_el, "box",
                           top=str(top), left=str(left),
                           width=str(width), height=str(height))

    for lm in sorted(landmarks, key=lambda lm: lm["id"]):
      ET.SubElement(box_el, "part",
                    name=str(lm["id"]),
                    x=str(int(lm["x"])),
                    y=str(int(lm["y"])))

  tree = ET.ElementTree(dataset)
  tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)
  print(out_xml_path)


if __name__ == "__main__":
  # args: project_root tag
  project_root = sys.argv[1]
  tag = sys.argv[2]
  json_to_dlib_xml(project_root, tag)
