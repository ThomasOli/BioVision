import sys
import os
import json
import dlib
import cv2

def predict_image(project_root, tag, image_path):
    modeldir = os.path.join(project_root, "models")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")

    if not os.path.exists(predictor_path):
        raise FileNotFoundError("Cannot find model at {predictor_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Could not read image {image_path}")
    
    h,w = img.shape[:2]
    rect = dlib.rectangle(left=0, top=0, right=w-1, bottom=h-1)
    
    predictor = dlib.shape_predictor(predictor_path)
    shape = predictor(img, rect)

    landmarks = []
    for i in range(shape.num_parts):
        part = shape.part(i)
        landmarks.append({"id":i, "x" : int(part.x), "y" : int(part.y)})
    
    return {"image": image_path, "landmarks": landmarks}
if __name__ == "__main__":
    project_root = sys.argv[1]
    tag = sys.argv[2]
    image_path = sys.argv[3]

    result = predict_image(project_root, tag, image_path)
    print(json.dumps(result))