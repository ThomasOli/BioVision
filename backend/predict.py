import sys
import os
import json
import dlib
import cv2

def predict_image(project_root, tag, image_path):
    modeldir = os.path.join(project_root, "models")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")



if __name__ == "__main__":
    project_root = sys.argv[1]
    tag = sys.argv[2]
    image_path = sys.argv[3]

    result = predict_image(project_root, tag, image_path)
    print(json.dumps(result))