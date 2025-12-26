# backend/train_shape_model.py
import os
import sys
import dlib


def train_shape_model(project_root, tag):
  project_root = os.path.abspath(project_root)
  xmldir = os.path.join(project_root, "xml")
  modeldir = os.path.join(project_root, "models")
  os.makedirs(modeldir, exist_ok=True)

  train_xml = os.path.join(xmldir, f"train_{tag}.xml")
  predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")

  if not os.path.exists(train_xml):
    raise FileNotFoundError(f"Train XML not found at {train_xml}")

  options = dlib.shape_predictor_training_options()
  options.tree_depth = 4
  options.nu = 0.1
  options.cascade_depth = 10
  options.feature_pool_size = 400
  options.num_test_splits = 50
  options.oversampling_amount = 20
  options.be_verbose = True
  options.num_threads = 4

  print("Training shape predictor…")
  dlib.train_shape_predictor(train_xml, predictor_path, options)
  print("MODEL_PATH", predictor_path)

  error = dlib.test_shape_predictor(train_xml, predictor_path)
  print("TRAIN_ERROR", error)


if __name__ == "__main__":
  project_root = sys.argv[1]
  tag = sys.argv[2]
  train_shape_model(project_root, tag)
