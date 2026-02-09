# backend/train_shape_model.py
"""Train dlib shape predictor with optimized parameters."""
import os
import sys
import json
import dlib


def get_training_options(num_images, num_landmarks):
    """Get optimized training options based on dataset size."""
    options = dlib.shape_predictor_training_options()

    is_tiny_dataset = num_images < 30
    is_small_dataset = num_images < 100

    # Tree depth
    if is_tiny_dataset:
        options.tree_depth = 3
    elif is_small_dataset:
        options.tree_depth = 3
    else:
        options.tree_depth = 4

    # Cascade depth (refinement stages)
    if is_tiny_dataset:
        options.cascade_depth = 12
    elif is_small_dataset:
        options.cascade_depth = 10
    else:
        options.cascade_depth = 12

    # Nu (regularization)
    if is_tiny_dataset:
        options.nu = 0.3
    elif is_small_dataset:
        options.nu = 0.2
    else:
        options.nu = 0.1

    # Feature pool size
    options.feature_pool_size = 400

    # Trees per cascade level
    if is_tiny_dataset:
        options.num_trees_per_cascade_level = 200
    elif is_small_dataset:
        options.num_trees_per_cascade_level = 150
    else:
        options.num_trees_per_cascade_level = 300

    # Test splits
    options.num_test_splits = 10 if is_tiny_dataset else 15

    # Oversampling (augmentation)
    if is_tiny_dataset:
        options.oversampling_amount = 300
    elif is_small_dataset:
        options.oversampling_amount = 150
    else:
        options.oversampling_amount = 40

    # Translation jitter
    if is_tiny_dataset:
        options.oversampling_translation_jitter = 0.15
    elif is_small_dataset:
        options.oversampling_translation_jitter = 0.1
    else:
        options.oversampling_translation_jitter = 0.05

    options.feature_pool_region_padding = 0.1
    options.lambda_param = 0.2 if is_tiny_dataset else 0.1
    options.random_seed = "42"
    options.be_verbose = True

    import multiprocessing
    options.num_threads = multiprocessing.cpu_count()

    return options


def count_landmarks_in_xml(xml_path):
    """Count images and landmarks in a dlib XML file."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images = root.find("images")
    if images is None:
        return 0, 0

    num_images = 0
    num_landmarks = 0
    for image in images.findall("image"):
        num_images += 1
        for box in image.findall("box"):
            parts = box.findall("part")
            if parts:
                num_landmarks = max(num_landmarks, len(parts))

    return num_images, num_landmarks


def train_shape_model(project_root, tag, custom_options=None):
    """Train a dlib shape predictor model."""
    project_root = os.path.abspath(project_root)
    xmldir = os.path.join(project_root, "xml")
    modeldir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    os.makedirs(modeldir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    train_xml = os.path.join(xmldir, f"train_{tag}.xml")
    test_xml = os.path.join(xmldir, f"test_{tag}.xml")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")

    if not os.path.exists(train_xml):
        raise FileNotFoundError(f"Train XML not found at {train_xml}")

    # Count images and landmarks to set optimal parameters
    num_images, num_landmarks = count_landmarks_in_xml(train_xml)
    print(f"Training set: {num_images} images, {num_landmarks} landmarks per image", file=sys.stderr)

    # Get optimized options
    options = get_training_options(num_images, num_landmarks)

    # Apply custom options if provided
    if custom_options:
        for key, value in custom_options.items():
            if hasattr(options, key):
                setattr(options, key, value)
                print(f"Custom option: {key} = {value}", file=sys.stderr)

    # Log training parameters
    params_log = {
        "num_images": num_images,
        "num_landmarks": num_landmarks,
        "tree_depth": options.tree_depth,
        "cascade_depth": options.cascade_depth,
        "nu": options.nu,
        "feature_pool_size": options.feature_pool_size,
        "num_trees_per_cascade_level": options.num_trees_per_cascade_level,
        "num_test_splits": options.num_test_splits,
        "oversampling_amount": options.oversampling_amount,
        "oversampling_translation_jitter": options.oversampling_translation_jitter,
        "feature_pool_region_padding": options.feature_pool_region_padding,
        "lambda_param": options.lambda_param
    }
    params_path = os.path.join(debug_dir, f"training_params_{tag}.json")
    with open(params_path, "w") as f:
        json.dump(params_log, f, indent=2)
    print(f"Training parameters saved to: {params_path}", file=sys.stderr)

    # Train the model
    print("Training shape predictor...", file=sys.stderr)
    dlib.train_shape_predictor(train_xml, predictor_path, options)
    print("MODEL_PATH", predictor_path)

    # Evaluate on training set
    train_error = dlib.test_shape_predictor(train_xml, predictor_path)
    print("TRAIN_ERROR", train_error)

    # Evaluate on test set if available
    test_error = None
    if os.path.exists(test_xml):
        test_error = dlib.test_shape_predictor(test_xml, predictor_path)
        print("TEST_ERROR", test_error)

    # Save results
    results = {
        "model_path": predictor_path,
        "train_error": train_error,
        "test_error": test_error,
        "num_images": num_images,
        "num_landmarks": num_landmarks
    }
    results_path = os.path.join(debug_dir, f"training_results_{tag}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}", file=sys.stderr)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_shape_model.py <project_root> <tag> [options_json]")
        print("  options_json: Optional JSON string with custom training options")
        sys.exit(1)

    project_root = sys.argv[1]
    tag = sys.argv[2]

    custom_options = None
    if len(sys.argv) > 3:
        custom_options = json.loads(sys.argv[3])

    train_shape_model(project_root, tag, custom_options)
