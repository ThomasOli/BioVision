"""
Train a CNN landmark predictor (configurable backbone + regression head).

Uses the same dlib XML files produced by prepare_dataset.py so no additional
data preparation is needed — just run this after prepare_dataset.py.

Output:
  models/cnn_{tag}.pth           — model weights
  models/cnn_{tag}_config.json   — {n_landmarks, landmark_ids, trained_at}

Stdout protocol (same as train_shape_model.py so main.ts can parse identically):
  MODEL_PATH <path>
  TRAIN_ERROR <value>
  TEST_ERROR <value>
"""
import os
import sys
import json
import math
import argparse
import time
from contextlib import nullcontext
import xml.etree.ElementTree as ET
from datetime import datetime

import cv2
import numpy as np

import sys as _sys, os as _os
_BACKEND_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _BACKEND_ROOT not in _sys.path:
    _sys.path.insert(0, _BACKEND_ROOT)

import bv_utils.debug_io as dio
import bv_utils.orientation_utils as ou

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models as tv_models
    from torchvision import transforms as tv_transforms
except ImportError as e:
    print(f"ERROR: torch/torchvision not installed: {e}", file=sys.stderr)
    sys.exit(1)

STANDARD_SIZE = 512


def _resolve_orientation_aug_policy(project_root):
    orientation_policy = ou.load_orientation_policy(project_root)
    orientation_mode = ou.get_orientation_mode(orientation_policy)
    aug_profile = ou.get_schema_augmentation_profile(orientation_mode, engine="cnn")
    return orientation_policy, orientation_mode, aug_profile


def _is_canonical_training_enabled(orientation_policy, orientation_mode):
    mode = str(orientation_mode or "").strip().lower()
    pca_mode = str((orientation_policy or {}).get("pcaLevelingMode", "off")).strip().lower()
    if pca_mode not in ("off", "on", "auto"):
        pca_mode = "off"
    return mode != "invariant" and pca_mode in ("on", "auto")


def _tune_cnn_directional_aug_profile(
    base_profile,
    *,
    orientation_mode,
    size_bucket,
    canonical_training_enabled,
):
    """
    Strengthen directional CNN robustness while preserving orientation semantics.

    - Canonicalized directional data (SAM2/PCA flow): moderate geometry.
    - Non-canonical directional data: stronger geometry.
    - Small datasets: slightly stronger augmentation for regularization.
    """
    profile = dict(base_profile or {})
    mode = str(orientation_mode or "").strip().lower()
    if mode != "directional":
        return profile

    if canonical_training_enabled:
        profile.update(
            {
                "flip_prob": 0.45,
                "vertical_flip_prob": 0.0,
                "rotation_range": (-16.0, 16.0),
                "rotate_180_prob": 0.0,
                "scale_range": (0.94, 1.06),
                "translate_ratio": 0.05,
            }
        )
    else:
        profile.update(
            {
                "flip_prob": 0.5,
                "vertical_flip_prob": 0.0,
                "rotation_range": (-30.0, 30.0),
                "rotate_180_prob": 0.0,
                "scale_range": (0.88, 1.12),
                "translate_ratio": 0.09,
            }
        )

    # Dataset-size adaptation: tiny/small datasets need extra regularization.
    if size_bucket == "tiny":
        lo, hi = profile["rotation_range"]
        profile["rotation_range"] = (float(lo) - 4.0, float(hi) + 4.0)
        s_lo, s_hi = profile["scale_range"]
        profile["scale_range"] = (max(0.82, float(s_lo) - 0.02), min(1.20, float(s_hi) + 0.02))
        profile["translate_ratio"] = min(0.11, float(profile["translate_ratio"]) + 0.015)
    elif size_bucket == "small":
        lo, hi = profile["rotation_range"]
        profile["rotation_range"] = (float(lo) - 2.0, float(hi) + 2.0)
        profile["translate_ratio"] = min(0.10, float(profile["translate_ratio"]) + 0.01)

    return profile


def _dataset_size_bucket(num_samples):
    """
    Resolve training-size bucket from effective training sample count.
    """
    n = int(max(0, num_samples))
    if n < 120:
        return "tiny"
    if n < 300:
        return "small"
    if n < 700:
        return "medium"
    if n < 1500:
        return "large"
    return "xlarge"


def _resolve_cnn_training_profile(num_samples, orientation_mode):
    """
    Return adaptive CNN optimization defaults by dataset size.
    """
    bucket = _dataset_size_bucket(num_samples)
    profiles = {
        # Small datasets need longer training + stronger regularization.
        "tiny": {
            "epochs": 140,
            "lr": 3e-4,
            "batch_size": 8,
            "weight_decay": 4e-4,
            "dropout": 0.35,
        },
        "small": {
            "epochs": 110,
            "lr": 2e-4,
            "batch_size": 10,
            "weight_decay": 3e-4,
            "dropout": 0.3,
        },
        # Mid-size dataset baseline.
        "medium": {
            "epochs": 90,
            "lr": 1.5e-4,
            "batch_size": 12,
            "weight_decay": 2e-4,
            "dropout": 0.25,
        },
        "large": {
            "epochs": 70,
            "lr": 1e-4,
            "batch_size": 16,
            "weight_decay": 1e-4,
            "dropout": 0.2,
        },
        "xlarge": {
            "epochs": 55,
            "lr": 8e-5,
            "batch_size": 24,
            "weight_decay": 8e-5,
            "dropout": 0.15,
        },
    }
    profile = dict(profiles[bucket])

    # Harder orientation regimes often need more optimization steps.
    mode = str(orientation_mode or "").strip().lower()
    if mode == "invariant":
        profile["epochs"] = int(round(profile["epochs"] * 1.15))
    elif mode == "axial":
        profile["epochs"] = int(round(profile["epochs"] * 1.08))

    return profile, bucket


def _load_id_mapping(project_root, tag):
    path = os.path.join(project_root, "debug", f"id_mapping_{tag}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _build_cnn_bilateral_index_pairs(project_root, tag, landmark_keys, orientation_policy):
    if ou.get_orientation_mode(orientation_policy) != "bilateral":
        return []
    bilateral_pairs = ou.get_bilateral_pairs(orientation_policy)
    if not bilateral_pairs:
        return []

    id_mapping = _load_id_mapping(project_root, tag)
    pair_swap = ou.build_pair_swap_map(bilateral_pairs)
    name_swap = ou.build_dlib_name_swap_map(id_mapping, pair_swap)
    if not name_swap:
        return []

    name_to_index = {str(name): i for i, name in enumerate(landmark_keys)}
    out = []
    seen = set()
    for left_name, right_name in name_swap.items():
        if left_name not in name_to_index or right_name not in name_to_index:
            continue
        a = int(name_to_index[left_name])
        b = int(name_to_index[right_name])
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


# ── Dataset ───────────────────────────────────────────────────────────────────

def _parse_dlib_xml(xml_path):
    """
    Parse a dlib training XML and return list of (image_path, landmarks_dict).
    landmarks_dict: {part_name_str: (x, y)}  — only valid (non-negative) parts.
    """
    if not os.path.exists(xml_path):
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images_el = root.find("images")
    if images_el is None:
        return []

    records = []
    for img_el in images_el.findall("image"):
        img_file = img_el.get("file", "")
        if not img_file or not os.path.exists(img_file):
            continue
        box_el = img_el.find("box")
        if box_el is None:
            continue
        parts = {}
        for p in box_el.findall("part"):
            name = p.get("name")
            x = int(p.get("x", -1))
            y = int(p.get("y", -1))
            if name is not None and x >= 0 and y >= 0:
                parts[name] = (x, y)
        if parts:
            records.append((img_file, parts))
    return records


def _collect_landmark_ids(records):
    """
    Return a sorted list of part-name keys that appear in ALL records.
    This mirrors dlib's behaviour: only universally-present parts are trained.
    """
    if not records:
        return []
    common = set(records[0][1].keys())
    for _, parts in records[1:]:
        common &= set(parts.keys())
    return sorted(common, key=lambda n: (int(n) if n.isdigit() else float("inf"), n))


class LandmarkDataset(Dataset):
    """
    Dataset backed by dlib XML (same crops as dlib training).
    Images are already 512×512 corrected crops.
    Landmarks are normalised to [0, 1] within the 512×512 space.
    """

    def __init__(
        self,
        records,
        landmark_keys,
        transform=None,
        augment=False,
        seed=42,
        flip_prob=0.5,
        vertical_flip_prob=0.0,
        rotation_range=(-35.0, 35.0),
        rotate_180_prob=0.0,
        scale_range=(0.88, 1.12),
        translate_ratio=0.08,
        bilateral_index_pairs=None,
    ):
        self.records = records
        self.landmark_keys = landmark_keys  # ordered list of part-name strings
        self.transform = transform
        self.augment = bool(augment)
        self._rng = np.random.default_rng(seed)
        self.flip_prob = float(max(0.0, min(1.0, flip_prob)))
        self.vertical_flip_prob = float(max(0.0, min(1.0, vertical_flip_prob)))
        self.rotation_range = (float(rotation_range[0]), float(rotation_range[1]))
        self.rotate_180_prob = float(max(0.0, min(1.0, rotate_180_prob)))
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.translate_ratio = float(max(0.0, translate_ratio))
        self.bilateral_index_pairs = [
            (int(a), int(b))
            for a, b in (bilateral_index_pairs or [])
            if int(a) != int(b)
        ]

    def _apply_geometric_augment(self, img_rgb, coords_px):
        """
        Apply synchronized geometric augmentation to image and landmark coordinates.

        Augmentations:
        - random horizontal mirror
        - random rotation/scale/translation (diagonal + pose variation)
        """
        h, w = img_rgb.shape[:2]
        coords = coords_px.astype(np.float32).copy()
        img = img_rgb.copy()

        if self._rng.random() < self.flip_prob:
            img = cv2.flip(img, 1)
            coords[:, 0] = (w - 1) - coords[:, 0]
            # Bilateral schemas need semantic channel swaps after mirror.
            for a, b in self.bilateral_index_pairs:
                if 0 <= a < coords.shape[0] and 0 <= b < coords.shape[0]:
                    tmp = coords[a].copy()
                    coords[a] = coords[b]
                    coords[b] = tmp

        if self._rng.random() < self.vertical_flip_prob:
            img = cv2.flip(img, 0)
            coords[:, 1] = (h - 1) - coords[:, 1]

        angle = float(self._rng.uniform(self.rotation_range[0], self.rotation_range[1]))
        if self.rotate_180_prob > 0.0 and self._rng.random() < self.rotate_180_prob:
            angle += 180.0
        scale = float(self._rng.uniform(self.scale_range[0], self.scale_range[1]))
        tx = float(self._rng.uniform(-self.translate_ratio * w, self.translate_ratio * w))
        ty = float(self._rng.uniform(-self.translate_ratio * h, self.translate_ratio * h))

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        img = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        ones = np.ones((coords.shape[0], 1), dtype=np.float32)
        hom = np.concatenate([coords, ones], axis=1)  # [N,3]
        coords_aug = hom @ M.T  # [N,2]
        coords_aug[:, 0] = np.clip(coords_aug[:, 0], 0.0, float(w - 1))
        coords_aug[:, 1] = np.clip(coords_aug[:, 1], 0.0, float(h - 1))
        return img, coords_aug

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, parts = self.records[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Return zeros on read failure (rare, but avoids crash)
            img = np.zeros((STANDARD_SIZE, STANDARD_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        coords = []
        for key in self.landmark_keys:
            x, y = parts.get(key, (STANDARD_SIZE // 2, STANDARD_SIZE // 2))
            coords.append((float(x), float(y)))

        coords_np = np.asarray(coords, dtype=np.float32)
        if self.augment:
            img, coords_np = self._apply_geometric_augment(img, coords_np)

        if self.transform:
            img = self.transform(img)

        denom = float(max(1, STANDARD_SIZE - 1))
        flat = []
        for i in range(coords_np.shape[0]):
            flat.append(float(coords_np[i, 0]) / denom)
            flat.append(float(coords_np[i, 1]) / denom)
        target = torch.tensor(flat, dtype=torch.float32)
        return img, target


# ── Model ─────────────────────────────────────────────────────────────────────

def _resolve_cnn_variant(requested_variant):
    variant = str(requested_variant or "simplebaseline").strip().lower()
    aliases = {
        "simplebase": "resnet50",
        "simplebaseline": "resnet50",
        "efficientnet": "efficientnet_b0",
        "efficientnet-b0": "efficientnet_b0",
        "mobilenet": "mobilenet_v3_large",
        "mobilenetv3": "mobilenet_v3_large",
        "mobilenet-v3-large": "mobilenet_v3_large",
        "resnet": "resnet50",
        "resnet-50": "resnet50",
        "hrnet": "hrnet_w32",
        "hrnet-w32": "hrnet_w32",
    }
    return aliases.get(variant, variant)


def _build_cnn_backbone(variant, use_pretrained=True):
    """
    Return (features_module, feature_dim, resolved_variant, fallback_reason)
    """
    v = _resolve_cnn_variant(variant)
    fallback_reason = None

    if v == "efficientnet_b0":
        weights = "IMAGENET1K_V1" if use_pretrained else None
        backbone = tv_models.efficientnet_b0(weights=weights)
        return backbone.features, 1280, v, fallback_reason

    if v == "mobilenet_v3_large":
        weights = "IMAGENET1K_V1" if use_pretrained else None
        backbone = tv_models.mobilenet_v3_large(weights=weights)
        return backbone.features, 960, v, fallback_reason

    if v == "resnet50":
        weights = "IMAGENET1K_V2" if use_pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        return features, 2048, v, fallback_reason

    if v == "hrnet_w32":
        ctor = getattr(tv_models, "hrnet_w32", None)
        if ctor is not None:
            try:
                weights = "IMAGENET1K_V1" if use_pretrained else None
                backbone = ctor(weights=weights)
                if hasattr(backbone, "features"):
                    return backbone.features, 2048, v, fallback_reason
                fallback_reason = "torchvision_hrnet_features_unavailable_fallback_resnet50"
            except Exception as exc:
                fallback_reason = f"hrnet_unavailable_fallback_resnet50:{exc}"
        else:
            fallback_reason = "torchvision_hrnet_missing_fallback_resnet50"
        # Safe fallback
        features, feat_dim, _, _ = _build_cnn_backbone("resnet50", use_pretrained=use_pretrained)
        return features, feat_dim, "resnet50", fallback_reason

    # Unknown variant -> safe default.
    fallback_reason = f"unknown_variant_{v}_fallback_efficientnet_b0"
    features, feat_dim, _, _ = _build_cnn_backbone("efficientnet_b0", use_pretrained=use_pretrained)
    return features, feat_dim, "efficientnet_b0", fallback_reason


def _spatial_soft_argmax_2d(heatmaps, beta=25.0):
    """
    Convert heatmaps [B,K,H,W] into normalized coordinates [B,K*2] in [0,1].
    """
    b, k, h, w = heatmaps.shape
    logits = heatmaps.view(b, k, -1) * float(beta)
    probs = torch.softmax(logits, dim=-1)

    xs = torch.linspace(0.0, 1.0, steps=w, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0.0, 1.0, steps=h, device=heatmaps.device, dtype=heatmaps.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx.reshape(1, 1, -1)
    gy = gy.reshape(1, 1, -1)

    exp_x = torch.sum(probs * gx, dim=-1)
    exp_y = torch.sum(probs * gy, dim=-1)
    coords = torch.stack([exp_x, exp_y], dim=-1)  # [B,K,2]
    return coords.reshape(b, k * 2)


class CNNLandmarkPredictor(nn.Module):
    """Backbone + deconvolution heatmap head + soft-argmax coordinate decoder."""

    def __init__(
        self,
        n_landmarks,
        model_variant="simplebaseline",
        dropout_rate=0.3,
        head_type="heatmap_deconv",
        deconv_layers=3,
        deconv_filters=256,
        softargmax_beta=25.0,
    ):
        super().__init__()
        features, feat_dim, resolved_variant, fallback_reason = _build_cnn_backbone(
            model_variant,
            use_pretrained=True,
        )
        dropout_rate = float(max(0.0, min(0.8, dropout_rate)))
        self.n_landmarks = int(n_landmarks)
        self.features = features
        self.head_type = str(head_type or "heatmap_deconv").strip().lower()
        if self.head_type not in ("heatmap_deconv", "regression"):
            self.head_type = "heatmap_deconv"
        self.softargmax_beta = float(softargmax_beta)
        self.deconv_layers = int(max(1, deconv_layers))
        self.deconv_filters = int(max(32, deconv_filters))

        if self.head_type == "regression":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, self.n_landmarks * 2),
                nn.Sigmoid(),
            )
            self.deconv = None
            self.heatmap_head = None
        else:
            layers = []
            in_ch = feat_dim
            for _ in range(self.deconv_layers):
                layers.extend(
                    [
                        nn.ConvTranspose2d(
                            in_ch,
                            self.deconv_filters,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.deconv_filters),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_ch = self.deconv_filters
            self.deconv = nn.Sequential(*layers)
            self.heatmap_head = nn.Conv2d(in_ch, self.n_landmarks, kernel_size=1, stride=1, padding=0)
            self.pool = None
            self.head = None

        self.model_variant = resolved_variant
        self.variant_fallback_reason = fallback_reason
        self.dropout_rate = dropout_rate

    def forward(self, x, return_heatmaps=False):
        x = self.features(x)
        if self.head_type == "regression":
            x = self.pool(x)
            return self.head(x)

        up = self.deconv(x)
        heatmaps = self.heatmap_head(up)
        coords = _spatial_soft_argmax_2d(heatmaps, beta=self.softargmax_beta)
        if return_heatmaps:
            return coords, heatmaps
        return coords


# ── Loss ──────────────────────────────────────────────────────────────────────

def wing_loss(pred, target, w=10.0, eps=2.0):
    """
    Wing loss — better than MSE for small-error landmark regression.
    Ref: Feng et al. "Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks" (CVPR 2018).
    """
    diff = torch.abs(pred - target)
    C = w - w * math.log(1 + w / eps)
    loss = torch.where(diff < w, w * torch.log(1 + diff / eps), diff - C)
    return loss.mean()


# ── Training ──────────────────────────────────────────────────────────────────

def _compute_error(model, loader, device):
    """
    Mean normalised per-landmark distance (in [0,1] coord space).
    Comparable to dlib's normalised pixel error.
    """
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        use_amp_eval = str(device) == "cuda"
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=use_amp_eval)
            targets = targets.to(device, non_blocking=use_amp_eval)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if use_amp_eval
                else nullcontext()
            )
            with amp_ctx:
                preds = model(imgs)
            diff = (preds - targets).view(preds.size(0), -1, 2)
            dists = torch.norm(diff, dim=2)  # shape [batch, n_landmarks]
            total += dists.mean(dim=1).sum().item()
            count += preds.size(0)
    return total / count if count > 0 else float("nan")


def _run_pipeline_parity_eval(project_root, tag, predictor_type, train_xml, test_xml, run_dir=None):
    """
    Evaluate end-to-end inference parity on train/test XML crops.
    """
    try:
        from training.pipeline_parity_eval import evaluate_pipeline_parity

        return evaluate_pipeline_parity(
            project_root=project_root,
            tag=tag,
            predictor_type=predictor_type,
            train_xml=train_xml,
            test_xml=test_xml if os.path.exists(test_xml) else None,
            debug_output_dir=run_dir,
            outlier_px_threshold=400.0,
        )
    except Exception as exc:
        return {"error": str(exc)}


def train_cnn_model(project_root, tag, epochs=None, lr=None, batch_size=None,
                    model_variant="simplebaseline", skip_parity=False):
    """
    Train a CNN landmark predictor for a given session + model tag.

    Args:
        project_root: Session root (contains xml/, models/, debug/).
        tag: Model tag — must match existing train_{tag}.xml from prepare_dataset.py.
        epochs: Training epochs (None -> adaptive by dataset size).
        lr: AdamW learning rate (None -> adaptive by dataset size).
        batch_size: Mini-batch size (None -> adaptive by dataset size).
    """
    project_root = os.path.abspath(project_root)
    orientation_policy, orientation_mode, aug_profile = _resolve_orientation_aug_policy(project_root)
    xmldir = os.path.join(project_root, "xml")
    modeldir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    os.makedirs(modeldir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    run_dir, run_id = dio.create_model_run_dir(project_root, "cnn", tag)
    dio.write_run_manifest(
        run_dir,
        model_type="cnn",
        tag=tag,
        project_root=project_root,
        extra={"status": "started"},
    )

    train_xml = os.path.join(xmldir, f"train_{tag}.xml")
    test_xml = os.path.join(xmldir, f"test_{tag}.xml")

    if not os.path.exists(train_xml):
        raise FileNotFoundError(f"train_{tag}.xml not found at {train_xml}. "
                                "Run prepare_dataset.py first.")

    # Parse records
    train_records = _parse_dlib_xml(train_xml)
    test_records = _parse_dlib_xml(test_xml) if os.path.exists(test_xml) else []

    if not train_records:
        raise ValueError(f"No valid training samples found in {train_xml}")

    requested_epochs = None
    try:
        if epochs is not None:
            requested_epochs = int(epochs)
    except Exception:
        requested_epochs = None
    if requested_epochs is not None and requested_epochs <= 0:
        requested_epochs = None

    requested_lr = None
    try:
        if lr is not None:
            requested_lr = float(lr)
    except Exception:
        requested_lr = None
    if requested_lr is not None and requested_lr <= 0:
        requested_lr = None

    requested_batch_size = None
    try:
        if batch_size is not None:
            requested_batch_size = int(batch_size)
    except Exception:
        requested_batch_size = None
    if requested_batch_size is not None and requested_batch_size <= 0:
        requested_batch_size = None

    adaptive_profile, size_bucket = _resolve_cnn_training_profile(
        len(train_records),
        orientation_mode,
    )
    canonical_training_enabled = _is_canonical_training_enabled(
        orientation_policy,
        orientation_mode,
    )
    aug_profile = _tune_cnn_directional_aug_profile(
        aug_profile,
        orientation_mode=orientation_mode,
        size_bucket=size_bucket,
        canonical_training_enabled=canonical_training_enabled,
    )
    resolved_epochs = requested_epochs if requested_epochs is not None else int(adaptive_profile["epochs"])
    resolved_lr = requested_lr if requested_lr is not None else float(adaptive_profile["lr"])
    resolved_batch_size = (
        requested_batch_size if requested_batch_size is not None else int(adaptive_profile["batch_size"])
    )
    resolved_weight_decay = float(adaptive_profile["weight_decay"])
    resolved_dropout = float(adaptive_profile["dropout"])

    # Determine landmark keys (part names that exist in all training records)
    landmark_keys = _collect_landmark_ids(train_records)
    n_landmarks = len(landmark_keys)
    if n_landmarks == 0:
        raise ValueError("No common landmarks found across all training images.")
    bilateral_index_pairs = _build_cnn_bilateral_index_pairs(
        project_root,
        tag,
        landmark_keys,
        orientation_policy,
    )

    print(f"Training CNN: {len(train_records)} train samples, "
          f"{len(test_records)} test samples, {n_landmarks} landmarks",
          file=sys.stderr)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}", file=sys.stderr)

    # Keep defaults robust across CPU/MPS systems.
    if str(device) == "cpu":
        resolved_batch_size = min(resolved_batch_size, 8)
    elif str(device) == "mps":
        resolved_batch_size = min(resolved_batch_size, 12)
    resolved_batch_size = max(1, int(resolved_batch_size))

    print(
        f"Resolved CNN profile: bucket={size_bucket}, epochs={resolved_epochs}, "
        f"lr={resolved_lr:g}, batch={resolved_batch_size}, wd={resolved_weight_decay:g}, "
        f"dropout={resolved_dropout:.2f}",
        file=sys.stderr,
    )

    # Throughput-oriented loader settings to keep GPU fed.
    cpu_count = max(1, int(os.cpu_count() or 1))
    if str(device) == "cuda":
        loader_workers = min(8, max(2, cpu_count // 2))
        pin_memory = True
    elif str(device) == "mps":
        loader_workers = min(4, max(1, cpu_count // 4))
        pin_memory = False
    else:
        loader_workers = 0
        pin_memory = False
    loader_kwargs = {
        "num_workers": int(loader_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(loader_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    use_amp = str(device) == "cuda"
    if use_amp:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    train_params_log = {
        "model_type": "cnn",
        "tag": tag,
        "run_id": run_id,
        "train_xml": train_xml,
        "test_xml": test_xml if os.path.exists(test_xml) else None,
        "dataset_size_bucket": size_bucket,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "n_landmarks": n_landmarks,
        "landmark_keys": landmark_keys,
        "epochs_requested": requested_epochs,
        "learning_rate_requested": requested_lr,
        "batch_size_requested": requested_batch_size,
        "epochs": resolved_epochs,
        "learning_rate": resolved_lr,
        "batch_size": resolved_batch_size,
        "weight_decay": resolved_weight_decay,
        "dropout": resolved_dropout,
        "device": str(device),
        "dataloader_workers": int(loader_workers),
        "pin_memory": bool(pin_memory),
        "amp_enabled": bool(use_amp),
        "standard_size": STANDARD_SIZE,
        "coord_denominator": STANDARD_SIZE - 1,
        "model_variant_requested": model_variant,
        "orientation_mode": orientation_mode,
        "orientation_policy": orientation_policy,
        "canonical_training_enabled": bool(canonical_training_enabled),
        "bilateral_index_pairs": bilateral_index_pairs,
        "geometric_augmentation": {
            "enabled": True,
            "profile": aug_profile,
        },
        "skip_parity": bool(skip_parity),
    }

    # Unified default: deconv heatmap head with soft-argmax decoding.
    cnn_head_type = "heatmap_deconv"
    cnn_deconv_layers = 3
    cnn_deconv_filters = 256
    cnn_softargmax_beta = 25.0
    train_params_log["cnn_head_type"] = cnn_head_type
    train_params_log["cnn_deconv_layers"] = cnn_deconv_layers
    train_params_log["cnn_deconv_filters"] = cnn_deconv_filters
    train_params_log["cnn_softargmax_beta"] = cnn_softargmax_beta
    dio.write_run_json(run_dir, "train_params.json", train_params_log)
    dio.write_json(os.path.join(debug_dir, f"training_params_{tag}_cnn.json"), train_params_log)

    # Transforms
    train_transform = tv_transforms.Compose([
        tv_transforms.ToPILImage(),
        tv_transforms.Resize((STANDARD_SIZE, STANDARD_SIZE)),
        tv_transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        tv_transforms.RandomApply(
            [tv_transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
            p=0.3,
        ),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])
    val_transform = tv_transforms.Compose([
        tv_transforms.ToPILImage(),
        tv_transforms.Resize((STANDARD_SIZE, STANDARD_SIZE)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

    train_ds = LandmarkDataset(
        train_records,
        landmark_keys,
        transform=train_transform,
        augment=True,
        seed=42,
        flip_prob=float(aug_profile.get("flip_prob", 0.5)),
        vertical_flip_prob=float(aug_profile.get("vertical_flip_prob", 0.0)),
        rotation_range=tuple(aug_profile.get("rotation_range", (-35.0, 35.0))),
        rotate_180_prob=float(aug_profile.get("rotate_180_prob", 0.0)),
        scale_range=tuple(aug_profile.get("scale_range", (0.88, 1.12))),
        translate_ratio=float(aug_profile.get("translate_ratio", 0.08)),
        bilateral_index_pairs=bilateral_index_pairs,
    )
    train_loader = DataLoader(train_ds, batch_size=resolved_batch_size,
                              shuffle=True, drop_last=False, **loader_kwargs)

    test_loader = None
    if test_records:
        test_ds = LandmarkDataset(
            test_records,
            landmark_keys,
            transform=val_transform,
            augment=False,
            seed=42,
        )
        test_loader = DataLoader(test_ds, batch_size=resolved_batch_size,
                                 shuffle=False, **loader_kwargs)

    # Train-set loader without augmentation for final error computation
    train_val_ds = LandmarkDataset(
        train_records,
        landmark_keys,
        transform=val_transform,
        augment=False,
        seed=42,
    )
    train_val_loader = DataLoader(train_val_ds, batch_size=resolved_batch_size,
                                  shuffle=False, **loader_kwargs)

    model = CNNLandmarkPredictor(
        n_landmarks,
        model_variant=model_variant,
        dropout_rate=resolved_dropout,
        head_type=cnn_head_type,
        deconv_layers=cnn_deconv_layers,
        deconv_filters=cnn_deconv_filters,
        softargmax_beta=cnn_softargmax_beta,
    ).to(device)
    resolved_variant = getattr(model, "model_variant", model_variant)
    variant_fallback_reason = getattr(model, "variant_fallback_reason", None)
    train_params_log["model_variant_resolved"] = resolved_variant
    if variant_fallback_reason:
        train_params_log["model_variant_fallback_reason"] = variant_fallback_reason
    dio.write_run_json(run_dir, "train_params.json", train_params_log)
    dio.write_json(os.path.join(debug_dir, f"training_params_{tag}_cnn.json"), train_params_log)
    optimizer = optim.AdamW(model.parameters(), lr=resolved_lr, weight_decay=resolved_weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=resolved_epochs)
    epoch_losses = []
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    train_started_at = time.time()

    print(
        "PROGRESS_JSON " + json.dumps({
            "percent": 8,
            "stage": "training",
            "substage": "init",
            "message": (
                f"Initialized CNN ({resolved_variant}) on {device} | "
                f"batch={resolved_batch_size}, workers={loader_workers}, amp={use_amp}"
            ),
            "device": str(device),
            "amp_enabled": bool(use_amp),
            "batch_size": int(resolved_batch_size),
            "workers": int(loader_workers),
            "epochs": int(resolved_epochs),
            "train_samples": int(len(train_records)),
            "test_samples": int(len(test_records)),
        }),
        file=sys.stderr,
    )

    for epoch in range(1, resolved_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_started_at = time.time()
        lr_now = float(scheduler.get_last_lr()[0]) if scheduler is not None else float(resolved_lr)
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=bool(pin_memory))
            targets = targets.to(device, non_blocking=bool(pin_memory))
            optimizer.zero_grad()
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                preds = model(imgs)
                loss = wing_loss(preds, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        scheduler.step()

        avg_loss = epoch_loss / max(len(train_records), 1)
        epoch_losses.append({"epoch": epoch, "loss": avg_loss})
        pct = int(10 + 85 * (epoch / resolved_epochs))
        epoch_sec = max(1e-6, time.time() - epoch_started_at)
        elapsed_sec = max(0.0, time.time() - train_started_at)
        avg_epoch_sec = elapsed_sec / max(1, epoch)
        eta_sec = max(0.0, (resolved_epochs - epoch) * avg_epoch_sec)
        samples_per_sec = float(len(train_records)) / epoch_sec
        message = (
            f"Epoch {epoch}/{resolved_epochs} | loss={avg_loss:.4f} | "
            f"lr={lr_now:.2e} | {samples_per_sec:.1f} img/s | eta={int(round(eta_sec))}s"
        )
        print(f"PROGRESS {pct} {message}", file=sys.stderr)
        print(
            "PROGRESS_JSON " + json.dumps(
                {
                    "percent": int(pct),
                    "stage": "training",
                    "substage": "epoch",
                    "message": message,
                    "epoch": int(epoch),
                    "epochs": int(resolved_epochs),
                    "loss": float(avg_loss),
                    "lr": float(lr_now),
                    "epoch_sec": float(epoch_sec),
                    "elapsed_sec": float(elapsed_sec),
                    "eta_sec": float(eta_sec),
                    "samples_per_sec": float(samples_per_sec),
                    "batch_size": int(resolved_batch_size),
                    "amp_enabled": bool(use_amp),
                    "device": str(device),
                }
            ),
            file=sys.stderr,
        )

    # Save weights
    model_path = os.path.join(modeldir, f"cnn_{tag}.pth")
    torch.save(model.state_dict(), model_path)

    # ── Resolve original schema landmark IDs ─────────────────────────────────
    # The dlib XML uses 0-indexed part names ("0", "1", ...) produced by
    # prepare_dataset.py.  We need to remap these back to the user's original
    # schema IDs (e.g. 1..11) via id_mapping_{tag}.json so that CNN inference
    # emits landmarks with the same IDs as the dlib predictor.
    id_mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")

    # Default: convert part-name strings to ints (may be 0-indexed dlib parts)
    landmark_ids_raw = []
    for k in landmark_keys:
        try:
            landmark_ids_raw.append(int(k))
        except ValueError:
            landmark_ids_raw.append(k)

    landmark_ids = landmark_ids_raw  # will be overwritten if mapping exists
    if os.path.exists(id_mapping_path):
        try:
            with open(id_mapping_path, "r", encoding="utf-8") as f:
                id_map = json.load(f)
            explicit = id_map.get("dlib_index_to_original", {})
            if explicit:
                # landmark_keys are the sorted XML part names ("0", "1", ..., "10").
                # Their sort order matches the dlib part index (0, 1, ..., n-1).
                sorted_keys = sorted(
                    landmark_keys,
                    key=lambda n: (int(n) if n.isdigit() else float("inf"), n),
                )
                remapped = []
                for i, _ in enumerate(sorted_keys):
                    mapped_id = explicit.get(str(i))
                    if mapped_id is not None:
                        remapped.append(int(mapped_id))
                    else:
                        remapped.append(landmark_ids_raw[i])
                landmark_ids = remapped
                print(
                    f"CNN config: remapped {len(landmark_ids)} landmark IDs "
                    f"from dlib indices to schema IDs via id_mapping_{tag}.json",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"Warning: could not load id_mapping_{tag}.json: {exc}", file=sys.stderr)

    config = {
        "n_landmarks": n_landmarks,
        "landmark_keys": landmark_keys,
        "landmark_ids": landmark_ids,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "epochs": resolved_epochs,
        "lr": resolved_lr,
        "batch_size": resolved_batch_size,
        "weight_decay": resolved_weight_decay,
        "dropout": resolved_dropout,
        "dataset_size_bucket": size_bucket,
        "coord_denominator": STANDARD_SIZE - 1,
        "model_variant_requested": model_variant,
        "model_variant_resolved": resolved_variant,
        "model_variant_fallback_reason": variant_fallback_reason,
        "cnn_head_type": cnn_head_type,
        "cnn_deconv_layers": cnn_deconv_layers,
        "cnn_deconv_filters": cnn_deconv_filters,
        "cnn_softargmax_beta": cnn_softargmax_beta,
    }
    config_path = os.path.join(modeldir, f"cnn_{tag}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    dio.write_run_json(run_dir, "model_config.json", config)

    print("PROGRESS 97 evaluating_train_test_error", file=sys.stderr)
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 97,
                "stage": "evaluation",
                "substage": "train_test_error",
                "message": "Computing train/test landmark error.",
            }
        ),
        file=sys.stderr,
    )

    # Compute train / test error (normalised mean landmark distance)
    train_error = _compute_error(model, train_val_loader, device)
    test_error = _compute_error(model, test_loader, device) if test_loader else None
    pipeline_parity = {"skipped": True, "reason": "skip_parity"} if skip_parity else None
    if skip_parity:
        print("PROGRESS 98 skipping_pipeline_parity", file=sys.stderr)
        print(
            "PROGRESS_JSON " + json.dumps(
                {
                    "percent": 98,
                    "stage": "evaluation",
                    "substage": "pipeline_parity_skipped",
                    "message": "Skipping pipeline parity evaluation by user option.",
                }
            ),
            file=sys.stderr,
        )
    else:
        print("PROGRESS 98 evaluating_pipeline_parity", file=sys.stderr)
        print(
            "PROGRESS_JSON " + json.dumps(
                {
                    "percent": 98,
                    "stage": "evaluation",
                    "substage": "pipeline_parity",
                    "message": "Running pipeline parity evaluation (GT + detected boxes).",
                }
            ),
            file=sys.stderr,
        )
        pipeline_parity = _run_pipeline_parity_eval(
            project_root,
            tag,
            "cnn",
            train_xml,
            test_xml,
            run_dir=run_dir,
        )
        if isinstance(pipeline_parity, dict) and not pipeline_parity.get("error"):
            try:
                train_gt = (
                    pipeline_parity.get("splits", {})
                    .get("train", {})
                    .get("gt_boxes", {})
                    .get("pixel_error_mean")
                )
                test_gt = (
                    pipeline_parity.get("splits", {})
                    .get("test", {})
                    .get("gt_boxes", {})
                    .get("pixel_error_mean")
                )
                print(
                    f"Pipeline parity (CNN, GT boxes): train_mean_px={train_gt}, test_mean_px={test_gt}",
                    file=sys.stderr,
                )
                orientation_test = (
                    pipeline_parity.get("orientation_signal_summary", {})
                    .get("test", {})
                    .get("detected_boxes", {})
                )
                if isinstance(orientation_test, dict):
                    print(
                        "Orientation signal (cnn, test/detected): "
                        f"hint_present={orientation_test.get('detector_hint_present', 0)} "
                        f"hint_missing={orientation_test.get('detector_hint_missing', 0)} "
                        f"warnings={orientation_test.get('warning_code_counts', {})}",
                        file=sys.stderr,
                    )
            except Exception:
                pass
    # Stdout protocol matching train_shape_model.py
    print("MODEL_PATH", model_path)
    print("TRAIN_ERROR", round(train_error, 6))
    if test_error is not None:
        print("TEST_ERROR", round(test_error, 6))

    results = {
        "model_path": model_path,
        "config_path": config_path,
        "train_error": train_error,
        "test_error": test_error,
        "n_landmarks": n_landmarks,
        "pipeline_parity": pipeline_parity,
    }
    dio.write_json(
        os.path.join(debug_dir, f"training_results_{tag}_cnn.json"),
        {
            **results,
            "tag": tag,
            "run_id": run_id,
        },
    )
    dio.write_run_json(
        run_dir,
        "train_results.json",
        {
            **results,
            "model_type": "cnn",
            "tag": tag,
            "run_id": run_id,
            "landmark_ids": landmark_ids,
        },
    )
    dio.write_run_json(run_dir, "loss_curve.json", {"epochs": epoch_losses})
    for name in [
        f"id_mapping_{tag}.json",
        f"crop_metadata_{tag}.json",
        f"orientation_{tag}.json",
        f"training_boxes_{tag}.json",
        f"split_info_{tag}.json",
    ]:
        dio.copy_json_if_exists(os.path.join(debug_dir, name), run_dir, name)
    dio.write_run_manifest(
        run_dir,
        model_type="cnn",
        tag=tag,
        project_root=project_root,
        extra={
            "status": "completed",
            "model_path": model_path,
            "train_error": train_error,
            "test_error": test_error,
        },
    )
    print(f"CNN run debug saved to: {run_dir}", file=sys.stderr)
    return results


if __name__ == "__main__":
    # Backward compatible positional mode:
    #   train_cnn_model.py <project_root> <tag> [epochs] [lr] [model_variant]
    if len(sys.argv) >= 3 and not any(arg.startswith("--") for arg in sys.argv[3:]):
        project_root = sys.argv[1]
        tag = sys.argv[2]
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else None
        lr = float(sys.argv[4]) if len(sys.argv) > 4 else None
        model_variant = sys.argv[5] if len(sys.argv) > 5 else "simplebaseline"
        train_cnn_model(project_root, tag, epochs=epochs, lr=lr, model_variant=model_variant)
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Train CNN landmark model with adaptive dataset-size hyperparameters."
    )
    parser.add_argument("project_root")
    parser.add_argument("tag")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-variant", type=str, default="simplebaseline")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    train_cnn_model(
        args.project_root,
        args.tag,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        model_variant=args.model_variant,
        skip_parity=bool(args.skip_parity),
    )
