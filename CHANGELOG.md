# Changelog

## [0.1.2] - 2026-08-26

### Training and inference reliability

- Enforce immutable OBB and landmark artifact provenance across training, promotion, and inference.
- Freeze validation cohorts and add deterministic mirrored validation for directional schemas.
- Validate exact landmark ID mappings and normalized crop geometry for CNN and dlib predictors.
- Pin orientation-safe OBB augmentations and prevent landmark-derived direction fallbacks.
- Preserve OBB direction and mirrored landmark coordinates through crop normalization and inverse mapping.
- Gate landmark predictor training until a verified OBB detector is active.

### HITL workflow

- Clarify review, correction, approval, and training-data commit states in the inference UI.
- Add transactional HITL commits with rollback-safe recovery and deterministic source staging.
- Verify all orientation modes through mocked live retraining and inference flows.

### Packaging

- Pin NumPy and OpenCV to preserve the PyTorch NumPy bridge on Intel macOS.
- Package the standard Linux AppImage with CPU-only PyTorch to keep the release
  asset below GitHub's 2 GiB limit. Linux CUDA acceleration remains available
  for source installations through `setup.sh`/`setup_backend.py`; Windows and
  macOS package acceleration behavior is unchanged.

## [0.1.0] - 2026-04-13

### Initial Release

#### Features
- Image annotation with landmark point placement on canvas
- dlib shape predictor model training pipeline
- Inference / prediction on new images using trained models
- Oriented Bounding Box (OBB) support for schema orientation
- Human-in-the-loop (HITL) inference session workflow
- Professional guided tutorial and onboarding system
- Contextual help panel
- Multi-platform installers: Windows (NSIS), macOS (DMG), Linux (AppImage)
- Configurable project root directory
- Undo/redo support for landmark edits
- Magnified zoom view for precision annotation
