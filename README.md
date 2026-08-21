# BioVision

BioVision is an Electron desktop application for annotating biological images,
training landmark and oriented-bounding-box (OBB) models, reviewing predictions,
and promoting measured model versions into production.

> **Status:** Active development. Keep an export or backup of important projects
> before upgrading between development builds.

## Architecture

BioVision has three cooperating layers:

- `src/` contains the React/TypeScript annotation, training, inference, model,
  and human-in-the-loop (HITL) interfaces.
- `electron/` owns IPC, filesystem transactions, model activation, and Python
  process orchestration. `electron/preload.ts` is the renderer boundary.
- `backend/` contains dataset preparation, orientation transforms, dlib/CNN
  landmark training and inference, YOLO OBB export/training, and lineage logic.

Training data and artifacts live below the selected project root. Every new
training run receives an immutable model ID and artifact directory. A mutable
display name is only a label; inference resolves the active immutable artifact.

## Training contracts

### Orientation is explicit

Each schema must save a confirmed orientation policy. Landmark IDs never imply
an anatomical left/right policy. The supported modes are:

- `directional`: a head/tail axis is normalized to the configured facing direction.
- `bilateral`: paired left/right anatomy is normalized along its symmetry axis.
- `axial`: elongated specimens are normalized along a dominant axis, but the
  two poles remain interchangeable and the OBB detector uses one class.
- `invariant`: no anatomical facing direction is enforced.

Dataset preparation and OBB training fail early when the policy is missing or
unconfirmed. Every required landmark must also be present; incomplete samples
are rejected with the source and missing landmark IDs instead of being silently
trained as partial geometry.

### Frozen evaluation cohorts

Source identity is content-based rather than filename-based. The first prepared
dataset persists source-group assignments, and existing samples never move
between train, validation, and test when more data is added. Reviewed HITL data
is train-only by default, so it cannot leak into the locked benchmark. These
guarantees are exact-file-content disjoint, not automatically biological-specimen
or acquisition-session disjoint. Distinct captures of one specimen therefore
need an importer-supplied group where supported or must be grouped before import.

Promotion is metric-gated on the same named metric and the same frozen cohort:

- Landmark models use normalized error on a locked validation cohort. The
  frozen test cohort is report-only and never selects the active model.
- OBB models compare the configured locked validation metric (normally
  `mAP50-95`) only when both runs also have the identical persisted evaluator
  protocol fingerprint (image size, NMS IoU, precision, and other validation
  settings). Confidence and NMS thresholds are calibrated on validation only;
  the selected thresholds are pinned to the artifact and used by inference by
  default. A user-customized detection profile is retained as an explicit
  runtime override.
- Only a model with a measured, exact-file-content-disjoint frozen validation cohort is
  activated automatically. After that, a later run must improve beyond a small
  numeric tolerance to replace it. A tie, regression, missing metric, cohort
  mismatch, evaluator-protocol mismatch, train/evaluation overlap, or (for
  landmark models) catastrophic validation tail leaves the run as a candidate.
  A first model without usable validation also remains a candidate; activating
  it is an explicit manual override recorded in the registry.

The minimum promotion delta is persisted with every run. Landmark validation
error must improve by the larger of `0.0001` absolute or `0.5%` relative; OBB
`mAP50-95` must improve by the larger of `0.001` absolute or `0.5%` relative.
These margins prevent numeric noise from replacing an active artifact. Automatic
promotion also requires at least two independent validation sources/groups; a
directional or bilateral OBB cohort must represent every configured class. Frozen
test metrics are measured every run and recorded under a separate
`testEvaluation` key, never in the metric block that selection reads, so they
are reported but structurally cannot choose a model. These gates
still compare point estimates rather than confidence intervals, so production
claims require a substantially larger representative cohort and uncertainty
analysis outside this regression workflow.

Registries retain dataset, split, schema, code, runtime, metric, initialization,
and comparison-baseline lineage. `parentModelId` is populated only when training
actually resumes or fine-tunes an earlier BioVision artifact; from-scratch dlib
runs and framework/base-checkpoint initialization record that source separately.
Each artifact also stores an attested effective-dataset closure: artifact-local
XML/export manifests plus content hashes for every consumed crop, image, label,
geometry target, split assignment, and generated training-only sample. Trainers
recompute that closure after fitting/evaluation and refuse publication if any
effective input changed mid-run.
The training protocol records deterministic seeds, augmentation inputs, worker
settings, optimizer-facing parameters, base-checkpoint identity, and evaluator
configuration. CNN lineage additionally pins the exact initializer tensor hash;
runtime lineage captures the backend dependency lock, installed distributions,
and CUDA/cuDNN/device state. Revisions are content hashes, so a changed byte
produces a new lineage identity instead of silently reusing the old one.
Landmark ID mappings and model configuration are copied into each
immutable artifact and hash-verified at inference. OBB artifacts likewise pin
their schema semantic fingerprint and orientation contract, preventing a stale
detector from silently reinterpreting class IDs after a schema change. Active
aliases and registries publish together with rollback on failure.

Imported dlib XML is normalized to canonical 512-pixel crops and zero-padded
part slots. Direct schema-ID part names are accepted; positional `0..N-1`
mapping requires explicit confirmation against the displayed schema order.
Train, validation, and optional test sources must be content-disjoint, and their
mapping plus frozen cohort metadata is verified again before training.

### Geometry-safe OBB export

Real OBB corners are validated before export. When repaired geometry extends
beyond the original image, the exporter pads the canvas and rigidly translates
the image, OBB, and landmarks together. It never clamps individual corners,
which would change edge lengths, angles, or correspondence. Invalid geometry
stops export with repair guidance.

A finalized review that explicitly accepts no boxes is a confirmed negative,
not missing data. Those images are exported with an empty YOLO label so the
detector learns from the operator's strongest false-positive evidence. They are
always train-only: an image with no geometry would otherwise change an evaluator
cohort's mAP denominator and break comparability with earlier models. A legacy
finalized image that never declared `acceptedBoxes` is still excluded, because
its emptiness is unverified rather than asserted.

Detector output uses OBB-aware suppression only. The former axis-aligned
post-deduplication step is intentionally absent because overlapping slender or
rotated specimens can share an AABB without being duplicates.

### Transactional HITL review

Selected inference images are first staged under the inference session with
their SHA-256 identity; they are not counted as training data. A committed
review atomically publishes only the reviewed content-addressed image,
canonical label, review event, finalized-image state, and session state. A
durable journal recovers an interrupted commit on the next run, and ordinary
failures roll the whole commit back. Same-name images with different bytes or
extensions get deterministic content suffixes, preventing accidental image or
label overwrites.

Each committed sample retains the exact landmark and detector model IDs,
detector artifact hash, original/reviewed prediction hashes, confidence and
uncertainty signals, review outcome, and review history. Detection-only reviews
remain eligible for OBB training but are explicitly excluded from landmark
training.

The review queue can prioritize measured detector uncertainty, landmark heatmap
entropy/confidence, horizontal-flip/TTA consistency disagreement, an OOD proxy,
and repeated failures. Missing signals—including true inter-model disagreement,
which is reserved for a future dual-model inference path—are omitted rather than
invented. Retraining batches report new, corrected, unchanged, rejected,
pending-review, and pending-commit counts since the active model.

## Development

Requirements:

- Node.js and npm
- Python environment at `venv/` with the backend dependencies installed
- dlib for landmark training; PyTorch for CNN models; Ultralytics for OBB models

Common commands from the repository root:

```powershell
npm install
npm run dev
npx tsc --noEmit
npm run lint
npm run test:contracts
npm run test:electron
.\venv\Scripts\python.exe -B -m unittest discover -s backend -p "test_*.py" -v
npm run build
```

The test suites include one mocked four-mode pipeline integration loop. It
crosses real annotation files, landmark preparation, OBB export,
immutable registries, inference adapters, a transactional Electron HITL commit,
train-only reviewed-data ingestion, retraining, validation-gated promotion, and
report-only testing while deterministically mocking optimizer, prediction, and
metric work rather than driving the renderer UI or IPC handlers. Additional
tests cover geometry preservation, frozen split
behavior, model promotion thresholds, immutable sidecars and lineage,
uncertainty/provenance propagation, and transactional HITL crash recovery.

## Training-pipeline smoke run

Run the lightweight end-to-end landmark check with:

```powershell
.\venv\Scripts\python.exe scripts\smoke_landmark_accuracy.py
```

The script creates a temporary synthetic project, trains a deliberately weak
baseline, adds mock human-reviewed corrections with retained provenance to the
training-only cohort, retrains, evaluates both artifacts on the same locked
validation IDs, verifies strict validation improvement and promotion, then
reports generalization on the untouched frozen test IDs. It cleans up its
temporary data afterward. Filesystem-level Electron tests separately run a
staged review through atomic image/label/event/finalization publication and
exercise rollback plus simulated crash recovery.

This is a real dlib execution through the production preparation, training,
registry, and promotion code, but its generated shapes are only a regression
fixture. Its improvement demonstrates that the pipeline can learn and that the
gate works; it is not evidence of biological accuracy on a production dataset.
Production accuracy must be established on a representative, independently
reviewed, frozen cohort.
