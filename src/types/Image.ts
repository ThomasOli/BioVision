export interface Point {
  x: number;
  y: number;
  id: number;

  // Skip flag for landmarks that don't apply to specimen
  isSkipped?: boolean; // True if landmark was skipped (e.g., fish without second dorsal fin)

  // Agentic fields (optional)
  label?: string; // "Left Forewing Apex"
  confidence?: number; // 0.0-1.0 from ensemble
  isPredicted?: boolean; // Auto-landmarked?
  isCorrected?: boolean; // Human corrected?
  predictionVersion?: string; // "monarch_v2"
}

export interface BoundingBox {
  id: number;
  left: number;
  top: number;
  width: number;
  height: number;
  landmarks: Point[];

  // Agentic fields (optional)
  confidence?: number; // Overall box confidence
  source?: "manual" | "predicted" | "corrected";
  predictedBy?: string; // Model name

  // SuperAnnotator fields
  maskOutline?: [number, number][]; // SAM2 polygon for canvas overlay
  className?: string; // Detected class (user's prompt text)
  instanceMetadata?: {
    center: [number, number];
    crop_origin: [number, number];
    crop_size: [number, number];
    rotation: number;
    scale: number;
  };
  detectionMethod?: string; // "yolo_obb", "yolo_obb+sam2", etc.
  // OBB fields (set when annotated via the session OBB detector or manual OBB tool)
  angle?: number;                  // OBB rotation angle in degrees
  obbCorners?: [number, number][]; // 4 corners [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
  class_id?: number;               // 0=canonical (left/up), 1=mirror (right/down), from OBB orientation
  orientation_override?: OrientationLabel;
  orientation_hint?: {
    orientation?: StoredOrientationLabel;
    confidence?: number;
    source?: string;
  };
}

export type FinalizePhase =
  | "idle"
  | "queued"
  | "running"
  | "saved"
  | "already_finalized"
  | "finalized_without_segments"
  | "failed";

export interface FinalizeFailureDetail {
  index: number;
  status: "saved" | "failed";
  maskSource?: string;
  reason?: string;
}

export interface FinalizePhaseMetadata {
  state: Exclude<FinalizePhase, "idle">;
  signature?: string;
  updatedAt: string;
  reason?: string;
  expectedCount?: number;
  savedCount?: number;
  details?: FinalizeFailureDetail[];
}

export interface AnnotatedImage {
  id: number;
  path: string;
  url: string;
  filename: string;
  boxes: BoundingBox[];
  selectedBoxId: number | null;
  history: BoundingBox[][];
  future: BoundingBox[][];

  // Session context
  speciesId?: string;
  diskPath?: string;    // Persisted path in session directory (survives restarts)
  isFinalized?: boolean; // True once the user has clicked "Finalize This Image"
  hasBoxes?: boolean; // Lightweight metadata from session restore for lazy box hydration
  finalizePhase?: FinalizePhaseMetadata;

  // Agentic metadata (optional)
  processingStatus?: "pending" | "predicted" | "review" | "approved";
  qualityScore?: number; // 0-1
}

// Landmark schema definition
export interface LandmarkDefinition {
  index: number; // 1, 2, 3...
  name: string; // "Left Forewing Apex"
  description?: string;
  category?: string; // "forewing", "head", "body", etc.
}

export type OrientationMode = "directional" | "bilateral" | "axial" | "invariant";
export type BilateralClassAxis = "vertical_obb";
export type StoredOrientationLabel = "left" | "right" | "up" | "down";
export type OrientationLabel = StoredOrientationLabel | "uncertain";
export type ObbModelTier = "nano" | "small" | "medium" | "large";
export type ObbImageSize = 640 | 960 | 1280;
export type ObbDetectionPreset =
  | "balanced"
  | "precision"
  | "recall"
  | "single_object"
  | "custom";

export interface RepresentativeImageDimensions {
  width: number;
  height: number;
  sampleCount?: number;
  megapixels?: number;
}

export interface OrientationPolicy {
  mode: OrientationMode;
  targetOrientation?: "left" | "right";
  headCategories?: string[];
  tailCategories?: string[];
  anteriorAnchorIds?: number[];
  posteriorAnchorIds?: number[];
  bilateralPairs?: [number, number][];
  bilateralClassAxis?: BilateralClassAxis;
  obbLevelingMode?: "on" | "off";  // Controls whether OBB rotation is applied during crop extraction
}

export interface ObbTrainingSettings {
  modelTier?: ObbModelTier;
  imgsz?: ObbImageSize;
  epochs?: number;
  batch?: number;
  iou?: number;
  cls?: number;
  box?: number;
}

export interface ObbDetectionSettings {
  detectionPreset?: ObbDetectionPreset;
  conf?: number;
  nmsIou?: number;
  maxObjects?: number;
  imgsz?: ObbImageSize;
}

export interface ObbTrainProgressDetails {
  epoch?: number;
  epochs?: number;
  batch?: number;
  batches?: number;
  loss?: number;
  lr?: number;
  elapsed_sec?: number;
  eta_sec?: number;
  workers?: number;
  device?: string;
  platform?: string;
  amp_enabled?: boolean;
}

export interface ObbTrainProgressEvent {
  percent: number;
  stage: string;
  message: string;
  details?: ObbTrainProgressDetails;
}

export interface GeometryMappingConfig {
  axisMode: "auto" | "manual_anchors";
  anchorLandmarkIds?: {
    anteriorIds: number[];
    posteriorIds: number[];
  };
  paddingMode: "tight" | "asymmetric";
  paddingProfile?: {
    forwardPct: number;
    backwardPct: number;
    topPct: number;
    bottomPct: number;
  };
}

export interface LandmarkSchema {
  id: string;
  name: string;
  description: string;
  landmarks: LandmarkDefinition[];
}

export interface ReusableSchemaTemplate extends LandmarkSchema {
  kind: "custom";
  orientationPolicy?: OrientationPolicy;
  sourcePresetId?: string;
  createdAt: string;
  updatedAt: string;
}

// Species registry
export interface Species {
  id: string; // UUID
  name: string;
  scientificName?: string; // "Danaus plexippus"
  description?: string;
  taxonomy?: {
    genus: string;
    species: string;
    family?: string;
  };
  landmarkTemplate: LandmarkDefinition[];
  orientationPolicy?: OrientationPolicy;
  obbTrainingSettings?: ObbTrainingSettings;
  obbDetectionSettings?: ObbDetectionSettings;
  representativeImageDimensions?: RepresentativeImageDimensions;
  models: SpeciesModel[];
  imageCount: number;
  annotationCount?: number;
  createdAt: string;
  lastModified?: string;
}

interface SpeciesModel {
  version: string; // "v1", "v2"
  modelPath: string;
  trainedAt: Date;
  accuracy?: number;
  trainingImageCount?: number;
  landmarkCount?: number;
}

// App navigation views
export type AppView = 'landing' | 'workspace' | 'models' | 'inference' | 'agent';

// Trained model metadata
export interface TrainedModel {
  name: string;
  path: string;
  size: number;
  createdAt: Date;
  predictorType?: "dlib" | "cnn";
  modelKind?: "landmark" | "obb_detector";
  speciesId?: string;
  schemaName?: string;
  status?: "active" | "deprecated";
  compatible?: boolean;
  reason?: string;
}
