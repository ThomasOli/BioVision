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
  orientation_hint?: {
    orientation?: "left" | "right";
    confidence?: number;
    source?: string;
  };
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

  // Agentic metadata (optional)
  processingStatus?: "pending" | "predicted" | "review" | "approved";
  qualityScore?: number; // 0-1
}

// Landmark schema definition
export interface LandmarkDefinition {
  index: number; // 0, 1, 2...
  name: string; // "Left Forewing Apex"
  description?: string;
  category?: string; // "forewing", "head", "body", etc.
}

export type OrientationMode = "directional" | "bilateral" | "axial" | "invariant";

export interface OrientationPolicy {
  mode: OrientationMode;
  targetOrientation?: "left" | "right";
  headCategories?: string[];
  tailCategories?: string[];
  bilateralPairs?: [number, number][];
  pcaLevelingMode?: "off" | "on" | "auto";
  obbLevelingMode?: "on" | "off";  // Controls whether OBB rotation is applied during crop extraction
}

export interface GeometryMappingConfig {
  axisMode: "auto" | "manual_anchors";
  anchorLandmarkIds?: {
    anteriorId: number;
    posteriorId: number;
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
  models: SpeciesModel[];
  imageCount: number;
  annotationCount?: number;
  createdAt: Date;
  lastModified?: Date;
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
  predictorType?: "dlib" | "cnn" | "yolo_pose";
  status?: "active" | "deprecated";
  compatible?: boolean;
  reason?: string;
}
