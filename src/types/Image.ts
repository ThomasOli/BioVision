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
  diskPath?: string; // Persisted path in session directory (survives restarts)

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
  models: SpeciesModel[];
  imageCount: number;
  annotationCount?: number;
  createdAt: Date;
  lastModified?: Date;
}

export interface SpeciesModel {
  version: string; // "v1", "v2"
  modelPath: string;
  trainedAt: Date;
  accuracy?: number;
  trainingImageCount?: number;
  landmarkCount?: number;
}

// Correction tracking for Judge Agent
export interface Correction {
  id: string;
  imageId: string;
  boxId: number;
  landmarkId: number;
  predictedX: number;
  predictedY: number;
  correctedX: number;
  correctedY: number;
  error: number; // Euclidean distance
  confidence: number;
  modelVersion: string;
  correctedAt: Date;
  correctedBy?: "human" | "auto";
}

// Tool modes for the annotation workflow
// Simplified: only landmark mode needed (boxes auto-created)
export type ToolMode = 'landmark';

// App navigation views
export type AppView = 'landing' | 'workspace' | 'models' | 'inference' | 'agent';

// Trained model metadata
export interface TrainedModel {
  name: string;
  path: string;
  size: number;
  createdAt: Date;
}
