import { AnnotatedImage, TrainedModel } from "./Image";
export {};

interface SaveLabelsResult {
  ok: boolean;
}

interface TrainOptions {
  testSplit?: number;  // Fraction for test set (default 0.2)
  seed?: number;       // Random seed for reproducibility
  customOptions?: Record<string, number>;  // Custom training parameters
  speciesId?: string;  // Session-scoped training
}

interface TrainModelResult {
  ok: boolean;
  output?: string;
  error?: string;
  trainError?: number | null;  // Training set error (normalized)
  testError?: number | null;   // Test set error (normalized)
  modelPath?: string | null;   // Path to saved model
}

interface TestModelResult {
  ok: boolean;
  error?: string;
  results?: {
    dlib_error: number;
    num_images: number;
    num_landmarks: number;
    total_predictions: number;
    mean_pixel_error?: number;
    median_pixel_error?: number;
    min_pixel_error?: number;
    max_pixel_error?: number;
    per_landmark_stats?: Record<number, {
      mean: number;
      max: number;
      min: number;
      count: number;
    }>;
  };
  output?: string;
}

interface PredictImageResult {
  ok: boolean;
  data?: {
    image: string;
    landmarks: { id: number; x: number; y: number }[];
    detected_box?: {
      left: number;
      top: number;
      right: number;
      bottom: number;
      width: number;
      height: number;
    };
    image_dimensions?: {
      width: number;
      height: number;
    };
  };
  error?: string;
}

interface SelectImageFolderResult {
  canceled: boolean;
  images?: { filename: string; path: string; data: string; mimeType: string }[];
}

// Multi-specimen detection types
interface DetectionOptions {
  confThreshold?: number;  // Now used as min_area_ratio (0.01-0.2)
}

interface DetectedBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
  confidence: number;
  class_id: number;
  class_name: string;
}

interface DetectSpecimensResult {
  ok: boolean;
  boxes: DetectedBox[];
  error?: string;
  image_width?: number;
  image_height?: number;
  num_detections?: number;
  fallback?: boolean;
}

interface CheckYoloResult {
  available: boolean;
  primary_method?: string;
  error?: string;
}

declare global {
  interface SessionImage {
    filename: string;
    diskPath: string;
    data: string;
    mimeType: string;
    boxes: import("./Image").BoundingBox[];
  }

  interface SessionMeta {
    speciesId: string;
    name: string;
    imageCount: number;
    lastModified: string;
    landmarkCount: number;
  }

  interface Window {
    api: {
      saveLabels: (fileArray: AnnotatedImage []) => Promise<SaveLabelsResult>;
      trainModel: (modelName: string, options?: TrainOptions) => Promise<TrainModelResult>;
      predictImage: (imagePath: string, tag: string) => Promise<PredictImageResult>;
      selectImageFolder: () => Promise<SelectImageFolderResult>;
      getProjectRoot: () => Promise<{ projectRoot: string }>;
      selectProjectRoot: () => Promise<{ canceled?: boolean; projectRoot?: string }>;
      listModels: () => Promise<{ ok: boolean; models?: TrainedModel[]; error?: string }>;
      deleteModel: (modelName: string) => Promise<{ ok: boolean; error?: string }>;
      renameModel: (oldName: string, newName: string) => Promise<{ ok: boolean; error?: string }>;
      getModelInfo: (modelName: string) => Promise<{ ok: boolean; model?: TrainedModel; error?: string }>;
      selectImages: () => Promise<{ canceled: boolean; files?: { path: string; name: string; data: string; mimeType: string }[] }>;
      testModel: (modelName: string) => Promise<TestModelResult>;
      // Multi-specimen detection
      detectSpecimens: (imagePath: string, options?: DetectionOptions) => Promise<DetectSpecimensResult>;
      checkYolo: () => Promise<CheckYoloResult>;
      // Session management
      sessionCreate: (speciesId: string, name: string, landmarkTemplate: import("./Image").LandmarkDefinition[]) => Promise<{ ok: boolean; error?: string }>;
      sessionSaveImage: (speciesId: string, imageData: string, filename: string, mimeType: string) => Promise<{ ok: boolean; diskPath: string; error?: string }>;
      sessionSaveAnnotations: (speciesId: string, filename: string, boxes: import("./Image").BoundingBox[]) => Promise<{ ok: boolean; error?: string }>;
      sessionLoad: (speciesId: string) => Promise<{ ok: boolean; images: SessionImage[]; meta?: { name?: string; landmarkTemplate?: import("./Image").LandmarkDefinition[]; speciesId?: string }; error?: string }>;
      sessionList: () => Promise<{ ok: boolean; sessions: SessionMeta[]; error?: string }>;
      sessionDeleteImage: (speciesId: string, filename: string) => Promise<{ ok: boolean; error?: string }>;
    };
  }
}
