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
  useImportedXml?: boolean; // Train directly from xml/train_{tag}.xml
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

interface DetectionOptions {}

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

interface ImportPreAnnotatedDatasetResult {
  ok: boolean;
  canceled?: boolean;
  sourceDir?: string;
  importedImages?: number;
  importedLabels?: number;
  overwrittenImages?: number;
  overwrittenLabels?: number;
  warnings?: string[];
  error?: string;
}

interface ImportDlibXmlResult {
  ok: boolean;
  canceled?: boolean;
  trainXmlPath?: string;
  testXmlPath?: string;
  trainStats?: { num_images: number; num_boxes: number; num_parts: number };
  testStats?: { num_images: number; num_boxes: number; num_parts: number } | null;
  warnings?: string[];
  error?: string;
}

interface TrainingPreflightResult {
  ok: boolean;
  useImportedXml: boolean;
  workspaceImages?: number;
  importedImages?: number;
  totalTrainableImages?: number;
  trainXmlImages?: number;
  testXmlImages?: number;
  landmarkStatus?: "ok" | "warning";
  landmarkMessage: string;
  warnings?: string[];
  error?: string;
}

// SuperAnnotator pipeline types
interface SuperAnnotateOptions {
  confThreshold?: number;
  samEnabled?: boolean;
  maxObjects?: number;
  detectionMode?: string;  // "auto" | "manual"
  detectionPreset?: "balanced" | "precision" | "recall" | "single_object";
}

interface InstanceMetadata {
  center: [number, number];
  crop_origin: [number, number];
  crop_size: [number, number];
  rotation: number;
  scale: number;
}

interface SuperAnnotateObject {
  box: DetectedBox;
  mask_outline: [number, number][];
  landmarks: { id: number; x: number; y: number }[];
  confidence: number;
  class_name: string;
  instance_metadata: InstanceMetadata;
  detection_method: string;
}

interface SuperAnnotateResult {
  ok: boolean;
  objects: SuperAnnotateObject[];
  image_width: number;
  image_height: number;
  detection_method: string;
  num_detections: number;
  error?: string;
}

interface CheckSuperAnnotatorResult {
  available: boolean;
  mode: "auto_high_performance" | "auto_lite" | "classic_fallback";
  gpu: boolean;
  yolo_ready: boolean;
  sam2_ready: boolean;
  yolo_failed?: boolean;
  sam2_failed?: boolean;
  yolo_error?: string | null;
  sam2_error?: string | null;
  error?: string;
}

interface InitSuperAnnotatorResult {
  ok: boolean;
  status?: "ready";
  mode?: "auto_high_performance" | "auto_lite" | "classic_fallback";
  gpu?: boolean;
  yolo_loaded?: boolean;
  sam2_loaded?: boolean;
  error?: string;
}

interface SuperAnnotateProgress {
  message: string;
  percent: number;
  stage: "init" | "detection" | "segmentation" | "normalization" | "prediction" | "processing" | "training" | "done";
}

interface TrainYoloResult {
  ok: boolean;
  modelPath?: string;
  candidateModelPath?: string;
  version?: number;
  promoted?: boolean;
  candidateMap50?: number | null;
  incumbentMap50?: number | null;
  dataset?: {
    yaml_path: string;
    total_records: number;
    val_records: number;
    train_records: number;
    positive_images: number;
    negative_crops: number;
  };
  registryPath?: string;
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
      importPreAnnotatedDataset: (options?: { speciesId?: string }) => Promise<ImportPreAnnotatedDatasetResult>;
      importDlibXml: (args: { modelName: string; speciesId?: string }) => Promise<ImportDlibXmlResult>;
      trainingPreflight: (args: {
        speciesId?: string;
        modelName: string;
        useImportedXml?: boolean;
        workspaceImages?: number;
        importedImagesHint?: number;
      }) => Promise<TrainingPreflightResult>;
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
      // Classic CV detection
      detectSpecimens: (imagePath: string, options?: DetectionOptions) => Promise<DetectSpecimensResult>;
      checkYolo: () => Promise<CheckYoloResult>;
      // Session management
      sessionCreate: (speciesId: string, name: string, landmarkTemplate: import("./Image").LandmarkDefinition[]) => Promise<{ ok: boolean; error?: string }>;
      sessionSaveImage: (speciesId: string, imageData: string, filename: string, mimeType: string) => Promise<{ ok: boolean; diskPath: string; error?: string }>;
      sessionSaveAnnotations: (speciesId: string, filename: string, boxes: import("./Image").BoundingBox[]) => Promise<{ ok: boolean; error?: string }>;
      sessionAddRejectedDetection: (
        speciesId: string,
        filename: string,
        rejectedDetection: {
          left: number;
          top: number;
          width: number;
          height: number;
          confidence?: number;
          className?: string;
          detectionMethod?: string;
        }
      ) => Promise<{ ok: boolean; error?: string }>;
      sessionLoad: (speciesId: string) => Promise<{ ok: boolean; images: SessionImage[]; meta?: { name?: string; landmarkTemplate?: import("./Image").LandmarkDefinition[]; speciesId?: string }; error?: string }>;
      sessionList: () => Promise<{ ok: boolean; sessions: SessionMeta[]; error?: string }>;
      sessionDeleteImage: (speciesId: string, filename: string) => Promise<{ ok: boolean; error?: string }>;
      // SuperAnnotator pipeline
      superAnnotate: (imagePath: string, className: string, modelTag?: string, options?: SuperAnnotateOptions, speciesId?: string) => Promise<SuperAnnotateResult>;
      checkSuperAnnotator: () => Promise<CheckSuperAnnotatorResult>;
      initSuperAnnotator: () => Promise<InitSuperAnnotatorResult>;
      refineSam: (imagePath: string, objectIndex: number, clickPoint: [number, number], clickLabel: number) => Promise<{ ok: boolean; mask_outline?: [number, number][]; error?: string }>;
      trainYolo: (
        speciesId: string,
        className: string,
        epochs?: number,
        detectionPreset?: "balanced" | "precision" | "recall" | "single_object"
      ) => Promise<TrainYoloResult>;
      onSuperAnnotateProgress: (callback: (data: SuperAnnotateProgress) => void) => () => void;
    };
  }
}
