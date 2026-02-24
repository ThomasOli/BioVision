import { AnnotatedImage, TrainedModel, OrientationPolicy } from "./Image";
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
  predictorType?: "dlib" | "cnn"; // Predictor backend (default: "dlib")
  cnnVariant?: string; // CNN backbone id (e.g. simplebaseline, mobilenet_v3_large)
}

interface CnnVariantOption {
  id: string;
  label: string;
  description: string;
  selectable: boolean;
  recommended?: boolean;
  reason?: string | null;
}

interface GetCnnVariantsResult {
  ok: boolean;
  variants: CnnVariantOption[];
  defaultVariant?: string | null;
  device?: "cuda" | "mps" | "cpu" | string;
  torchAvailable?: boolean;
  torchvisionAvailable?: boolean;
  gpuName?: string | null;
  gpuMemoryGb?: number | null;
  warning?: string;
}

interface TrainModelResult {
  ok: boolean;
  output?: string;
  error?: string;
  trainError?: number | null;  // Training set error (normalized)
  testError?: number | null;   // Test set error (normalized)
  modelPath?: string | null;   // Path to saved model
  auditReport?: Record<string, unknown>;  // Dataset audit results from audit_dataset.py
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
    specimens?: {
      box: {
        left: number;
        top: number;
        right: number;
        bottom: number;
        width: number;
        height: number;
        confidence?: number;
        class_id?: number;
        class_name?: string;
      };
      landmarks: { id: number; x: number; y: number }[];
      num_landmarks: number;
      inference_metadata?: {
        mask_source?: "sam2" | "rough_otsu" | string;
        pca_rotation?: number;
        pca_angle?: number;
        canonical_flip_applied?: boolean;
        direction_source?: string;
        inferred_direction?: "left" | "right" | null;
        inferred_direction_confidence?: number;
        direction_confidence?: number;
        used_flipped_crop?: boolean;
        was_flipped?: boolean;
        selection_reason?: string;
        detector_hint_orientation?: "left" | "right" | string | null;
        detector_hint_source?: string | null;
        orientation_warning?: {
          code?: string;
          message?: string;
        } | null;
      };
    }[];
    num_specimens?: number;
    inference_metadata?: {
      mask_source?: "sam2" | "rough_otsu" | string;
      pca_rotation?: number;
      pca_angle?: number;
      canonical_flip_applied?: boolean;
      direction_source?: string;
      inferred_direction?: "left" | "right" | null;
      inferred_direction_confidence?: number;
      direction_confidence?: number;
      used_flipped_crop?: boolean;
      was_flipped?: boolean;
      selection_reason?: string;
      detector_hint_orientation?: "left" | "right" | string | null;
      detector_hint_source?: string | null;
      orientation_warning?: {
        code?: string;
        message?: string;
      } | null;
    };
  };
  error?: string;
}

interface PredictOptions {
  multiSpecimen?: boolean;
  predictorType?: "dlib" | "cnn";
  allowIncompatible?: boolean;
  boxes?: Array<{
    left: number;
    top: number;
    width: number;
    height: number;
    right?: number;
    bottom?: number;
    orientation_hint?: {
      orientation?: "left" | "right";
      confidence?: number;
      source?: string;
      head_point?: [number, number];
      tail_point?: [number, number];
    };
  }>;
}

interface ModelCompatibilityIssue {
  code: string;
  severity: "error" | "warning";
  message: string;
}

interface ModelCompatibilityResult {
  ok: boolean;
  compatible: boolean;
  blocking: boolean;
  requiresOverride: boolean;
  issues: ModelCompatibilityIssue[];
  runtime?: {
    sam2Ready: boolean;
    sam2Required: boolean;
    requirementSource?: string;
    trainedMaskSource?: "none" | "segments" | "rough_otsu" | "mixed" | "unknown";
    checkedAt: string;
    error?: string;
  };
  error?: string;
}

interface SelectImageFolderResult {
  canceled: boolean;
  images?: { filename: string; path: string; data: string; mimeType: string }[];
}

interface DetectionOptions {
  speciesId?: string;
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
  orientation_override?: "left" | "right" | "uncertain";
  orientation_hint?: {
    orientation?: "left" | "right";
    confidence?: number;
    source?: string;
    head_point?: [number, number];
    tail_point?: [number, number];
  };
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

interface LoadAnnotatedFolderResult {
  ok: boolean;
  images?: Array<{
    filename: string;
    mimeType: string;
    diskPath: string;
    boxes: import("./Image").BoundingBox[];
  }>;
  unmatched?: string[];
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
  pcaMode?: "off" | "on" | "auto";
  useOrientationHint?: boolean;
}

interface InstanceMetadata {
  center: [number, number];
  crop_origin: [number, number];
  crop_size: [number, number];
  rotation: number;
  scale: number;
  canonical_flip_applied?: boolean;
  canonicalization?: Record<string, unknown>;
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

interface TrainProgressEvent {
  percent: number;
  stage: "preflight" | "prepare_dataset" | "training" | "evaluation" | "done" | "error" | string;
  message: string;
  predictorType: "dlib" | "cnn";
  modelName: string;
  details?: {
    substage?: string;
    epoch?: number;
    epochs?: number;
    loss?: number;
    lr?: number;
    epoch_sec?: number;
    elapsed_sec?: number;
    eta_sec?: number;
    samples_per_sec?: number;
    split?: string;
    eval_mode?: string;
    records_total?: number;
    records_done?: number;
    train_samples?: number;
    test_samples?: number;
    batch_size?: number;
    workers?: number;
    amp_enabled?: boolean;
    device?: string;
  };
}

interface TrainYoloResult {
  ok: boolean;
  modelPath?: string;
  candidateModelPath?: string;
  version?: number;
  promoted?: boolean;
  evaluationMetricType?: "box" | "pose" | string;
  candidateMap50?: number | null;
  candidateMap50_95?: number | null;
  incumbentMap50?: number | null;
  incumbentMap50_95?: number | null;
  candidatePoseMap50?: number | null;
  candidatePoseMap50_95?: number | null;
  candidateBoxMap50?: number | null;
  candidateBoxMap50_95?: number | null;
  incumbentPoseMap50?: number | null;
  incumbentPoseMap50_95?: number | null;
  incumbentBoxMap50?: number | null;
  incumbentBoxMap50_95?: number | null;
  candidateMetrics?: {
    box_map50?: number | null;
    box_map50_95?: number | null;
    pose_map50?: number | null;
    pose_map50_95?: number | null;
  };
  incumbentMetrics?: {
    box_map50?: number | null;
    box_map50_95?: number | null;
    pose_map50?: number | null;
    pose_map50_95?: number | null;
  };
  detectionPreset?: "balanced" | "precision" | "recall" | "single_object" | string;
  autoTune?: boolean;
  datasetSizeEffective?: number;
  datasetSizeSource?: "user" | "export" | string;
  resolvedTrainParams?: {
    size_bucket?: string;
    epochs?: number;
    batch?: number;
    freeze?: number;
    lr0?: number;
    mosaic?: number;
    close_mosaic?: number;
    degrees?: number;
    translate?: number;
    scale?: number;
    fliplr?: number;
    patience?: number;
    augment?: boolean;
    imgsz?: number;
    dataset_size?: number;
    detection_preset?: string;
  };
  preflightWarnings?: string[];
  dataset?: {
    yaml_path: string;
    total_records: number;
    val_records: number;
    train_records: number;
    positive_images: number;
    negative_crops: number;
    orientation_class_enabled?: boolean;
    real_orientation_left_boxes?: number;
    real_orientation_right_boxes?: number;
    real_orientation_unknown_boxes?: number;
    orientation_preflight_warnings?: string[];
    synthetic_mode?: "standard" | "supplement" | string;
    synthetic_max_images?: number | null;
    synthetic_max_instances?: number | null;
    synthetic_instances_generated?: number;
    synthetic_left_instances?: number;
    synthetic_right_instances?: number;
  };
  registryPath?: string;
  error?: string;
}

interface YoloTrainPlanResult {
  ok: boolean;
  dataset?: {
    yaml_path: string;
    finalized_only?: boolean;
    skipped_unfinalized_images?: number;
    finalized_label_files?: number;
    finalized_fallback_to_boxes?: number;
    total_records: number;
    val_records: number;
    train_records: number;
    positive_images: number;
    negative_crops: number;
    head_id?: number | null;
    tail_id?: number | null;
    pose_boxes_with_kp?: number;
    total_boxes?: number;
    num_synthetic?: number;
    synthetic_enabled?: boolean;
    synthetic_disabled_reason?: string | null;
    orientation_class_enabled?: boolean;
    real_orientation_left_boxes?: number;
    real_orientation_right_boxes?: number;
    real_orientation_unknown_boxes?: number;
    orientation_preflight_warnings?: string[];
    synthetic_mode?: "standard" | "supplement" | string;
    synthetic_max_images?: number | null;
    synthetic_max_instances?: number | null;
    synthetic_instances_generated?: number;
    synthetic_left_instances?: number;
    synthetic_right_instances?: number;
    use_pose?: boolean;
  };
  preflightWarnings?: string[];
  usePose?: boolean;
  detectionPreset?: "balanced" | "precision" | "recall" | "single_object" | string;
  autoTune?: boolean;
  datasetSizeEffective?: number;
  datasetSizeSource?: "user" | "export" | string;
  resolvedTrainParams?: {
    size_bucket?: string;
    epochs?: number;
    batch?: number;
    freeze?: number;
    lr0?: number;
    mosaic?: number;
    close_mosaic?: number;
    degrees?: number;
    translate?: number;
    scale?: number;
    fliplr?: number;
    patience?: number;
    augment?: boolean;
    imgsz?: number;
    dataset_size?: number;
    detection_preset?: string;
  };
  error?: string;
}

interface InferenceReviewDraftSpecimen {
  box: {
    left: number;
    top: number;
    width: number;
    height: number;
    confidence?: number;
    class_id?: number;
    class_name?: string;
    orientation_override?: "left" | "right" | "uncertain";
    orientation_hint?: {
      orientation?: "left" | "right";
      confidence?: number;
      source?: string;
      head_point?: [number, number];
      tail_point?: [number, number];
    };
  };
  landmarks: { id: number; x: number; y: number }[];
}

interface InferenceReviewDraftItem {
  key: string;
  imagePath: string;
  filename: string;
  specimens: InferenceReviewDraftSpecimen[];
  edited: boolean;
  saved: boolean;
  updatedAt: string;
}

interface InferenceSessionManifest {
  version: 1;
  sessionId: string;
  speciesId: string;
  models: {
    landmark: {
      key: string;
      name?: string;
      predictorType?: "dlib" | "cnn" | "yolo_pose";
    };
    detection: {
      key: string;
      name?: string;
    };
  };
  createdAt: string;
  updatedAt: string;
}

interface RetrainQueueItem {
  key: string;
  speciesId: string;
  inferenceSessionId?: string;
  landmarkModelKey?: string;
  landmarkModelName?: string;
  landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
  detectionModelKey?: string;
  detectionModelName?: string;
  filename: string;
  imagePath?: string;
  source: string;
  boxesCount: number;
  landmarksCount: number;
  queuedAt: string;
  updatedAt: string;
}

declare global {
  interface SessionImage {
    filename: string;
    diskPath: string;
    data: string;
    mimeType: string;
    boxes: import("./Image").BoundingBox[];
    finalized?: boolean;
  }

interface SessionMeta {
  speciesId: string;
  name: string;
  imageCount: number;
  lastModified: string;
  landmarkCount: number;
  orientationPolicy?: OrientationPolicy;
  orientationPolicyConfigured?: boolean;
}

  interface Window {
    api: {
      saveLabels: (fileArray: AnnotatedImage []) => Promise<SaveLabelsResult>;
      trainModel: (modelName: string, options?: TrainOptions) => Promise<TrainModelResult>;
      getCnnVariants: () => Promise<GetCnnVariantsResult>;
      importPreAnnotatedDataset: (options?: { speciesId?: string }) => Promise<ImportPreAnnotatedDatasetResult>;
      importDlibXml: (args: { modelName: string; speciesId?: string }) => Promise<ImportDlibXmlResult>;
      trainingPreflight: (args: {
        speciesId?: string;
        modelName: string;
        useImportedXml?: boolean;
        workspaceImages?: number;
        importedImagesHint?: number;
      }) => Promise<TrainingPreflightResult>;
      predictImage: (
        imagePath: string,
        tag: string,
        speciesId?: string,
        options?: PredictOptions
      ) => Promise<PredictImageResult>;
      checkModelCompatibility: (args: {
        speciesId?: string;
        modelName: string;
        predictorType?: "dlib" | "cnn";
        includeRuntime?: boolean;
      }) => Promise<ModelCompatibilityResult>;
      selectImageFolder: () => Promise<SelectImageFolderResult>;
      getProjectRoot: () => Promise<{ projectRoot: string }>;
      selectProjectRoot: () => Promise<{ canceled?: boolean; projectRoot?: string }>;
      listModels: (speciesId?: string) => Promise<{ ok: boolean; models?: TrainedModel[]; error?: string }>;
      deleteModel: (
        modelName: string,
        speciesId?: string,
        predictorType?: "dlib" | "cnn" | "yolo_pose"
      ) => Promise<{ ok: boolean; error?: string }>;
      renameModel: (
        oldName: string,
        newName: string,
        speciesId?: string,
        predictorType?: "dlib" | "cnn" | "yolo_pose"
      ) => Promise<{ ok: boolean; error?: string }>;
      getModelInfo: (modelName: string, speciesId?: string) => Promise<{ ok: boolean; model?: TrainedModel; error?: string }>;
      selectImages: () => Promise<{ canceled: boolean; files?: { path: string; name: string; data: string; mimeType: string }[] }>;
      selectFolderPath: () => Promise<{ canceled: boolean; folderPath?: string }>;
      selectAnnotationFile: () => Promise<{ canceled: boolean; filePath?: string }>;
      loadAnnotatedFolder: (args: {
        imageFolderPath: string;
        annotationFilePath: string;
        speciesId: string;
      }) => Promise<LoadAnnotatedFolderResult>;
      testModel: (modelName: string, speciesId?: string) => Promise<TestModelResult>;
      // Classic CV detection
      detectSpecimens: (imagePath: string, options?: DetectionOptions) => Promise<DetectSpecimensResult>;
      checkYolo: () => Promise<CheckYoloResult>;
      // Session management
      sessionCreate: (
        speciesId: string,
        name: string,
        landmarkTemplate: import("./Image").LandmarkDefinition[],
        orientationPolicy?: OrientationPolicy
      ) => Promise<{ ok: boolean; error?: string }>;
      sessionUpdateOrientationPolicy: (
        speciesId: string,
        orientationPolicy: OrientationPolicy
      ) => Promise<{ ok: boolean; error?: string }>;
      sessionSaveImage: (speciesId: string, imageData: string, filename: string, mimeType: string) => Promise<{ ok: boolean; diskPath: string; error?: string }>;
      sessionSaveAnnotations: (speciesId: string, filename: string, boxes: import("./Image").BoundingBox[]) => Promise<{ ok: boolean; error?: string }>;
      sessionFinalizeAcceptedBoxes: (
        speciesId: string,
        filename: string,
        boxes: {
          left: number;
          top: number;
          width: number;
          height: number;
          landmarks?: { id: number; x: number; y: number; isSkipped?: boolean }[];
        }[],
        imagePath?: string
      ) => Promise<{
        ok: boolean;
        finalized?: boolean;
        acceptedCount?: number;
        signature?: string;
        segmentSaveAttempted?: boolean;
        skipped?: boolean;
        error?: string;
      }>;
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
      sessionLoad: (speciesId: string) => Promise<{
        ok: boolean;
        images: SessionImage[];
        meta?: {
          name?: string;
          landmarkTemplate?: import("./Image").LandmarkDefinition[];
          speciesId?: string;
          orientationPolicy?: OrientationPolicy;
          orientationPolicyConfigured?: boolean;
        };
        error?: string;
      }>;
      sessionList: () => Promise<{ ok: boolean; sessions: SessionMeta[]; error?: string }>;
      sessionDeleteImage: (speciesId: string, filename: string) => Promise<{ ok: boolean; error?: string }>;
      sessionDeleteAllImages: (speciesId: string) => Promise<{ ok: boolean; error?: string }>;
      // SuperAnnotator pipeline
      superAnnotate: (imagePath: string, className: string, modelTag?: string, options?: SuperAnnotateOptions, speciesId?: string) => Promise<SuperAnnotateResult>;
      checkSuperAnnotator: () => Promise<CheckSuperAnnotatorResult>;
      initSuperAnnotator: () => Promise<InitSuperAnnotatorResult>;
      refineSam: (imagePath: string, objectIndex: number, clickPoint: [number, number], clickLabel: number) => Promise<{ ok: boolean; mask_outline?: [number, number][]; error?: string }>;
      resegmentBox: (imagePath: string, boxXyxy: [number, number, number, number]) => Promise<{ ok: boolean; maskOutline?: [number, number][]; score?: number; error?: string }>;
      trainYolo: (
        speciesId: string,
        className: string,
        epochs?: number,
        detectionPreset?: "balanced" | "precision" | "recall" | "single_object",
        datasetSize?: number,
        autoTune?: boolean
      ) => Promise<TrainYoloResult>;
      getYoloTrainPlan: (
        speciesId: string,
        className: string,
        epochs?: number,
        detectionPreset?: "balanced" | "precision" | "recall" | "single_object",
        datasetSize?: number,
        autoTune?: boolean
      ) => Promise<YoloTrainPlanResult>;
      onSuperAnnotateProgress: (callback: (data: SuperAnnotateProgress) => void) => () => void;
      onPredictProgress: (callback: (data: { percent: number; stage: string }) => void) => () => void;
      onTrainProgress: (callback: (data: TrainProgressEvent) => void) => () => void;
      sessionSaveInferenceCorrection: (
        speciesId: string,
        imagePath: string,
        box?: { left: number; top: number; width: number; height: number },
        landmarks?: { id: number; x: number; y: number }[],
        filename?: string,
        specimens?: {
          box: {
            left: number;
            top: number;
            width: number;
            height: number;
            confidence?: number;
            class_id?: number;
            class_name?: string;
            orientation_override?: "left" | "right" | "uncertain";
            orientation_hint?: {
              orientation?: "left" | "right";
              confidence?: number;
              source?: string;
              head_point?: [number, number];
              tail_point?: [number, number];
            };
          };
          landmarks: { id: number; x: number; y: number }[];
        }[],
        rejectedDetections?: {
          left: number;
          top: number;
          width: number;
          height: number;
          confidence?: number;
          className?: string;
          detectionMethod?: string;
        }[],
        options?: { allowEmpty?: boolean }
      ) => Promise<{ ok: boolean; savedPath?: string; error?: string }>;
      sessionSaveDetectionCorrection: (
        speciesId: string,
        imagePath: string,
        boxes: { left: number; top: number; width: number; height: number }[],
        imageWidth: number,
        imageHeight: number,
        filename?: string
      ) => Promise<{ ok: boolean; savedPath?: string; error?: string }>;
      sessionOpenInferenceSession: (args: {
        speciesId: string;
        landmarkModelKey: string;
        landmarkModelName?: string;
        landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
        detectionModelKey?: string;
        detectionModelName?: string;
      }) => Promise<{
        ok: boolean;
        inferenceSessionId?: string;
        manifest?: InferenceSessionManifest;
        error?: string;
      }>;
      sessionSaveInferenceReviewDraft: (
        speciesId: string,
        inferenceSessionId?: string,
        imagePath: string,
        specimens: InferenceReviewDraftSpecimen[],
        options?: { filename?: string; edited?: boolean; saved?: boolean; clear?: boolean }
      ) => Promise<{ ok: boolean; error?: string }>;
      sessionLoadInferenceReviewDrafts: (
        speciesId: string,
        inferenceSessionId?: string
      ) => Promise<{ ok: boolean; drafts?: InferenceReviewDraftItem[]; error?: string }>;
      sessionQueueRetrainItem: (
        speciesId: string,
        inferenceSessionId?: string,
        filename: string,
        options?: {
          imagePath?: string;
          source?: string;
          boxesCount?: number;
          landmarksCount?: number;
          landmarkModelKey?: string;
          landmarkModelName?: string;
          landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
          detectionModelKey?: string;
          detectionModelName?: string;
        }
      ) => Promise<{ ok: boolean; item?: RetrainQueueItem; queuedCount?: number; error?: string }>;
      sessionGetRetrainQueue: (
        speciesId: string,
        inferenceSessionId?: string
      ) => Promise<{ ok: boolean; items?: RetrainQueueItem[]; count?: number; error?: string }>;
      sessionClearRetrainQueue: (
        speciesId: string,
        inferenceSessionId?: string,
        filenames?: string[]
      ) => Promise<{ ok: boolean; count?: number; error?: string }>;
    };
  }
}
