import { AnnotatedImage, GeometryMappingConfig, TrainedModel, OrientationPolicy, ReusableSchemaTemplate } from "./Image";
export {};

interface SaveLabelsResult {
  ok: boolean;
}

interface TrainOptions {
  testSplit?: number;  // Fraction for test set (default 0.2)
  seed?: number;       // Random seed for reproducibility
  customOptions?: Record<string, number | boolean>;  // Custom training parameters
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
  recommendationReason?: string | null;
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
  trainError?: number | null;        // Training set mean error (normalized)
  testError?: number | null;         // Test set mean error (normalized)
  trainMedianError?: number | null;  // Training set median error (normalized)
  testMedianError?: number | null;   // Test set median error (normalized)
  modelPath?: string | null;         // Path to saved model
  auditReport?: Record<string, unknown>;  // Dataset audit results from audit_dataset.py
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
        canonical_flip_applied?: boolean;
        direction_source?: string;
        inferred_direction?: "left" | "right" | null;
        inferred_direction_confidence?: number;
        direction_confidence?: number;
        used_flipped_crop?: boolean;
        was_flipped?: boolean;
        selection_reason?: string;
        detector_hint_orientation?: import("./Image").StoredOrientationLabel | string | null;
        detector_hint_source?: string | null;
        orientation_warning?: {
          code?: string;
          message?: string;
        } | null;
        clamped_landmark_count?: number;
        inferenceSignature?: string;
      };
    }[];
    num_specimens?: number;
    boxSignature?: string;
    inferenceSignature?: string;
    inference_metadata?: {
      mask_source?: "sam2" | "rough_otsu" | string;
      canonical_flip_applied?: boolean;
      direction_source?: string;
      inferred_direction?: "left" | "right" | null;
      inferred_direction_confidence?: number;
      direction_confidence?: number;
      used_flipped_crop?: boolean;
      was_flipped?: boolean;
      selection_reason?: string;
      detector_hint_orientation?: import("./Image").StoredOrientationLabel | string | null;
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
    obbCorners?: [number, number][];
    angle?: number;
    class_id?: number;
    orientation_hint?: {
      orientation?: import("./Image").StoredOrientationLabel;
      confidence?: number;
      source?: string;
      head_point?: [number, number];
      tail_point?: [number, number];
    };
  }>;
}

interface PredictBatchArgs {
  speciesId?: string;
  modelName: string;
  predictorType?: "dlib" | "cnn";
  allowIncompatible?: boolean;
  items: Array<{
    batchIndex: number;
    imagePath: string;
    filename?: string;
    boxes: NonNullable<PredictOptions["boxes"]>;
  }>;
}

interface PredictBatchResult {
  ok: boolean;
  results?: Array<{
    batchIndex: number;
    imagePath: string;
    filename?: string;
    ok: boolean;
    data?: PredictImageResult["data"];
    error?: string;
  }>;
  error?: string;
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
    trainedMaskSource?: "none" | "segments" | "rough_otsu" | "mixed" | "unknown" | "obb_geometry";
    checkedAt: string;
    error?: string;
  };
  error?: string;
  obbDetectorReady?: boolean;
  obbDetectorPath?: string;
}

interface SelectImageFolderResult {
  canceled: boolean;
  images?: { filename: string; path: string; data: string; mimeType: string }[];
}

interface DetectionOptions {
  speciesId?: string;
  conf?: number;
  nmsIou?: number;
  maxObjects?: number;
  imgsz?: import("./Image").ObbImageSize;
  detectionPreset?: import("./Image").ObbDetectionPreset;
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
  orientation_override?: import("./Image").OrientationLabel;
  orientation_hint?: {
    orientation?: import("./Image").StoredOrientationLabel;
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
  importSummary?: {
    sourceObbPreserved: number;
    manualAnchorDerived: number;
    autoDerived: number;
    fallbackBoxes: number;
    usedAsymmetricPadding: boolean;
    translatedToFitImage?: number;
  };
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
  samEnabled?: boolean;
  maxObjects?: number;
  detectionMode?: string;  // "auto" | "manual"
  detectionPreset?: import("./Image").ObbDetectionPreset;
  conf?: number;
  nmsIou?: number;
  imgsz?: import("./Image").ObbImageSize;
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
  class_id?: number;
  orientation_hint?: {
    orientation?: import("./Image").StoredOrientationLabel;
    confidence?: number;
    source?: string;
  };
  obb?: {
    angle: number;
    corners: [number, number][];
    center: [number, number];
    size: [number, number];
  } | null;
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
  mode: "unknown" | "auto_high_performance" | "auto_lite" | "classic_fallback";
  gpu: boolean;
  yolo_ready: boolean;
  sam2_ready: boolean;
  runtimeState?: "not_started" | "checking" | "not_initialized" | "initializing" | "ready" | "failed";
  statusSource?: "local_estimate" | "python_probe" | "python_check";
  pythonPath?: string;
  usingRepoVenv?: boolean;
  yolo_failed?: boolean;
  sam2_failed?: boolean;
  yolo_error?: string | null;
  sam2_error?: string | null;
  error?: string;
  obbCapable?: boolean;
  obbModelTier?: "none" | "nano" | "small" | "medium";
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

interface SegmentSaveStatus {
  state:
    | "idle"
    | "queued"
    | "running"
    | "saved"
    | "already_finalized"
    | "finalized_without_segments"
    | "skipped"
    | "failed";
  signature?: string;
  updatedAt: string;
  reason?: string;
  expectedCount?: number;
  savedCount?: number;
  details?: import("./Image").FinalizeFailureDetail[];
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
    obbCorners?: [number, number][];
    angle?: number;
    orientation_override?: import("./Image").OrientationLabel;
    orientation_hint?: {
      orientation?: import("./Image").StoredOrientationLabel;
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
  reviewComplete?: boolean;
  committedAt?: string | null;
  landmarkModelKey?: string;
  landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
  boxSignature?: string;
  inferenceSignature?: string;
  updatedAt: string;
}

interface InferenceSessionManifest {
  version: 1;
  sessionId: string;
  speciesId: string;
  displayName?: string;
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
  preferences?: {
    lastUsedLandmarkModelKey?: string;
    lastUsedPredictorType?: "dlib" | "cnn" | "yolo_pose";
    detectionModelKey?: string;
    detectionModelName?: string;
  };
  createdAt: string;
  updatedAt: string;
}

interface InferenceSessionSummary {
  speciesId: string;
  schemaName: string;
  schemaImageCount: number;
  schemaUpdatedAt: string;
  schemaGroupKey: string;
  canonicalSpeciesId: string;
  hiddenSessionCount?: number;
  exists: boolean;
  inferenceSessionId?: string;
  displayName?: string;
  createdAt?: string;
  updatedAt?: string;
  migratedFrom?: string;
}

declare global {
  interface SessionImage {
    filename: string;
    diskPath: string;
    mimeType: string;
    hasBoxes?: boolean;
    boxes?: import("./Image").BoundingBox[];
    finalized?: boolean;
  }

interface AugmentationPolicy {
  gravity_aligned?: boolean;
  rotation_range?: [number, number];
  scale_range?: [number, number];
  flip_prob?: number;
  vertical_flip_prob?: number;
  rotate_180_prob?: number;
  translate_ratio?: number;
}

interface SessionMeta {
  speciesId: string;
  name: string;
  imageCount: number;
  lastModified: string;
  landmarkCount: number;
  schemaFingerprint?: string;
  schemaKind?: "default" | "custom";
  schemaSourceId?: string;
  orientationPolicy?: OrientationPolicy;
  orientationPolicyConfigured?: boolean;
  obbDetectorReady?: boolean;
  augmentationPolicy?: AugmentationPolicy;
  obbTrainingSettings?: import("./Image").ObbTrainingSettings;
  obbDetectionSettings?: import("./Image").ObbDetectionSettings;
  obbTrainingSettingsCustomized?: boolean;
  obbDetectionSettingsCustomized?: boolean;
  representativeImageDimensions?: import("./Image").RepresentativeImageDimensions;
}

  interface Window {
    api: {
      saveLabels: (fileArray: AnnotatedImage []) => Promise<SaveLabelsResult>;
      trainModel: (modelName: string, options?: TrainOptions) => Promise<TrainModelResult>;
      getCnnVariants: () => Promise<GetCnnVariantsResult>;
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
      predictImagesBatch: (args: PredictBatchArgs) => Promise<PredictBatchResult>;
      checkModelCompatibility: (args: {
        speciesId?: string;
        modelName: string;
        predictorType?: "dlib" | "cnn";
        includeRuntime?: boolean;
      }) => Promise<ModelCompatibilityResult>;
      selectImageFolder: () => Promise<SelectImageFolderResult>;
      getProjectRoot: () => Promise<{ projectRoot: string }>;
      selectProjectRoot: () => Promise<{ canceled?: boolean; projectRoot?: string }>;
      listModels: (args?: string | {
        speciesId?: string;
        activeOnly?: boolean;
        includeDeprecated?: boolean;
      }) => Promise<{ ok: boolean; models?: TrainedModel[]; error?: string }>;
      deleteModel: (
        modelName: string,
        speciesId?: string,
        predictorType?: "dlib" | "cnn",
        modelKind?: "landmark" | "obb_detector"
      ) => Promise<{ ok: boolean; error?: string }>;
      renameModel: (
        oldName: string,
        newName: string,
        speciesId?: string,
        predictorType?: "dlib" | "cnn",
        modelKind?: "landmark" | "obb_detector"
      ) => Promise<{ ok: boolean; error?: string }>;
      selectImages: () => Promise<{ canceled: boolean; files?: { path: string; name: string; data: string; mimeType: string }[] }>;
      selectFolderPath: () => Promise<{ canceled: boolean; folderPath?: string }>;
      selectAnnotationFile: () => Promise<{ canceled: boolean; filePath?: string }>;
      loadAnnotatedFolder: (args: {
        imageFolderPath: string;
        annotationFilePath: string;
        speciesId: string;
        geometryConfig?: GeometryMappingConfig;
        useSam2BoxDerivation?: boolean;
      }) => Promise<LoadAnnotatedFolderResult>;
      importPreAnnotatedDataset: (args?: {
        speciesId?: string;
        geometryConfig?: GeometryMappingConfig;
      }) => Promise<{
        ok: boolean;
        canceled?: boolean;
        sourceDir?: string;
        importedImages?: number;
        importedLabels?: number;
        overwrittenImages?: number;
        overwrittenLabels?: number;
        warnings?: string[];
        importSummary?: {
          sourceObbPreserved: number;
          manualAnchorDerived: number;
          autoDerived: number;
          fallbackBoxes: number;
          usedAsymmetricPadding: boolean;
          translatedToFitImage?: number;
        };
        error?: string;
      }>;
      // Classic CV detection
      detectSpecimens: (imagePath: string, options?: DetectionOptions) => Promise<DetectSpecimensResult>;
      // Session management
      sessionCreate: (
        speciesId: string,
        name: string,
        landmarkTemplate: import("./Image").LandmarkDefinition[],
        orientationPolicy?: OrientationPolicy,
        schemaMetadata?: {
          schemaKind?: "default" | "custom";
          schemaSourceId?: string;
          schemaFingerprint?: string;
        }
      ) => Promise<{ ok: boolean; error?: string }>;
      sessionUpdateOrientationPolicy: (
        speciesId: string,
        orientationPolicy: OrientationPolicy
      ) => Promise<{ ok: boolean; error?: string }>;
      sessionUpdateAugmentation: (
        speciesId: string,
        augmentationPolicy: AugmentationPolicy
      ) => Promise<{ ok: boolean; error?: string }>;
      sessionUpdateObbDetectorSettings: (
        speciesId: string,
        settings: {
          obbTrainingSettings?: import("./Image").ObbTrainingSettings;
          obbDetectionSettings?: import("./Image").ObbDetectionSettings;
          obbTrainingSettingsCustomized?: boolean;
          obbDetectionSettingsCustomized?: boolean;
        }
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
          orientation_override?: import("./Image").OrientationLabel;
          obbCorners?: [number, number][];
          angle?: number;
          class_id?: number;
          orientation_hint?: {
            orientation?: import("./Image").StoredOrientationLabel;
            confidence?: number;
            source?: string;
          };
          landmarks?: { id: number; x: number; y: number; isSkipped?: boolean }[];
        }[],
        imagePath?: string,
        generateSegments?: boolean
      ) => Promise<{
        ok: boolean;
        finalized?: boolean;
        queued?: boolean;
        acceptedCount?: number;
        signature?: string;
        skipped?: boolean;
        segmentSaveQueued?: boolean;
        segmentQueueState?:
          | "queued"
          | "running"
          | "saved"
          | "already_finalized"
          | "finalized_without_segments"
          | "skipped";
        reason?: string;
        expectedCount?: number;
        savedCount?: number;
        details?: import("./Image").FinalizeFailureDetail[];
        error?: string;
      }>;
      sessionUnfinalizeImage: (
        speciesId: string,
        filename: string,
        imagePath?: string
      ) => Promise<{
        ok: boolean;
        unfinalized?: boolean;
        filename?: string;
        removedFromList?: boolean;
        removedSegments?: number;
        hadFinalizedDetection?: boolean;
        error?: string;
      }>;
      sessionUnfinalizeImages: (
        speciesId: string,
        filenames?: string[]
      ) => Promise<{
        ok: boolean;
        requested: number;
        succeeded: number;
        failed: number;
        removedSegmentsTotal: number;
        errors?: { filename: string; error: string }[];
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
        warnings?: string[];
        meta?: {
          name?: string;
          landmarkTemplate?: import("./Image").LandmarkDefinition[];
          speciesId?: string;
          imageCount?: number;
          schemaFingerprint?: string;
          schemaKind?: "default" | "custom";
          schemaSourceId?: string;
          orientationPolicy?: OrientationPolicy;
          orientationPolicyConfigured?: boolean;
          augmentationPolicy?: AugmentationPolicy;
          obbDetectorReady?: boolean;
          obbTrainingSettings?: import("./Image").ObbTrainingSettings;
          obbDetectionSettings?: import("./Image").ObbDetectionSettings;
          obbTrainingSettingsCustomized?: boolean;
          obbDetectionSettingsCustomized?: boolean;
          representativeImageDimensions?: import("./Image").RepresentativeImageDimensions;
        };
        error?: string;
      }>;
      sessionLoadAnnotation: (
        speciesId: string,
        filename: string
      ) => Promise<{
        ok: boolean;
        boxes: import("./Image").BoundingBox[];
        finalized?: boolean;
        warnings?: string[];
        error?: string;
      }>;
      sessionGetSegmentSaveStatus: (
        speciesId: string,
        filename: string
      ) => Promise<{ ok: boolean; status?: SegmentSaveStatus; error?: string }>;
      sessionList: () => Promise<{ ok: boolean; sessions: SessionMeta[]; error?: string }>;
      sessionDeleteImage: (speciesId: string, filename: string) => Promise<{ ok: boolean; error?: string }>;
      sessionDeleteAllImages: (speciesId: string) => Promise<{ ok: boolean; error?: string }>;
      schemaListTemplates: () => Promise<{
        ok: boolean;
        templates: ReusableSchemaTemplate[];
        error?: string;
      }>;
      schemaSaveCustomTemplate: (template: {
        name: string;
        description: string;
        landmarks: import("./Image").LandmarkDefinition[];
        orientationPolicy?: import("./Image").OrientationPolicy;
        sourcePresetId?: string;
      }) => Promise<{
        ok: boolean;
        template?: ReusableSchemaTemplate;
        error?: string;
      }>;
      schemaUpdateCustomTemplate: (templateId: string, updates: {
        name: string;
        description: string;
        landmarks: import("./Image").LandmarkDefinition[];
        orientationPolicy?: import("./Image").OrientationPolicy;
        sourcePresetId?: string;
      }) => Promise<{
        ok: boolean;
        template?: ReusableSchemaTemplate;
        error?: string;
      }>;
      // SuperAnnotator pipeline
      superAnnotate: (imagePath: string, className: string, modelTag?: string, options?: SuperAnnotateOptions, speciesId?: string) => Promise<SuperAnnotateResult>;
      checkSuperAnnotator: () => Promise<CheckSuperAnnotatorResult>;
      resegmentBox: (
        imagePath: string,
        boxXyxy: [number, number, number, number],
        iterative?: boolean
      ) => Promise<{
        ok: boolean;
        maskOutline?: [number, number][];
        obbCorners?: [number, number][];
        angle?: number;
        boxXyxy?: [number, number, number, number];
        score?: number;
        error?: string;
      }>;
      trainObbDetector: (speciesId: string, options?: {
        epochs?: number;
        batch?: number;
        modelTier?: import("./Image").ObbModelTier;
        imgsz?: import("./Image").ObbImageSize;
        iou?: number;
        cls?: number;
        box?: number;
        samEnabled?: boolean;
      }) => Promise<{
        ok: boolean;
        modelPath?: string;
        map50?: number | null;
        warnings?: string[];
        error?: string;
      }>;
      onSuperAnnotateProgress: (callback: (data: SuperAnnotateProgress) => void) => () => void;
      onPredictProgress: (callback: (data: {
        percent: number;
        stage: string;
        currentIndex?: number;
        total?: number;
        imagePath?: string;
      }) => void) => () => void;
      onTrainProgress: (callback: (data: TrainProgressEvent) => void) => () => void;
      onObbTrainProgress: (callback: (data: import("./Image").ObbTrainProgressEvent) => void) => () => void;
      onSegmentSaveStatus: (callback: (data: {
        speciesId: string;
        filename: string;
        state: SegmentSaveStatus["state"];
        signature?: string;
        updatedAt: string;
        reason?: string;
        expectedCount?: number;
        savedCount?: number;
        details?: import("./Image").FinalizeFailureDetail[];
      }) => void) => () => void;
      sessionListInferenceSessions: () => Promise<{
        ok: boolean;
        sessions: InferenceSessionSummary[];
        error?: string;
      }>;
      sessionDeleteSchemaSession: (speciesId: string) => Promise<{
        ok: boolean;
        deleted?: boolean;
        error?: string;
      }>;
      sessionCreateInferenceSession: (
        speciesId: string,
        displayName?: string
      ) => Promise<{
        ok: boolean;
        inferenceSessionId?: string;
        manifest?: InferenceSessionManifest;
        error?: string;
      }>;
      sessionGetInferenceSession: (speciesId: string) => Promise<{
        ok: boolean;
        exists?: boolean;
        inferenceSessionId?: string;
        manifest?: InferenceSessionManifest;
        migratedFrom?: string;
        error?: string;
      }>;
      sessionUpdateInferenceSessionPreferences: (
        speciesId: string,
        inferenceSessionId?: string,
        options?: {
          displayName?: string;
          preferences?: {
            lastUsedLandmarkModelKey?: string;
            lastUsedPredictorType?: "dlib" | "cnn" | "yolo_pose";
            detectionModelKey?: string;
            detectionModelName?: string;
          };
        }
      ) => Promise<{
        ok: boolean;
        inferenceSessionId?: string;
        manifest?: InferenceSessionManifest;
        error?: string;
      }>;
      sessionCommitInferenceReview: (
        speciesId: string,
        inferenceSessionId?: string,
        options?: { onlyReviewComplete?: boolean }
      ) => Promise<{
        ok: boolean;
        inferenceSessionId?: string;
        committed?: number;
        skipped?: number;
        failed?: number;
        failures?: { filename: string; error: string }[];
        error?: string;
      }>;
      sessionSaveInferenceReviewDraft: (
        speciesId: string,
        inferenceSessionId?: string,
        imagePath: string,
        specimens: InferenceReviewDraftSpecimen[],
        options?: {
          filename?: string;
          edited?: boolean;
          saved?: boolean;
          reviewComplete?: boolean;
          committedAt?: string | null;
          landmarkModelKey?: string | null;
          landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose" | null;
          boxSignature?: string | null;
          inferenceSignature?: string | null;
          clear?: boolean;
        }
      ) => Promise<{ ok: boolean; error?: string }>;
      sessionLoadInferenceReviewDrafts: (
        speciesId: string,
        inferenceSessionId?: string
      ) => Promise<{ ok: boolean; drafts?: InferenceReviewDraftItem[]; error?: string }>;
      sessionSaveInferenceImagePaths: (
        speciesId: string,
        inferenceSessionId: string,
        imagePaths: { path: string; name: string }[]
      ) => Promise<{ ok: boolean; error?: string }>;
      sessionLoadInferenceImagePaths: (
        speciesId: string,
        inferenceSessionId: string
      ) => Promise<{ ok: boolean; images?: { path: string; name: string; data: string; mimeType: string }[] }>;
      /** Lightweight hardware probe — called once at startup to populate Redux hardwareSlice */
      probeHardware: () => Promise<HardwareCapabilities>;
    };
  }
}

interface HardwareCapabilities {
  /** Active compute device detected at startup */
  device: "cpu" | "mps" | "cuda";
  /** Human-readable GPU name, or null for CPU-only */
  gpuName: string | null;
  /** Total system RAM in GB, or null if detection failed */
  ramGb: number | null;
  runtimeState?: "not_started" | "checking" | "not_initialized" | "initializing" | "ready" | "failed";
  statusSource?: "local_estimate" | "python_probe" | "python_check";
  pythonPath?: string;
  usingRepoVenv?: boolean;
}
