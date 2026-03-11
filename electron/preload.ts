import { contextBridge, ipcRenderer } from 'electron'
import { AnnotatedImage, GeometryMappingConfig, OrientationPolicy } from '../src/types/Image';

// --------- Expose some API to the Renderer process ---------
contextBridge.exposeInMainWorld('ipcRenderer', withPrototype(ipcRenderer))
window.ipcRenderer = ipcRenderer;

// `exposeInMainWorld` can't detect attributes and methods of `prototype`, manually patching it.
function withPrototype(obj: Record<string, any>) {
  const protos = Object.getPrototypeOf(obj)

  for (const [key, value] of Object.entries(protos)) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) continue

    if (typeof value === 'function') {
      // Some native APIs, like `NodeJS.EventEmitter['on']`, don't work in the Renderer process. Wrapping them into a function.
      obj[key] = function (...args: any) {
        return value.call(obj, ...args)
      }
    } else {
      obj[key] = value
    }
  }
  return obj
}

interface TrainOptions {
  testSplit?: number;
  seed?: number;
  customOptions?: Record<string, number | boolean>;
  speciesId?: string;
  useImportedXml?: boolean;
  predictorType?: "dlib" | "cnn";
  cnnVariant?: string;
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
      orientation?: "left" | "right";
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

interface DetectionOptions {
  speciesId?: string;
}

contextBridge.exposeInMainWorld("api", {
  saveLabels: (data: AnnotatedImage[]) => ipcRenderer.invoke("ml:save-labels", data),
  trainModel: (modelName: string, options?: TrainOptions) =>
    ipcRenderer.invoke("ml:train", modelName, options),
  getCnnVariants: () => ipcRenderer.invoke("ml:get-cnn-variants"),
  trainingPreflight: (args: {
    speciesId?: string;
    modelName: string;
    useImportedXml?: boolean;
    workspaceImages?: number;
    importedImagesHint?: number;
  }) => ipcRenderer.invoke("ml:training-preflight", args),
  predictImage: (
    imagePath: string,
    tag: string,
    speciesId?: string,
    options?: PredictOptions
  ) => ipcRenderer.invoke("ml:predict", imagePath, tag, speciesId, options),
  predictImagesBatch: (args: PredictBatchArgs) =>
    ipcRenderer.invoke("ml:predict-batch", args),
  checkModelCompatibility: (args: {
    speciesId?: string;
    modelName: string;
    predictorType?: "dlib" | "cnn";
    includeRuntime?: boolean;
  }) => ipcRenderer.invoke("ml:check-model-compatibility", args),
  selectImageFolder: () => ipcRenderer.invoke("select-image-folder"),
  getProjectRoot: () => ipcRenderer.invoke("ml:get-project-root"),
  selectProjectRoot: () => ipcRenderer.invoke("ml:select-project-root"),
  listModels: (args?: string | {
    speciesId?: string;
    activeOnly?: boolean;
    includeDeprecated?: boolean;
  }) => ipcRenderer.invoke("ml:list-models", args),
  deleteModel: (
    modelName: string,
    speciesId?: string,
    predictorType?: "dlib" | "cnn" | "yolo_pose"
  ) => ipcRenderer.invoke("ml:delete-model", modelName, speciesId, predictorType),
  renameModel: (
    oldName: string,
    newName: string,
    speciesId?: string,
    predictorType?: "dlib" | "cnn" | "yolo_pose"
  ) => ipcRenderer.invoke("ml:rename-model", oldName, newName, speciesId, predictorType),
  selectImages: () => ipcRenderer.invoke("select-images"),
  selectFolderPath: () => ipcRenderer.invoke("select-folder-path"),
  selectAnnotationFile: () => ipcRenderer.invoke("select-annotation-file"),
  loadAnnotatedFolder: (args: {
    imageFolderPath: string;
    annotationFilePath: string;
    speciesId: string;
    geometryConfig?: GeometryMappingConfig;
    useSam2BoxDerivation?: boolean;
  }) => ipcRenderer.invoke("ml:load-annotated-folder", args),
  importPreAnnotatedDataset: (args?: {
    speciesId?: string;
    geometryConfig?: GeometryMappingConfig;
  }) => ipcRenderer.invoke("ml:import-preannotated-dataset", args),
  // Multi-specimen detection
  detectSpecimens: (imagePath: string, options?: DetectionOptions) =>
    ipcRenderer.invoke("ml:detect-specimens", imagePath, options),
  // Session management
  sessionCreate: (
    speciesId: string,
    name: string,
    landmarkTemplate: any[],
    orientationPolicy?: OrientationPolicy
  ) => ipcRenderer.invoke("session:create", { speciesId, name, landmarkTemplate, orientationPolicy }),
  sessionUpdateOrientationPolicy: (speciesId: string, orientationPolicy: OrientationPolicy) =>
    ipcRenderer.invoke("session:update-orientation-policy", { speciesId, orientationPolicy }),
  sessionUpdateAugmentation: (speciesId: string, augmentationPolicy: Record<string, unknown>) =>
    ipcRenderer.invoke("session:update-augmentation", { speciesId, augmentationPolicy }),
  sessionSaveImage: (speciesId: string, imageData: string, filename: string, mimeType: string) =>
    ipcRenderer.invoke("session:save-image", { speciesId, imageData, filename, mimeType }),
  sessionSaveAnnotations: (speciesId: string, filename: string, boxes: any[]) =>
    ipcRenderer.invoke("session:save-annotations", { speciesId, filename, boxes }),
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
  ) => ipcRenderer.invoke("session:finalize-accepted-boxes", { speciesId, filename, boxes, imagePath }),
  sessionUnfinalizeImage: (
    speciesId: string,
    filename: string,
    imagePath?: string
  ) => ipcRenderer.invoke("session:unfinalize-image", { speciesId, filename, imagePath }),
  sessionUnfinalizeImages: (
    speciesId: string,
    filenames?: string[]
  ) => ipcRenderer.invoke("session:unfinalize-images", { speciesId, filenames }),
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
  ) => ipcRenderer.invoke("session:add-rejected-detection", { speciesId, filename, rejectedDetection }),
  sessionLoad: (speciesId: string) => ipcRenderer.invoke("session:load", { speciesId }),
  sessionLoadAnnotation: (speciesId: string, filename: string) =>
    ipcRenderer.invoke("session:load-annotation", { speciesId, filename }),
  sessionGetSegmentSaveStatus: (speciesId: string, filename: string) =>
    ipcRenderer.invoke("session:get-segment-save-status", { speciesId, filename }),
  sessionList: () => ipcRenderer.invoke("session:list"),
  sessionDeleteImage: (speciesId: string, filename: string) =>
    ipcRenderer.invoke("session:delete-image", { speciesId, filename }),
  sessionDeleteAllImages: (speciesId: string) => ipcRenderer.invoke("session:delete-all-images", { speciesId }),
  // SuperAnnotator pipeline
  superAnnotate: (
    imagePath: string,
    className: string,
    modelTag?: string,
    options?: {
      samEnabled?: boolean;
      maxObjects?: number;
      detectionMode?: string;
      detectionPreset?: string;
      pcaMode?: "off" | "on" | "auto";
      useOrientationHint?: boolean;
    },
    speciesId?: string
  ) => ipcRenderer.invoke("ml:super-annotate", { imagePath, className, modelTag, options, speciesId }),
  checkSuperAnnotator: () => ipcRenderer.invoke("ml:check-super-annotator"),
  resegmentBox: (
    imagePath: string,
    boxXyxy: [number, number, number, number],
    iterative?: boolean
  ) => ipcRenderer.invoke("ml:resegment-box", { imagePath, boxXyxy, iterative }),
  trainObbDetector: (
    speciesId: string,
    options?: { epochs?: number; modelTier?: "nano" | "small"; iou?: number; cls?: number; box?: number }
  ) =>
    ipcRenderer.invoke("ml:train-obb-detector", speciesId, options),
  onSuperAnnotateProgress: (callback: (data: any) => void) => {
    ipcRenderer.on("ml:super-annotate-progress", (_event: any, data: any) => callback(data));
    return () => { ipcRenderer.removeAllListeners("ml:super-annotate-progress"); };
  },
  onPredictProgress: (callback: (data: {
    percent: number;
    stage: string;
    currentIndex?: number;
    total?: number;
    imagePath?: string;
  }) => void) => {
    ipcRenderer.on("ml:predict-progress", (_event: any, data: any) => callback(data));
    return () => { ipcRenderer.removeAllListeners("ml:predict-progress"); };
  },
  onTrainProgress: (callback: (data: {
    percent: number;
    stage: string;
    message: string;
    predictorType: "dlib" | "cnn";
    modelName: string;
    details?: Record<string, unknown>;
  }) => void) => {
    const handler = (_event: any, data: any) => callback(data);
    ipcRenderer.on("ml:train-progress", handler);
    return () => { ipcRenderer.removeListener("ml:train-progress", handler); };
  },
  onSegmentSaveStatus: (callback: (data: {
    speciesId: string;
    filename: string;
    state: "idle" | "queued" | "running" | "saved" | "skipped" | "failed";
    signature?: string;
    updatedAt: string;
    reason?: string;
    expectedCount?: number;
    savedCount?: number;
  }) => void) => {
    const handler = (_event: any, data: any) => callback(data);
    ipcRenderer.on("session:segment-save-status", handler);
    return () => { ipcRenderer.removeListener("session:segment-save-status", handler); };
  },
  sessionListInferenceSessions: () => ipcRenderer.invoke("session:list-inference-sessions"),
  sessionDeleteSchemaSession: (speciesId: string) =>
    ipcRenderer.invoke("session:delete-schema-session", { speciesId }),
  sessionCreateInferenceSession: (speciesId: string, displayName?: string) =>
    ipcRenderer.invoke("session:create-inference-session", { speciesId, displayName }),
  sessionGetInferenceSession: (speciesId: string) =>
    ipcRenderer.invoke("session:get-inference-session", { speciesId }),
  sessionUpdateInferenceSessionPreferences: (
    speciesId: string,
    inferenceSessionId: string | undefined,
    options?: {
      displayName?: string;
      preferences?: {
        lastUsedLandmarkModelKey?: string;
        lastUsedPredictorType?: "dlib" | "cnn" | "yolo_pose";
        detectionModelKey?: string;
        detectionModelName?: string;
      };
    }
  ) =>
    ipcRenderer.invoke("session:update-inference-session-preferences", {
      speciesId,
      inferenceSessionId,
      displayName: options?.displayName,
      preferences: options?.preferences,
    }),
  sessionCommitInferenceReview: (
    speciesId: string,
    inferenceSessionId?: string,
    options?: { onlyReviewComplete?: boolean }
  ) =>
    ipcRenderer.invoke("session:commit-inference-review", {
      speciesId,
      inferenceSessionId,
      onlyReviewComplete: options?.onlyReviewComplete,
    }),
  sessionSaveInferenceReviewDraft: (
    speciesId: string,
    inferenceSessionId: string | undefined,
    imagePath: string,
    specimens: {
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
  ) =>
    ipcRenderer.invoke("session:save-inference-review-draft", {
      speciesId,
      inferenceSessionId,
      imagePath,
      specimens,
      filename: options?.filename,
      edited: options?.edited,
      saved: options?.saved,
      reviewComplete: options?.reviewComplete,
      committedAt: options?.committedAt,
      landmarkModelKey: options?.landmarkModelKey,
      landmarkPredictorType: options?.landmarkPredictorType,
      boxSignature: options?.boxSignature,
      inferenceSignature: options?.inferenceSignature,
      clear: options?.clear,
    }),
  sessionLoadInferenceReviewDrafts: (speciesId: string, inferenceSessionId?: string) =>
    ipcRenderer.invoke("session:load-inference-review-drafts", { speciesId, inferenceSessionId }),
  sessionSaveInferenceImagePaths: (
    speciesId: string,
    inferenceSessionId: string,
    imagePaths: { path: string; name: string }[]
  ) =>
    ipcRenderer.invoke("session:save-inference-image-paths", { speciesId, inferenceSessionId, imagePaths }),
  sessionLoadInferenceImagePaths: (speciesId: string, inferenceSessionId: string) =>
    ipcRenderer.invoke("session:load-inference-image-paths", { speciesId, inferenceSessionId }),
  // Hardware capability probe - called once at app startup
  probeHardware: () => ipcRenderer.invoke("system:probe-hardware"),
});

// --------- Preload scripts loading ---------
function domReady(condition: DocumentReadyState[] = ['complete', 'interactive']) {
  return new Promise(resolve => {
    if (condition.includes(document.readyState)) {
      resolve(true)
    } else {
      document.addEventListener('readystatechange', () => {
        if (condition.includes(document.readyState)) {
          resolve(true)
        }
      })
    }
  })
}

const safeDOM = {
  append(parent: HTMLElement, child: HTMLElement) {
    if (!Array.from(parent.children).find(e => e === child)) {
      parent.appendChild(child)
    }
  },
  remove(parent: HTMLElement, child: HTMLElement) {
    if (Array.from(parent.children).find(e => e === child)) {
      parent.removeChild(child)
    }
  },
}

/**
 * https://tobiasahlin.com/spinkit
 * https://connoratherton.com/loaders
 * https://projects.lukehaas.me/css-loaders
 * https://matejkustec.github.io/SpinThatShit
 */
function useLoading() {
  const className = `loaders-css__square-spin`
  const styleContent = `
@keyframes square-spin {
  25% { transform: perspective(100px) rotateX(180deg) rotateY(0); }
  50% { transform: perspective(100px) rotateX(180deg) rotateY(180deg); }
  75% { transform: perspective(100px) rotateX(0) rotateY(180deg); }
  100% { transform: perspective(100px) rotateX(0) rotateY(0); }
}
.${className} > div {
  animation-fill-mode: both;
  width: 50px;
  height: 50px;
  background: #fff;
  animation: square-spin 3s 0s cubic-bezier(0.09, 0.57, 0.49, 0.9) infinite;
}
.app-loading-wrap {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #282c34;
  z-index: 9;
}
    `
  const oStyle = document.createElement('style')
  const oDiv = document.createElement('div')

  oStyle.id = 'app-loading-style'
  oStyle.innerHTML = styleContent
  oDiv.className = 'app-loading-wrap'
  oDiv.innerHTML = `<div class="${className}"><div></div></div>`

  return {
    appendLoading() {
      safeDOM.append(document.head, oStyle)
      safeDOM.append(document.body, oDiv)
    },
    removeLoading() {
      safeDOM.remove(document.head, oStyle)
      safeDOM.remove(document.body, oDiv)
    },
  }
}

// ----------------------------------------------------------------------

const { appendLoading, removeLoading } = useLoading()
domReady().then(appendLoading)

window.onmessage = ev => {
  ev.data.payload === 'removeLoading' && removeLoading()
}

setTimeout(removeLoading, 4999)
