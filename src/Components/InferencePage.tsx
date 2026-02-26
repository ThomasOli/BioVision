import React, { useEffect, useState, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  Microscope,
  Upload,
  Play,
  Save,
  Download,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Loader2,
  ImageIcon,
  X,
  FileJson,
  FileSpreadsheet,
  Square,
} from "lucide-react";
import { toast } from "sonner";
import { useSelector } from "react-redux";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { Label } from "@/Components/ui/label";
import { Switch } from "@/Components/ui/switch";
import { ScrollArea } from "@/Components/ui/scroll-area";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/Components/ui/popover";
import { staggerContainer, staggerItem, buttonHover, buttonTap, cardHover } from "@/lib/animations";
import { TrainedModel, AppView } from "@/types/Image";
import type { RootState } from "@/state/store";

const STAGE_LABELS: Record<string, string> = {
  loading_model: "Loading model...",
  detecting: "Detecting specimen...",
  predicting: "Predicting landmarks...",
  mapping: "Applying transforms...",
  done: "Complete",
};

const SIDEBAR_MIN_WIDTH = 240;
const SIDEBAR_MAX_WIDTH = 560;
const SIDEBAR_DEFAULT_WIDTH = 288;
const SIDEBAR_WIDTH_STORAGE_KEY = "biovision.inference.sidebarWidth";
const AUTO_RETRAIN_TOGGLE_STORAGE_KEY = "biovision.inference.autoRetrainEnabled";
const AUTO_RETRAIN_LAST_AT_STORAGE_KEY = "biovision.inference.lastAutoRetrainAt";
const AUTO_RETRAIN_MIN_QUEUE = 10;
const AUTO_RETRAIN_COOLDOWN_MS = 20 * 60 * 1000;

interface InferencePageProps {
  onNavigate: (view: AppView) => void;
  initialModel?: string;
}

interface PredictionLandmark {
  id: number;
  x: number;
  y: number;
}

interface DetectedBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
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
}

interface ImageDimensions {
  width: number;
  height: number;
}

interface PredictedSpecimen {
  box: DetectedBox;
  landmarks: PredictionLandmark[];
  num_landmarks: number;
  mask_outline?: [number, number][];
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
}

interface RejectedDetection {
  left: number;
  top: number;
  width: number;
  height: number;
  confidence?: number;
  className?: string;
  detectionMethod?: string;
}

interface InferenceResult {
  image: string;
  landmarks?: PredictionLandmark[];
  detected_box?: DetectedBox;
  image_dimensions?: ImageDimensions;
  specimens?: PredictedSpecimen[];
  num_specimens?: number;
}

interface InferenceImage {
  path: string;
  name: string;
  url: string;
  results?: InferenceResult;
  error?: string;
}

interface InferenceReviewDraft {
  key: string;
  imagePath: string;
  filename: string;
  specimens: {
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
  }[];
  edited: boolean;
  saved: boolean;
  updatedAt: string;
}

interface LocalInferenceSessionManifest {
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

export const InferencePage: React.FC<InferencePageProps> = ({
  onNavigate,
  initialModel,
}) => {
  const activeSpeciesId = useSelector((state: RootState) => state.species.activeSpeciesId);
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [selectedModelKey, setSelectedModelKey] = useState<string>("");
  const [images, setImages] = useState<InferenceImage[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [isSavingCorrections, setIsSavingCorrections] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [showBoundingBox, setShowBoundingBox] = useState(true);
  const [showMaskOverlay, setShowMaskOverlay] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    if (typeof window === "undefined") return SIDEBAR_DEFAULT_WIDTH;
    const raw = window.localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY);
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return SIDEBAR_DEFAULT_WIDTH;
    return Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, parsed));
  });
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const sidebarResizeRef = useRef<{ startX: number; startWidth: number } | null>(null);

  // Inference progress
  const [inferProgress, setInferProgress] = useState<{ percent: number; stage: string } | null>(null);

  // HITL correction state
  const [correctedSpecimensMap, setCorrectedSpecimensMap] = useState<Map<number, PredictedSpecimen[]>>(new Map());
  const [editedImageIndices, setEditedImageIndices] = useState<Set<number>>(new Set());
  const [savedImageIndices, setSavedImageIndices] = useState<Set<number>>(new Set());
  const [reviewFinalizedImageIndices, setReviewFinalizedImageIndices] = useState<Set<number>>(new Set());
  const [queuedImageIndices, setQueuedImageIndices] = useState<Set<number>>(new Set());
  const [selectedSpecimenIndex, setSelectedSpecimenIndex] = useState<number | null>(null);
  const draggingRef = useRef<{ specIdx: number; lmIdx: number } | null>(null);
  const liveSpecimensRef = useRef<PredictedSpecimen[]>([]); // mutable during drag
  const [isQueueingRetrain, setIsQueueingRetrain] = useState(false);
  const [retrainQueueCount, setRetrainQueueCount] = useState(0);
  const [inferenceSessionId, setInferenceSessionId] = useState<string>("");
  const [inferenceSessionManifest, setInferenceSessionManifest] = useState<LocalInferenceSessionManifest | null>(null);
  const [autoRetrainEnabled, setAutoRetrainEnabled] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem(AUTO_RETRAIN_TOGGLE_STORAGE_KEY) === "1";
  });
  const [lastAutoRetrainAt, setLastAutoRetrainAt] = useState<number>(() => {
    if (typeof window === "undefined") return 0;
    const parsed = Number(window.localStorage.getItem(AUTO_RETRAIN_LAST_AT_STORAGE_KEY));
    return Number.isFinite(parsed) ? parsed : 0;
  });
  const [isAutoRetraining, setIsAutoRetraining] = useState(false);
  const [autoRetrainProgress, setAutoRetrainProgress] = useState<{
    percent: number;
    stage: string;
    message: string;
  } | null>(null);
  const [obbDetectorReady, setObbDetectorReady] = useState<boolean | null>(null);

  const markImageEdited = useCallback((index: number) => {
    setEditedImageIndices((prev) => {
      const next = new Set(prev);
      next.add(index);
      return next;
    });
    setSavedImageIndices((prev) => {
      const next = new Set(prev);
      next.delete(index);
      return next;
    });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(
      AUTO_RETRAIN_TOGGLE_STORAGE_KEY,
      autoRetrainEnabled ? "1" : "0"
    );
  }, [autoRetrainEnabled]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(
      AUTO_RETRAIN_LAST_AT_STORAGE_KEY,
      String(lastAutoRetrainAt || 0)
    );
  }, [lastAutoRetrainAt]);

  const shiftIndicesInSet = useCallback((source: Set<number>, removedIndex: number): Set<number> => {
    const next = new Set<number>();
    source.forEach((idx) => {
      if (idx === removedIndex) return;
      next.add(idx > removedIndex ? idx - 1 : idx);
    });
    return next;
  }, []);

  const shiftIndicesInMap = useCallback(
    (source: Map<number, PredictedSpecimen[]>, removedIndex: number): Map<number, PredictedSpecimen[]> => {
      const next = new Map<number, PredictedSpecimen[]>();
      source.forEach((value, key) => {
        if (key === removedIndex) return;
        next.set(key > removedIndex ? key - 1 : key, value);
      });
      return next;
    },
    []
  );

  const modelToKey = useCallback((model: TrainedModel): string => {
    return `${model.name}::${model.predictorType ?? "dlib"}`;
  }, []);

  const getSelectedModel = useCallback((): TrainedModel | undefined => {
    return models.find((m) => modelToKey(m) === selectedModelKey);
  }, [models, modelToKey, selectedModelKey]);

  const predictorLabel = useCallback((predictorType?: TrainedModel["predictorType"]) => {
    if (predictorType === "cnn") return "CNN";
    return "dlib";
  }, []);

  const resolveInitialModelKey = useCallback((available: TrainedModel[]): string => {
    if (!available.length) return "";
    if (!initialModel) return modelToKey(available[0]);
    const exact = available.find((m) => modelToKey(m) === initialModel);
    if (exact) return modelToKey(exact);
    const byName = available.filter((m) => m.name === initialModel);
    if (byName.length > 0) {
      const preferred = byName.find((m) => m.predictorType === "dlib") ?? byName[0];
      return modelToKey(preferred);
    }
    return modelToKey(available[0]);
  }, [initialModel, modelToKey]);

  // Bounding-box drag/resize state
  type BoxDragMode = "move" | "resize-tl" | "resize-tr" | "resize-bl" | "resize-br";
  const boxDraggingRef = useRef<{
    specIdx: number;
    mode: BoxDragMode;
    startMouse: { x: number; y: number };
    startBox: { left: number; top: number; width: number; height: number };
  } | null>(null);
  const [isDrawBoxMode, setIsDrawBoxMode] = useState(false);
  const drawBoxRef = useRef<{
    start: { x: number; y: number };
    current: { x: number; y: number };
  } | null>(null);

  const clampSidebarWidth = useCallback((value: number) => {
    return Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, value));
  }, []);

  const handleSidebarResizeStart = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    sidebarResizeRef.current = {
      startX: e.clientX,
      startWidth: sidebarWidth,
    };
    setIsResizingSidebar(true);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, [sidebarWidth]);

  const getSpecimensFromResult = (result?: InferenceResult): PredictedSpecimen[] => {
    if (!result) return [];
    if (Array.isArray(result.specimens) && result.specimens.length > 0) {
      return result.specimens;
    }
    if (Array.isArray(result.landmarks) && result.landmarks.length > 0) {
      const syntheticBox: DetectedBox = result.detected_box || {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
        width: 0,
        height: 0,
      };
      return [{
        box: syntheticBox,
        landmarks: result.landmarks,
        num_landmarks: result.landmarks.length,
      }];
    }
    return [];
  };

  const getTotalLandmarks = (result?: InferenceResult): number => {
    const specimens = getSpecimensFromResult(result);
    return specimens.reduce((sum, specimen) => sum + specimen.landmarks.length, 0);
  };

  const getSpecimensForImageIndex = useCallback((index: number): PredictedSpecimen[] => {
    const image = images[index];
    if (!image?.results) return [];
    return correctedSpecimensMap.get(index) ?? getSpecimensFromResult(image.results);
  }, [images, correctedSpecimensMap]);

  const intersectionOverUnion = useCallback(
    (
      a: { left: number; top: number; width: number; height: number },
      b: { left: number; top: number; width: number; height: number }
    ): number => {
      const ax1 = a.left;
      const ay1 = a.top;
      const ax2 = a.left + a.width;
      const ay2 = a.top + a.height;
      const bx1 = b.left;
      const by1 = b.top;
      const bx2 = b.left + b.width;
      const by2 = b.top + b.height;

      const ix1 = Math.max(ax1, bx1);
      const iy1 = Math.max(ay1, by1);
      const ix2 = Math.min(ax2, bx2);
      const iy2 = Math.min(ay2, by2);
      const iw = Math.max(0, ix2 - ix1);
      const ih = Math.max(0, iy2 - iy1);
      const inter = iw * ih;
      if (inter <= 0) return 0;

      const aArea = Math.max(0, a.width) * Math.max(0, a.height);
      const bArea = Math.max(0, b.width) * Math.max(0, b.height);
      const union = aArea + bArea - inter;
      if (union <= 0) return 0;
      return inter / union;
    },
    []
  );

  const normalizePathForMatch = useCallback((value: string): string => {
    return (value || "").replace(/\\/g, "/").toLowerCase();
  }, []);

  const toDraftSpecimens = useCallback((specimens: PredictedSpecimen[]) => {
    return (specimens || [])
      .filter((s) => s?.box && s.box.width > 0 && s.box.height > 0)
      .map((s) => ({
        box: {
          left: Math.round(s.box.left),
          top: Math.round(s.box.top),
          width: Math.round(s.box.width),
          height: Math.round(s.box.height),
          confidence: Number.isFinite(Number(s.box.confidence))
            ? Number(s.box.confidence)
            : undefined,
          class_id: Number.isFinite(Number(s.box.class_id))
            ? Number(s.box.class_id)
            : undefined,
          class_name: s.box.class_name,
          orientation_override: s.box.orientation_override,
          orientation_hint: s.box.orientation_hint,
        },
        landmarks: (s.landmarks || []).map((lm) => ({
          id: Number(lm.id),
          x: Math.round(lm.x),
          y: Math.round(lm.y),
        })),
      }));
  }, []);

  const mergeDraftLandmarksWithInference = useCallback(
    (
      draftSpecimens: PredictedSpecimen[],
      inferenceSpecimens: PredictedSpecimen[]
    ): PredictedSpecimen[] => {
      if (!draftSpecimens.length) return draftSpecimens;
      if (!inferenceSpecimens.length) return draftSpecimens;

      const usedInferenceIdx = new Set<number>();
      return draftSpecimens.map((draft) => {
        const draftLandmarks = Array.isArray(draft.landmarks) ? draft.landmarks : [];
        let bestIdx = -1;
        let bestIou = 0;
        inferenceSpecimens.forEach((candidate, idx) => {
          if (!candidate?.box) return;
          const iou = intersectionOverUnion(
            {
              left: draft.box.left,
              top: draft.box.top,
              width: draft.box.width,
              height: draft.box.height,
            },
            {
              left: candidate.box.left,
              top: candidate.box.top,
              width: candidate.box.width,
              height: candidate.box.height,
            }
          );
          if (iou > bestIou && (!usedInferenceIdx.has(idx) || iou >= 0.9)) {
            bestIou = iou;
            bestIdx = idx;
          }
        });

        if (bestIdx >= 0 && bestIou >= 0.15) {
          usedInferenceIdx.add(bestIdx);
          const matched = inferenceSpecimens[bestIdx];
          const mergedLandmarks =
            draftLandmarks.length > 0
              ? draftLandmarks
              : (matched.landmarks || []).map((lm) => ({
                  id: Number(lm.id),
                  x: Number(lm.x),
                  y: Number(lm.y),
                }));
          const mergedBox: DetectedBox = {
            ...matched.box,
            left: draft.box.left,
            top: draft.box.top,
            width: draft.box.width,
            height: draft.box.height,
            right: draft.box.left + draft.box.width,
            bottom: draft.box.top + draft.box.height,
            confidence: Number.isFinite(Number(draft.box.confidence))
              ? Number(draft.box.confidence)
              : matched.box.confidence,
            class_id: Number.isFinite(Number(draft.box.class_id))
              ? Number(draft.box.class_id)
              : matched.box.class_id,
            class_name: draft.box.class_name || matched.box.class_name,
            orientation_override: draft.box.orientation_override || matched.box.orientation_override,
            orientation_hint: draft.box.orientation_hint || matched.box.orientation_hint,
          };
          return {
            ...draft,
            box: mergedBox,
            landmarks: mergedLandmarks,
            num_landmarks: mergedLandmarks.length,
            mask_outline: Array.isArray(matched.mask_outline) ? matched.mask_outline : draft.mask_outline,
            inference_metadata: matched.inference_metadata ?? draft.inference_metadata,
          };
        }

        return draft;
      });
    },
    [intersectionOverUnion]
  );

  const loadPersistedFinalizedFilenames = useCallback(async (): Promise<Set<string>> => {
    if (!activeSpeciesId) return new Set<string>();
    try {
      const res = await window.api.sessionLoad(activeSpeciesId);
      if (!res.ok || !Array.isArray(res.images)) return new Set<string>();
      return new Set(
        res.images
          .filter((img) => Boolean(img.finalized))
          .map((img) => (img.filename || "").toLowerCase())
      );
    } catch {
      return new Set<string>();
    }
  }, [activeSpeciesId]);

  const loadPersistedQueuedSummary = useCallback(async (): Promise<{ filenames: Set<string>; count: number }> => {
    if (!activeSpeciesId || !inferenceSessionId) {
      setRetrainQueueCount(0);
      return { filenames: new Set<string>(), count: 0 };
    }
    try {
      const res = await window.api.sessionGetRetrainQueue(activeSpeciesId, inferenceSessionId);
      if (!res.ok || !Array.isArray(res.items)) {
        setRetrainQueueCount(0);
        return { filenames: new Set<string>(), count: 0 };
      }
      const filenames = new Set(
        res.items.map((item) => (item.filename || "").toLowerCase())
      );
      const count = Number.isFinite(Number(res.count)) ? Number(res.count) : res.items.length;
      setRetrainQueueCount(count);
      return { filenames, count };
    } catch {
      setRetrainQueueCount(0);
      return { filenames: new Set<string>(), count: 0 };
    }
  }, [activeSpeciesId, inferenceSessionId]);

  const refreshRetrainQueueSummary = useCallback(async (): Promise<number> => {
    const summary = await loadPersistedQueuedSummary();
    return summary.count;
  }, [loadPersistedQueuedSummary]);

  const hydratePersistedReviewDrafts = useCallback(
    async (targetImages?: InferenceImage[]) => {
      if (!activeSpeciesId || !inferenceSessionId) return;
      const sourceImages = targetImages ?? images;
      if (!sourceImages.length) return;

      const [draftResult, finalizedFilenames, queuedSummary] = await Promise.all([
        window.api.sessionLoadInferenceReviewDrafts(activeSpeciesId, inferenceSessionId),
        loadPersistedFinalizedFilenames(),
        loadPersistedQueuedSummary(),
      ]);
      const queuedFilenames = queuedSummary.filenames;
      const drafts = draftResult.ok && Array.isArray(draftResult.drafts) ? draftResult.drafts : [];

      const byPath = new Map<string, InferenceReviewDraft>();
      const byName = new Map<string, InferenceReviewDraft>();
      drafts.forEach((draft: InferenceReviewDraft) => {
        if (draft.imagePath) {
          byPath.set(normalizePathForMatch(draft.imagePath), draft);
        }
        if (draft.filename) {
          byName.set(draft.filename.toLowerCase(), draft);
        }
      });

      const nextCorrected = new Map<number, PredictedSpecimen[]>();
      const nextEdited = new Set<number>();
      const nextSaved = new Set<number>();
      const nextReviewFinalized = new Set<number>();
      const nextQueued = new Set<number>();

      sourceImages.forEach((img, idx) => {
        const draft =
          byPath.get(normalizePathForMatch(img.path)) ??
          byName.get((img.name || "").toLowerCase());
        const finalizedBySession = finalizedFilenames.has((img.name || "").toLowerCase());
        const queuedBySession = queuedFilenames.has((img.name || "").toLowerCase());

        if (!draft) {
          if (finalizedBySession) {
            nextSaved.add(idx);
          }
          if (queuedBySession) {
            nextQueued.add(idx);
          }
          return;
        }

        const convertedDraftSpecimens: PredictedSpecimen[] = (draft.specimens || []).map((s: InferenceReviewDraft["specimens"][number]) => {
          const left = Number(s.box.left) || 0;
          const top = Number(s.box.top) || 0;
          const width = Number(s.box.width) || 0;
          const height = Number(s.box.height) || 0;
          const landmarks = (s.landmarks || []).map((lm: InferenceReviewDraft["specimens"][number]["landmarks"][number]) => ({
            id: Number(lm.id),
            x: Number(lm.x),
            y: Number(lm.y),
          }));
          return {
            box: {
              left,
              top,
              width,
              height,
              right: left + width,
              bottom: top + height,
              confidence: Number.isFinite(Number(s.box.confidence)) ? Number(s.box.confidence) : undefined,
              class_id: Number.isFinite(Number(s.box.class_id)) ? Number(s.box.class_id) : undefined,
              class_name: s.box.class_name,
              orientation_override: s.box.orientation_override,
              orientation_hint: s.box.orientation_hint,
            },
            landmarks,
            num_landmarks: landmarks.length,
          };
        });

        const inferredSpecimens = getSpecimensFromResult(img.results);
        const mergedSpecimens = mergeDraftLandmarksWithInference(
          convertedDraftSpecimens,
          inferredSpecimens
        );
        nextCorrected.set(idx, mergedSpecimens);
        if (finalizedBySession || draft.saved) nextSaved.add(idx);
        if (draft.saved) nextReviewFinalized.add(idx);
        else if (draft.edited) nextEdited.add(idx);
        if (queuedBySession) nextQueued.add(idx);
      });

      setCorrectedSpecimensMap(nextCorrected);
      setEditedImageIndices(nextEdited);
      setSavedImageIndices(nextSaved);
      setReviewFinalizedImageIndices(nextReviewFinalized);
      setQueuedImageIndices(nextQueued);
    },
    [
      activeSpeciesId,
      inferenceSessionId,
      images,
      loadPersistedFinalizedFilenames,
      loadPersistedQueuedSummary,
      mergeDraftLandmarksWithInference,
      normalizePathForMatch,
    ]
  );

  const persistReviewDraft = useCallback(
    async (
      index: number,
      options?: {
        specimens?: PredictedSpecimen[];
        edited?: boolean;
        saved?: boolean;
        clear?: boolean;
      }
    ) => {
      if (!activeSpeciesId || !inferenceSessionId) return;
      const image = images[index];
      if (!image) return;

      const specimens = toDraftSpecimens(
        options?.specimens ?? getSpecimensForImageIndex(index)
      );

      const response = await window.api.sessionSaveInferenceReviewDraft(
        activeSpeciesId,
        inferenceSessionId,
        image.path,
        specimens,
        {
          filename: image.name,
          edited: options?.edited ?? editedImageIndices.has(index),
          saved: options?.saved ?? savedImageIndices.has(index),
          clear: options?.clear,
        }
      );
      if (!response.ok) {
        console.warn(
          "Failed to persist inference review draft:",
          image.name,
          response.error
        );
      }
    },
    [
      activeSpeciesId,
      inferenceSessionId,
      editedImageIndices,
      getSpecimensForImageIndex,
      images,
      savedImageIndices,
      toDraftSpecimens,
    ]
  );

  const saveCorrectionsForImage = useCallback(async (index: number): Promise<boolean> => {
    if (!activeSpeciesId) {
      toast.error("Select an active session/species before saving corrections.");
      return false;
    }
    const image = images[index];
    if (!image?.results) return false;

    const specimens = getSpecimensForImageIndex(index)
      .filter((s) => s?.box && s.box.width > 0 && s.box.height > 0)
      .map((s) => ({
        box: {
          left: Math.round(s.box.left),
          top: Math.round(s.box.top),
          width: Math.round(s.box.width),
          height: Math.round(s.box.height),
          orientation_override: s.box.orientation_override,
        },
        landmarks: (s.landmarks || []).map((lm) => ({
          id: Number(lm.id),
          x: Math.round(lm.x),
          y: Math.round(lm.y),
        })),
      }));
    const acceptedBoxes = specimens.map((s) => s.box);

    const originalSpecimens = getSpecimensFromResult(image.results).filter(
      (s) => s?.box && s.box.width > 0 && s.box.height > 0
    );
    const originalBoxes = originalSpecimens.map((s) => ({
      left: Math.round(s.box.left),
      top: Math.round(s.box.top),
      width: Math.round(s.box.width),
      height: Math.round(s.box.height),
      confidence: Number.isFinite(Number(s.box.confidence))
        ? Number(s.box.confidence)
        : undefined,
      className: s.box.class_name,
    }));
    const rejectedDetections: RejectedDetection[] = originalBoxes
      .filter((orig) => {
        if (acceptedBoxes.length === 0) return true;
        return !acceptedBoxes.some((acc) => intersectionOverUnion(orig, acc) >= 0.5);
      })
      .map((orig) => ({
        left: orig.left,
        top: orig.top,
        width: orig.width,
        height: orig.height,
        ...(orig.confidence !== undefined ? { confidence: orig.confidence } : {}),
        ...(orig.className ? { className: orig.className } : {}),
        detectionMethod: "inference_deleted_or_rejected",
      }));

    const dims = image.results.image_dimensions;
    const imageWidth = Math.max(1, Math.round(dims?.width ?? 0));
    const imageHeight = Math.max(1, Math.round(dims?.height ?? 0));

    if (specimens.length === 0) {
      const [landmarkClearSave, detectionOnlySave] = await Promise.all([
        window.api.sessionSaveInferenceCorrection(
          activeSpeciesId,
          image.path,
          undefined,
          undefined,
          image.name,
          [],
          rejectedDetections,
          { allowEmpty: true }
        ),
        window.api.sessionSaveDetectionCorrection(
          activeSpeciesId,
          image.path,
          [],
          imageWidth,
          imageHeight,
          image.name
        ),
      ]);
      if (!landmarkClearSave.ok) {
        toast.error(
          landmarkClearSave.error ||
            `Failed to clear inference landmarks for ${image.name}.`
        );
        return false;
      }
      if (!detectionOnlySave.ok) {
        toast.error(
          detectionOnlySave.error ||
            `Failed to save detection-only correction for ${image.name}.`
        );
        return false;
      }
      setSavedImageIndices((prev) => {
        const next = new Set(prev);
        next.add(index);
        return next;
      });
      setReviewFinalizedImageIndices((prev) => {
        const next = new Set(prev);
        next.add(index);
        return next;
      });
      setEditedImageIndices((prev) => {
        const next = new Set(prev);
        next.delete(index);
        return next;
      });
      await persistReviewDraft(index, { edited: false, saved: true, specimens: [] });
      return true;
    }

    const first = specimens[0];

    const [landmarkSave, detectionSave] = await Promise.all([
      window.api.sessionSaveInferenceCorrection(
        activeSpeciesId,
        image.path,
        first.box,
        first.landmarks,
        image.name,
        specimens,
        rejectedDetections
      ),
      window.api.sessionSaveDetectionCorrection(
        activeSpeciesId,
        image.path,
        specimens.map((s) => s.box),
        imageWidth,
        imageHeight,
        image.name
      ),
    ]);

    if (!landmarkSave.ok) {
      toast.error(landmarkSave.error || `Failed to save landmark corrections for ${image.name}.`);
      return false;
    }
    if (!detectionSave.ok) {
      toast.error(detectionSave.error || `Failed to save detection corrections for ${image.name}.`);
      return false;
    }

    setSavedImageIndices((prev) => {
      const next = new Set(prev);
      next.add(index);
      return next;
    });
    setReviewFinalizedImageIndices((prev) => {
      const next = new Set(prev);
      next.add(index);
      return next;
    });
    setEditedImageIndices((prev) => {
      const next = new Set(prev);
      next.delete(index);
      return next;
    });
    await persistReviewDraft(index, { edited: false, saved: true, specimens: getSpecimensForImageIndex(index) });
    return true;
  }, [activeSpeciesId, images, getSpecimensForImageIndex, intersectionOverUnion, persistReviewDraft]);

  const resolveOrientationLabelFromBox = useCallback((box?: DetectedBox): "left" | "right" | "uncertain" => {
    if (!box) return "uncertain";
    if (box.orientation_override === "left" || box.orientation_override === "right") {
      return box.orientation_override;
    }
    if (box.orientation_override === "uncertain") {
      return "uncertain";
    }

    const hintOrientationRaw = box.orientation_hint?.orientation;
    const hintOrientation =
      hintOrientationRaw === "left" || hintOrientationRaw === "right"
        ? hintOrientationRaw
        : null;
    const hintConfidence = Number(box.orientation_hint?.confidence);
    if (
      hintOrientation &&
      (!Number.isFinite(hintConfidence) || hintConfidence >= 0.35)
    ) {
      return hintOrientation;
    }

    const classToken = String(box.class_name || "")
      .trim()
      .toLowerCase()
      .replace(/[-\s]+/g, "_");
    if (
      classToken.endsWith("_left") ||
      classToken === "left" ||
      classToken.includes("_left_")
    ) {
      return "left";
    }
    if (
      classToken.endsWith("_right") ||
      classToken === "right" ||
      classToken.includes("_right_")
    ) {
      return "right";
    }
    return "uncertain";
  }, []);

  const handleSetSpecimenOrientation = useCallback(
    (specimenIndex: number, orientation: "left" | "right" | "uncertain") => {
      const current = getSpecimensForImageIndex(currentIndex);
      if (specimenIndex < 0 || specimenIndex >= current.length) return;
      const updated = JSON.parse(JSON.stringify(current)) as PredictedSpecimen[];
      const target = updated[specimenIndex];
      if (!target?.box) return;

      target.box.orientation_override = orientation;
      if (orientation === "left" || orientation === "right") {
        target.box.orientation_hint = {
          orientation,
          confidence: 1.0,
          source: "user_review",
        };
      } else {
        if (target.box.orientation_hint?.source === "user_review") {
          delete target.box.orientation_hint;
        }
      }

      liveSpecimensRef.current = updated;
      setCorrectedSpecimensMap((prev) => {
        const next = new Map(prev);
        next.set(currentIndex, updated);
        return next;
      });
      markImageEdited(currentIndex);
      void persistReviewDraft(currentIndex, { specimens: updated, edited: true, saved: false });
    },
    [currentIndex, getSpecimensForImageIndex, markImageEdited, persistReviewDraft]
  );

  const queueRetrainForImage = useCallback(async (index: number): Promise<boolean> => {
    if (!activeSpeciesId || !inferenceSessionId) {
      toast.error("Select an active session/species before queueing retrain.");
      return false;
    }
    const image = images[index];
    if (!image) return false;
    const specimens = getSpecimensForImageIndex(index);
    const boxesCount = specimens.filter((s) => s?.box && s.box.width > 0 && s.box.height > 0).length;
    const landmarksCount = specimens.reduce((sum, s) => sum + (s.landmarks?.length || 0), 0);
    const selectedModel = getSelectedModel();
    const result = await window.api.sessionQueueRetrainItem(activeSpeciesId, inferenceSessionId, image.name, {
      imagePath: image.path,
      source: "inference_review",
      boxesCount,
      landmarksCount,
      landmarkModelKey: selectedModelKey,
      landmarkModelName: selectedModel?.name,
      landmarkPredictorType: selectedModel?.predictorType ?? "dlib",
      detectionModelKey: "session_detection_default",
      detectionModelName: "Session Detection Model",
    });
    if (!result.ok) {
      toast.error(result.error || `Failed to queue ${image.name} for retraining.`);
      return false;
    }
    setQueuedImageIndices((prev) => {
      const next = new Set(prev);
      next.add(index);
      return next;
    });
    if (typeof result.queuedCount === "number") {
      setRetrainQueueCount(result.queuedCount);
    } else {
      void refreshRetrainQueueSummary();
    }
    return true;
  }, [
    activeSpeciesId,
    inferenceSessionId,
    getSelectedModel,
    getSpecimensForImageIndex,
    images,
    refreshRetrainQueueSummary,
    selectedModelKey,
  ]);

  const attemptAutoRetrainIfReady = useCallback(async (): Promise<boolean> => {
    if (!autoRetrainEnabled || !activeSpeciesId || isAutoRetraining) return false;

    const selectedModel = models.find((m) => modelToKey(m) === selectedModelKey);
    if (!selectedModel) return false;
    const selectedPredictor = selectedModel.predictorType ?? "dlib";
    if (selectedPredictor === "yolo_pose") {
      return false;
    }

    if (!inferenceSessionId) return false;
    const queueRes = await window.api.sessionGetRetrainQueue(activeSpeciesId, inferenceSessionId);
    if (!queueRes.ok) {
      toast.error(queueRes.error || "Failed to read retrain queue.");
      return false;
    }

    const queuedCount = Number.isFinite(Number(queueRes.count))
      ? Number(queueRes.count)
      : Array.isArray(queueRes.items)
      ? queueRes.items.length
      : 0;
    setRetrainQueueCount(queuedCount);

    if (queuedCount < AUTO_RETRAIN_MIN_QUEUE) {
      return false;
    }

    const now = Date.now();
    if (lastAutoRetrainAt > 0 && now - lastAutoRetrainAt < AUTO_RETRAIN_COOLDOWN_MS) {
      return false;
    }

    setIsAutoRetraining(true);
    setAutoRetrainProgress({
      percent: 0,
      stage: "starting",
      message: `Preparing auto-retrain for ${selectedModel.name}...`,
    });

    const expectedPredictor: "dlib" | "cnn" = selectedPredictor;
    const unsubscribe = window.api.onTrainProgress((data) => {
      if (data.modelName !== selectedModel.name) return;
      if (data.predictorType !== expectedPredictor) return;
      setAutoRetrainProgress({
        percent: Math.max(0, Math.min(100, Math.round(data.percent ?? 0))),
        stage: data.stage,
        message: data.message || data.stage,
      });
    });

    try {
      const result = await window.api.trainModel(selectedModel.name, {
        speciesId: activeSpeciesId,
        predictorType: expectedPredictor,
      });

      if (!result.ok) {
        toast.error(
          result.error ||
            `Auto-retrain failed for ${selectedModel.name}. Corrections remain queued.`
        );
        return false;
      }

      const clearRes = await window.api.sessionClearRetrainQueue(activeSpeciesId, inferenceSessionId);
      if (!clearRes.ok) {
        toast.warning(
          clearRes.error ||
            "Auto-retrain finished, but retrain queue could not be cleared."
        );
      } else {
        setRetrainQueueCount(0);
        setQueuedImageIndices(new Set());
      }

      setLastAutoRetrainAt(now);
      toast.success(
        `Auto-retrained ${selectedModel.name} (${expectedPredictor.toUpperCase()}).`
      );
      return true;
    } finally {
      unsubscribe();
      setIsAutoRetraining(false);
      setAutoRetrainProgress(null);
    }
  }, [
    autoRetrainEnabled,
    activeSpeciesId,
    inferenceSessionId,
    isAutoRetraining,
    models,
    modelToKey,
    selectedModelKey,
    lastAutoRetrainAt,
  ]);

  useEffect(() => {
    if (!activeSpeciesId || !inferenceSessionId || images.length === 0) return;
    void hydratePersistedReviewDrafts(images);
  }, [activeSpeciesId, inferenceSessionId, images, hydratePersistedReviewDrafts]);

  useEffect(() => {
    if (!activeSpeciesId || !inferenceSessionId) {
      setRetrainQueueCount(0);
      return;
    }
    void refreshRetrainQueueSummary();
  }, [activeSpeciesId, inferenceSessionId, refreshRetrainQueueSummary]);

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      const activeResize = sidebarResizeRef.current;
      if (!activeResize) return;
      const nextWidth = clampSidebarWidth(
        activeResize.startWidth + (event.clientX - activeResize.startX)
      );
      setSidebarWidth(nextWidth);
    };

    const handleMouseUp = () => {
      if (!sidebarResizeRef.current) return;
      sidebarResizeRef.current = null;
      setIsResizingSidebar(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(sidebarWidth));
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [clampSidebarWidth, sidebarWidth]);

  // Extracted draw function Ã¢â‚¬â€ callable from both useEffect and mouse handlers
  const drawToCanvas = useCallback((specimens: PredictedSpecimen[]) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const imgWidth = canvas.width;
    const imgHeight = canvas.height;
    const diagonal = Math.sqrt(imgWidth ** 2 + imgHeight ** 2);
    const pointRadius = Math.max(3, diagonal * 0.005);
    const fontSize = Math.max(10, diagonal * 0.012);
    const lineWidth = Math.max(1, diagonal * 0.002);

    const palette = [
      "rgba(255, 0, 0, 0.9)",
      "rgba(0, 200, 255, 0.9)",
      "rgba(255, 140, 0, 0.9)",
      "rgba(170, 255, 0, 0.9)",
    ];

    if (showMaskOverlay && specimens.length > 0) {
      specimens.forEach((specimen, idx) => {
        const outline = Array.isArray(specimen.mask_outline) ? specimen.mask_outline : [];
        if (outline.length < 3) return;
        const color = palette[idx % palette.length];
        ctx.save();
        ctx.beginPath();
        outline.forEach((pt, pi) => {
          const px = Number(pt[0]);
          const py = Number(pt[1]);
          if (pi === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.18;
        ctx.fill();
        ctx.globalAlpha = 0.55;
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(1, lineWidth * 1.5);
        ctx.stroke();
        ctx.restore();
      });
    }

    if (showBoundingBox && specimens.length > 0) {
      const handleR = Math.max(5, diagonal * 0.008);
      specimens.forEach((specimen, idx) => {
        const box = specimen.box;
        if (!box || box.width <= 0 || box.height <= 0) return;
        const color = palette[idx % palette.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = idx === selectedSpecimenIndex ? lineWidth * 3 : lineWidth * 2;
        ctx.setLineDash([10, 5]);
        ctx.strokeRect(box.left, box.top, box.width, box.height);
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font = `bold ${fontSize}px sans-serif`;
        const orientationLabel = resolveOrientationLabelFromBox(specimen?.box);
        const orientationSuffix = ` Â· ${orientationLabel}`;
        const label = idx === selectedSpecimenIndex
          ? `Specimen ${idx + 1} (selected)${orientationSuffix}`
          : `Specimen ${idx + 1}${orientationSuffix}`;
        ctx.fillText(label, box.left + 5, box.top - 5);

        // Corner resize handles
        const corners = [
          { x: box.left,               y: box.top },
          { x: box.left + box.width,   y: box.top },
          { x: box.left,               y: box.top + box.height },
          { x: box.left + box.width,   y: box.top + box.height },
        ];
        corners.forEach(({ x, y }) => {
          ctx.fillStyle = "white";
          ctx.strokeStyle = color;
          ctx.lineWidth = lineWidth * 1.5;
          ctx.beginPath();
          ctx.rect(x - handleR, y - handleR, handleR * 2, handleR * 2);
          ctx.fill();
          ctx.stroke();
        });
      });
    }

    specimens.forEach((specimen, idx) => {
      const color = palette[idx % palette.length];
      specimen.landmarks.forEach((lm) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(lm.x, lm.y, pointRadius, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = "white";
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        ctx.fillStyle = "white";
        ctx.strokeStyle = "black";
        ctx.lineWidth = lineWidth * 1.5;
        ctx.font = `bold ${fontSize}px sans-serif`;
        const labelX = lm.x + pointRadius + 4;
        const labelY = lm.y + fontSize / 3;
        ctx.strokeText(String(lm.id), labelX, labelY);
        ctx.fillText(String(lm.id), labelX, labelY);
      });
    });

    const draftRect = drawBoxRef.current;
    if (draftRect) {
      const x = Math.min(draftRect.start.x, draftRect.current.x);
      const y = Math.min(draftRect.start.y, draftRect.current.y);
      const w = Math.abs(draftRect.current.x - draftRect.start.x);
      const h = Math.abs(draftRect.current.y - draftRect.start.y);
      if (w > 1 && h > 1) {
        ctx.save();
        ctx.setLineDash([10, 5]);
        ctx.lineWidth = Math.max(2, lineWidth * 2);
        ctx.strokeStyle = "rgba(34, 197, 94, 0.95)";
        ctx.fillStyle = "rgba(34, 197, 94, 0.12)";
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(34, 197, 94, 0.95)";
        ctx.font = `bold ${fontSize}px sans-serif`;
        ctx.fillText("New Box", x + 6, Math.max(fontSize + 2, y - 4));
        ctx.restore();
      }
    }
  }, [showBoundingBox, showMaskOverlay, selectedSpecimenIndex, resolveOrientationLabelFromBox]);

  // Load models on mount (or when active session changes)
  useEffect(() => {
    const loadModels = async () => {
      if (!activeSpeciesId) {
        setModels([]);
        setSelectedModelKey("");
        setLoadingModels(false);
        return;
      }
      try {
        const result = await window.api.listModels(activeSpeciesId);
        if (result.ok && result.models) {
          const available = result.models.filter(
            (m) => (m.predictorType ?? "dlib") === "dlib" || m.predictorType === "cnn"
          );
          setModels(available);
          setSelectedModelKey((prev) => {
            if (prev && available.some((m) => modelToKey(m) === prev)) {
              return prev;
            }
            return resolveInitialModelKey(available);
          });
        }
      } catch (err) {
        console.error("Failed to load models:", err);
        toast.error("Failed to load models");
      } finally {
        setLoadingModels(false);
      }
    };
    loadModels();
  }, [activeSpeciesId, modelToKey, resolveInitialModelKey]);

  useEffect(() => {
    if (!initialModel || models.length === 0) return;
    const exact = models.find((m) => modelToKey(m) === initialModel);
    if (exact) {
      setSelectedModelKey(modelToKey(exact));
      return;
    }
    const byName = models.filter((m) => m.name === initialModel);
    if (byName.length > 0) {
      const preferred = byName.find((m) => m.predictorType === "dlib") ?? byName[0];
      setSelectedModelKey(modelToKey(preferred));
    }
  }, [initialModel, models, modelToKey]);

  useEffect(() => {
    let cancelled = false;
    const openInferenceSession = async () => {
      if (!activeSpeciesId) {
        setInferenceSessionId("");
        setInferenceSessionManifest(null);
        return;
      }
      const selectedModel = getSelectedModel();
      if (!selectedModel) {
        setInferenceSessionId("");
        setInferenceSessionManifest(null);
        return;
      }

      const res = await window.api.sessionOpenInferenceSession({
        speciesId: activeSpeciesId,
        landmarkModelKey: selectedModelKey,
        landmarkModelName: selectedModel.name,
        landmarkPredictorType: selectedModel.predictorType ?? "dlib",
        detectionModelKey: "session_detection_default",
        detectionModelName: "Session Detection Model",
      });
      if (cancelled) return;
      if (!res.ok || !res.inferenceSessionId) {
        setInferenceSessionId("");
        setInferenceSessionManifest(null);
        toast.error(res.error || "Failed to open inference session context.");
        return;
      }
      setInferenceSessionId(res.inferenceSessionId);
      setInferenceSessionManifest(res.manifest ?? null);
    };
    void openInferenceSession();
    return () => {
      cancelled = true;
    };
  }, [activeSpeciesId, getSelectedModel, selectedModelKey]);

  // Fetch OBB detector readiness whenever species or selected model changes
  useEffect(() => {
    if (!activeSpeciesId || !selectedModelKey) {
      setObbDetectorReady(null);
      return;
    }
    const selectedModel = models.find((m) => modelToKey(m) === selectedModelKey);
    if (!selectedModel) {
      setObbDetectorReady(null);
      return;
    }
    window.api
      .checkModelCompatibility({ speciesId: activeSpeciesId, modelName: selectedModel.name })
      .then((result) => {
        setObbDetectorReady(result?.obbDetectorReady ?? false);
      })
      .catch(() => setObbDetectorReady(null));
  }, [activeSpeciesId, selectedModelKey, models, modelToKey]);

  const handleSelectImages = async () => {
    if (!activeSpeciesId) {
      toast.error("Select an active session first. Inference is session-scoped.");
      return;
    }
    try {
      const result = await window.api.selectImages();
      if (!result.canceled && result.files) {
        const newImages: InferenceImage[] = result.files.map((file) => {
          const byteCharacters = atob(file.data);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: file.mimeType });
          const url = URL.createObjectURL(blob);
          return { path: file.path, name: file.name, url };
        });
        const combinedImages = [...images, ...newImages];
        setImages(combinedImages);
        await hydratePersistedReviewDrafts(combinedImages);
        toast.success(`Added ${result.files.length} image(s)`);
      }
    } catch (err) {
      console.error("Failed to select images:", err);
      toast.error("Failed to select images");
    }
  };

  const handleRemoveImage = (index: number) => {
    setImages((prev) => {
      const removed = prev[index];
      if (removed?.url) {
        URL.revokeObjectURL(removed.url);
      }
      const next = prev.filter((_, i) => i !== index);
      if (next.length === 0) {
        setCurrentIndex(0);
      } else if (currentIndex >= next.length) {
        setCurrentIndex(next.length - 1);
      } else if (index < currentIndex) {
        setCurrentIndex((c) => Math.max(0, c - 1));
      }
      return next;
    });
    setCorrectedSpecimensMap((prev) => shiftIndicesInMap(prev, index));
    setEditedImageIndices((prev) => shiftIndicesInSet(prev, index));
    setSavedImageIndices((prev) => shiftIndicesInSet(prev, index));
    setReviewFinalizedImageIndices((prev) => shiftIndicesInSet(prev, index));
    setQueuedImageIndices((prev) => shiftIndicesInSet(prev, index));
  };

  const handleRunDetection = async () => {
    if (!activeSpeciesId) {
      toast.error("Select an active session first. Inference is session-scoped.");
      return;
    }
    if (images.length === 0) {
      toast.error("Please add images first.");
      return;
    }

    setIsRunning(true);
    setIsDrawBoxMode(false);
    setShowMaskOverlay(false);
    drawBoxRef.current = null;
    setCorrectedSpecimensMap(new Map());
    setEditedImageIndices(new Set());
    setSavedImageIndices(new Set());
    setReviewFinalizedImageIndices(new Set());
    setQueuedImageIndices(new Set());
    setSelectedSpecimenIndex(null);
    setInferProgress({ percent: 0, stage: "detecting" });

    let successCount = 0;
    let errorCount = 0;
    const updatedImages = [...images];

    for (let i = 0; i < images.length; i++) {
      const img = images[i];
      setInferProgress({
        percent: Math.round((i / Math.max(images.length, 1)) * 100),
        stage: "detecting",
      });
      try {
        const detectResult = await window.api.detectSpecimens(img.path, {
          speciesId: activeSpeciesId,
        });
        if (detectResult.ok && Array.isArray(detectResult.boxes)) {
          const boxes = detectResult.boxes
            .filter((b) => Number(b.width) > 0 && Number(b.height) > 0)
            .map((b) => ({
              left: Number(b.left),
              top: Number(b.top),
              right: Number(b.right),
              bottom: Number(b.bottom),
              width: Number(b.width),
              height: Number(b.height),
              confidence: Number.isFinite(Number(b.confidence)) ? Number(b.confidence) : undefined,
              class_id: Number.isFinite(Number(b.class_id)) ? Number(b.class_id) : undefined,
              class_name: b.class_name,
              orientation_hint: b.orientation_hint,
            }));

          const inferredWidth = detectResult.image_width && detectResult.image_width > 0
            ? detectResult.image_width
            : boxes.reduce((max, b) => Math.max(max, Math.round((b.right ?? b.left + b.width))), 0);
          const inferredHeight = detectResult.image_height && detectResult.image_height > 0
            ? detectResult.image_height
            : boxes.reduce((max, b) => Math.max(max, Math.round((b.bottom ?? b.top + b.height))), 0);

          const detectionResult: InferenceResult = {
            image: img.path,
            specimens: boxes.map((box) => ({
              box,
              landmarks: [],
              num_landmarks: 0,
            })),
            num_specimens: boxes.length,
            image_dimensions: {
              width: Math.max(1, Math.round(inferredWidth || 1)),
              height: Math.max(1, Math.round(inferredHeight || 1)),
            },
          };
          updatedImages[i] = { ...updatedImages[i], results: detectionResult, error: undefined };
          successCount++;
        } else {
          updatedImages[i] = { ...updatedImages[i], error: detectResult.error || "Detection failed" };
          errorCount++;
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Detection failed";
        updatedImages[i] = { ...updatedImages[i], error: errorMessage };
        errorCount++;
      }
    }
    setImages(updatedImages);
    await hydratePersistedReviewDrafts(updatedImages);
    setInferProgress({ percent: 100, stage: "done" });
    setInferProgress(null);
    setIsRunning(false);
    if (successCount > 0) {
      toast.success(`Detection complete: ${successCount} succeeded, ${errorCount} failed`);
    } else {
      toast.error("All detections failed");
    }
  };

  const handleRunInference = async () => {
    if (!activeSpeciesId) {
      toast.error("Select an active session first. Inference is session-scoped.");
      return;
    }
    if (!selectedModelKey || images.length === 0) {
      toast.error("Please select a model and add images.");
      return;
    }
    const selectedModel = models.find((m) => modelToKey(m) === selectedModelKey);
    if (!selectedModel) {
      toast.error("Selected model is unavailable. Reload models and try again.");
      return;
    }

    const selectedPredictor: "dlib" | "cnn" =
      selectedModel.predictorType === "cnn" ? "cnn" : "dlib";

    const hasAnyDetectionBoxes = images.some((_, idx) => {
      const specimens = getSpecimensForImageIndex(idx);
      return specimens.some((s) => s?.box && s.box.width > 0 && s.box.height > 0);
    });
    if (!hasAnyDetectionBoxes) {
      toast.error("Run detection and keep at least one box before landmark inference.");
      return;
    }

    let allowIncompatible = false;
    try {
      const compatibility = await window.api.checkModelCompatibility({
        speciesId: activeSpeciesId,
        modelName: selectedModel.name,
        predictorType: selectedPredictor,
        includeRuntime: true,
      });
      if (!compatibility.ok) {
        toast.error(compatibility.error || "Compatibility check failed.");
        return;
      }
      const blockingIssues = (compatibility.issues || []).filter(
        (issue) => issue.severity === "error"
      );
      if (blockingIssues.length > 0) {
        const message = blockingIssues
          .map((issue, idx) => `${idx + 1}. ${issue.message}`)
          .join("\n");
        const sam2RuntimeNote =
          compatibility.runtime?.sam2Required && !compatibility.runtime?.sam2Ready
            ? "\n\nSAM2 requirement: model was trained with SAM2 segment masks, but SAM2 is not currently available on this system."
            : "";
        const proceed = window.confirm(
          `Model/session compatibility checks blocked inference:\n\n${message}${sam2RuntimeNote}\n\nPress OK to override and run anyway, or Cancel to stop.`
        );
        if (!proceed) return;
        allowIncompatible = true;
        toast.warning("Running with compatibility override for this inference batch.");
      } else {
        const warningIssues = (compatibility.issues || []).filter(
          (issue) => issue.severity === "warning"
        );
        if (warningIssues.length > 0) {
          toast.warning(warningIssues.map((issue) => issue.message).join(" "));
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Compatibility check failed.";
      toast.error(message);
      return;
    }

    setIsRunning(true);
    setInferProgress({ percent: 0, stage: "starting" });
    const unsubscribeProgress = window.api.onPredictProgress((data) => {
      setInferProgress(data);
    });

    const updatedImages = [...images];
    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < images.length; i++) {
      const img = images[i];
      const acceptedBoxes = getSpecimensForImageIndex(i)
        .filter((s) => s?.box && s.box.width > 0 && s.box.height > 0)
        .map((s) => ({
          left: Math.round(s.box.left),
          top: Math.round(s.box.top),
          width: Math.round(s.box.width),
          height: Math.round(s.box.height),
          right: Math.round(s.box.left + s.box.width),
          bottom: Math.round(s.box.top + s.box.height),
          orientation_hint:
            s.box.orientation_override === "left" || s.box.orientation_override === "right"
              ? {
                  orientation: s.box.orientation_override,
                  confidence: 1.0,
                  source: "user_review",
                }
              : s.box.orientation_hint,
        }));
      if (acceptedBoxes.length === 0) {
        updatedImages[i] = {
          ...updatedImages[i],
          error: "No accepted detection boxes for this image.",
        };
        errorCount++;
        continue;
      }

      try {
        const result = await window.api.predictImage(
          img.path,
          selectedModel.name,
          activeSpeciesId,
          {
            multiSpecimen: true,
            predictorType: selectedPredictor,
            allowIncompatible,
            boxes: acceptedBoxes,
          }
        );
        if (result.ok && result.data) {
          updatedImages[i] = { ...updatedImages[i], results: result.data, error: undefined };
          successCount++;
        } else {
          updatedImages[i] = { ...updatedImages[i], error: result.error || "Landmark inference failed" };
          errorCount++;
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Landmark inference failed";
        updatedImages[i] = { ...updatedImages[i], error: errorMessage };
        errorCount++;
      }
    }

    setImages(updatedImages);
    await hydratePersistedReviewDrafts(updatedImages);
    unsubscribeProgress();
    setInferProgress(null);
    setIsRunning(false);
    if (successCount > 0) {
      toast.success(`Landmark inference complete: ${successCount} succeeded, ${errorCount} failed`);
    } else {
      toast.error("All landmark inferences failed");
    }
  };

  const handleSaveAllCorrections = useCallback(async () => {
    if (!activeSpeciesId || !inferenceSessionId) {
      toast.error("Inference session is not ready yet.");
      return;
    }
    const targets = images
      .map((img, idx) => ({ img, idx }))
      .filter(({ img, idx }) => Boolean(img.results) && editedImageIndices.has(idx))
      .map(({ idx }) => idx);
    if (targets.length === 0) {
      toast.message("No unsaved edited images to persist.");
      return;
    }

    setIsSavingCorrections(true);
    let saved = 0;
    let failed = 0;
    try {
      for (const idx of targets) {
        const ok = await saveCorrectionsForImage(idx);
        if (ok) saved += 1;
        else failed += 1;
      }
    } finally {
      setIsSavingCorrections(false);
    }

    if (saved > 0) {
      toast.success(
        failed > 0
          ? `Saved ${saved} image(s), ${failed} failed.`
          : `Saved all ${saved} edited image(s).`
      );
    } else {
      toast.error("Failed to save edited images.");
    }
  }, [
    activeSpeciesId,
    inferenceSessionId,
    images,
    editedImageIndices,
    saveCorrectionsForImage,
  ]);

  const handleQueueFinalizedForRetrain = useCallback(async () => {
    if (!activeSpeciesId || !inferenceSessionId) {
      toast.error("Inference session is not ready yet.");
      return;
    }
    const targets = images
      .map((img, idx) => ({ img, idx }))
      .filter(
        ({ img, idx }) =>
          Boolean(img.results) &&
          savedImageIndices.has(idx) &&
          reviewFinalizedImageIndices.has(idx) &&
          !queuedImageIndices.has(idx)
      )
      .map(({ idx }) => idx);
    if (targets.length === 0) {
      toast.message("No finalized images available to queue.");
      return;
    }

    setIsQueueingRetrain(true);
    let queued = 0;
    let failed = 0;
    try {
      for (const idx of targets) {
        const ok = await queueRetrainForImage(idx);
        if (ok) queued += 1;
        else failed += 1;
      }
    } finally {
      setIsQueueingRetrain(false);
    }

    if (queued > 0) {
      toast.success(
        failed > 0
          ? `Queued ${queued} finalized image(s), ${failed} failed.`
          : `Queued ${queued} finalized image(s) for retraining.`
      );
      if (autoRetrainEnabled) {
        void attemptAutoRetrainIfReady();
      }
    } else {
      toast.error("No finalized images were queued.");
    }
  }, [
    activeSpeciesId,
    inferenceSessionId,
    images,
    savedImageIndices,
    reviewFinalizedImageIndices,
    queuedImageIndices,
    queueRetrainForImage,
    autoRetrainEnabled,
    attemptAutoRetrainIfReady,
  ]);

  const handleRemoveSpecimen = useCallback((specimenIndex: number) => {
    const current = getSpecimensForImageIndex(currentIndex);
    if (current.length === 0) return;
    if (specimenIndex < 0 || specimenIndex >= current.length) return;

    const updated = current.filter((_, idx) => idx !== specimenIndex);
    const cloned = JSON.parse(JSON.stringify(updated)) as PredictedSpecimen[];
    liveSpecimensRef.current = cloned;
    const nextSelected =
      cloned.length === 0
        ? null
        : Math.min(
            specimenIndex >= cloned.length ? cloned.length - 1 : specimenIndex,
            cloned.length - 1
          );
    setSelectedSpecimenIndex(nextSelected);
    setCorrectedSpecimensMap((prev) => {
      const next = new Map(prev);
      next.set(currentIndex, cloned);
      return next;
    });
    markImageEdited(currentIndex);
    drawToCanvas(cloned);
    void persistReviewDraft(currentIndex, { specimens: cloned, edited: true, saved: false });
    toast.success(`Removed specimen ${specimenIndex + 1}.`);
  }, [currentIndex, drawToCanvas, getSpecimensForImageIndex, markImageEdited, persistReviewDraft]);

  const handleDeleteSelectedSpecimen = useCallback(() => {
    const current = getSpecimensForImageIndex(currentIndex);
    if (current.length === 0) {
      toast.error("No detection boxes to delete.");
      return;
    }
    const target = selectedSpecimenIndex ?? 0;
    if (target < 0 || target >= current.length) {
      handleRemoveSpecimen(0);
      return;
    }
    handleRemoveSpecimen(target);
  }, [currentIndex, getSpecimensForImageIndex, handleRemoveSpecimen, selectedSpecimenIndex]);

  const handleExport = (format: "json" | "csv") => {
    const imagesWithResults = images.filter((img) => img.results);
    if (imagesWithResults.length === 0) {
      toast.error("No results to export");
      return;
    }

    let content: string;
    let filename: string;
    let mimeType: string;

    if (format === "json") {
      const data = imagesWithResults.map((img) => {
        const idx = images.findIndex((i) => i.path === img.path);
        const specimens =
          idx >= 0
            ? (correctedSpecimensMap.get(idx) ?? getSpecimensFromResult(img.results))
            : getSpecimensFromResult(img.results);
        const flattenedLandmarks =
          specimens.length === 1
            ? specimens[0].landmarks
            : [];
        return {
          filename: img.name,
          path: img.path,
          landmarks: flattenedLandmarks,
          specimens,
          num_specimens: specimens.length,
        };
      });
      content = JSON.stringify(data, null, 2);
      filename = `inference_results_${Date.now()}.json`;
      mimeType = "application/json";
    } else {
      const rows: string[] = [
        "filename,specimen_index,landmark_id,landmark_x,landmark_y",
      ];
      imagesWithResults.forEach((img) => {
        const idx = images.findIndex((i) => i.path === img.path);
        const specimens =
          idx >= 0
            ? (correctedSpecimensMap.get(idx) ?? getSpecimensFromResult(img.results))
            : getSpecimensFromResult(img.results);
        if (specimens.length > 0) {
          specimens.forEach((specimen, specimenIdx) => {
            specimen.landmarks.forEach((lm) => {
              rows.push(`"${img.name}",${specimenIdx + 1},${lm.id},${lm.x},${lm.y}`);
            });
          });
        }
      });
      content = rows.join("\n");
      filename = `inference_results_${Date.now()}.csv`;
      mimeType = "text/csv";
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Exported to ${filename}`);
  };

  // Canvas coordinate mapping: CSS coords Ã¢â€ â€™ canvas pixel coords
  const toCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const specs = liveSpecimensRef.current || [];
    const { x, y } = toCanvasCoords(e);
    const canvas = canvasRef.current!;
    const diagonal = Math.sqrt(canvas.width ** 2 + canvas.height ** 2);
    const lmHitRadius  = Math.max(8, diagonal * 0.012);
    const boxHitRadius = Math.max(5, diagonal * 0.008);

    if (isDrawBoxMode) {
      drawBoxRef.current = {
        start: { x, y },
        current: { x, y },
      };
      e.currentTarget.style.cursor = "crosshair";
      drawToCanvas(specs);
      return;
    }

    // 1) Landmark hit detection first so landmark edits are never preempted by box move.
    for (let si = 0; si < specs.length; si++) {
      for (let li = 0; li < specs[si].landmarks.length; li++) {
        const lm = specs[si].landmarks[li];
        const dist = Math.sqrt((lm.x - x) ** 2 + (lm.y - y) ** 2);
        if (dist <= lmHitRadius) {
          setSelectedSpecimenIndex(si);
          draggingRef.current = { specIdx: si, lmIdx: li };
          e.currentTarget.style.cursor = "grabbing";
          return;
        }
      }
    }

    // 2) Box interactions: corner handles (resize), then box interior (move)
    if (showBoundingBox) {
      for (let si = 0; si < specs.length; si++) {
        const box = specs[si].box;
        if (!box || box.width <= 0 || box.height <= 0) continue;

        const corners: { cx: number; cy: number; mode: BoxDragMode }[] = [
          { cx: box.left,             cy: box.top,             mode: "resize-tl" },
          { cx: box.left + box.width, cy: box.top,             mode: "resize-tr" },
          { cx: box.left,             cy: box.top + box.height, mode: "resize-bl" },
          { cx: box.left + box.width, cy: box.top + box.height, mode: "resize-br" },
        ];

        for (const corner of corners) {
          if (Math.abs(x - corner.cx) <= boxHitRadius && Math.abs(y - corner.cy) <= boxHitRadius) {
            setSelectedSpecimenIndex(si);
            boxDraggingRef.current = {
              specIdx: si,
              mode: corner.mode,
              startMouse: { x, y },
              startBox: { left: box.left, top: box.top, width: box.width, height: box.height },
            };
            const cursors: Record<BoxDragMode, string> = {
              "resize-tl": "nw-resize", "resize-tr": "ne-resize",
              "resize-bl": "sw-resize", "resize-br": "se-resize",
              move: "move",
            };
            e.currentTarget.style.cursor = cursors[corner.mode];
            return;
          }
        }

        // Box interior Ã¢â€ â€™ move
        if (
          x >= box.left && x <= box.left + box.width &&
          y >= box.top  && y <= box.top  + box.height
        ) {
          setSelectedSpecimenIndex(si);
          boxDraggingRef.current = {
            specIdx: si,
            mode: "move",
            startMouse: { x, y },
            startBox: { left: box.left, top: box.top, width: box.width, height: box.height },
          };
          e.currentTarget.style.cursor = "move";
          return;
        }
      }
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = toCanvasCoords(e);

    // Manual draw-new-box mode
    if (drawBoxRef.current) {
      drawBoxRef.current = {
        ...drawBoxRef.current,
        current: { x, y },
      };
      drawToCanvas(liveSpecimensRef.current || []);
      return;
    }

    // Box drag/resize
    if (boxDraggingRef.current) {
      const { specIdx, mode, startMouse, startBox } = boxDraggingRef.current;
      const dx = x - startMouse.x;
      const dy = y - startMouse.y;
      const box = liveSpecimensRef.current[specIdx].box;
      const MIN_SIZE = 10;

      if (mode === "move") {
        box.left  = startBox.left  + dx;
        box.top   = startBox.top   + dy;
      } else if (mode === "resize-tl") {
        const newW = Math.max(MIN_SIZE, startBox.width  - dx);
        const newH = Math.max(MIN_SIZE, startBox.height - dy);
        box.left  = startBox.left  + (startBox.width  - newW);
        box.top   = startBox.top   + (startBox.height - newH);
        box.width  = newW;
        box.height = newH;
      } else if (mode === "resize-tr") {
        box.top    = startBox.top + dy;
        box.width  = Math.max(MIN_SIZE, startBox.width  + dx);
        box.height = Math.max(MIN_SIZE, startBox.height - dy);
        box.top    = startBox.top + (startBox.height - box.height);
      } else if (mode === "resize-bl") {
        box.left   = startBox.left + dx;
        box.width  = Math.max(MIN_SIZE, startBox.width  - dx);
        box.height = Math.max(MIN_SIZE, startBox.height + dy);
      } else if (mode === "resize-br") {
        box.width  = Math.max(MIN_SIZE, startBox.width  + dx);
        box.height = Math.max(MIN_SIZE, startBox.height + dy);
      }
      // Sync right/bottom fields
      box.right  = box.left + box.width;
      box.bottom = box.top  + box.height;
      drawToCanvas(liveSpecimensRef.current);
      return;
    }

    // Landmark drag
    if (!draggingRef.current) return;
    const { specIdx, lmIdx } = draggingRef.current;
    liveSpecimensRef.current[specIdx].landmarks[lmIdx].x = x;
    liveSpecimensRef.current[specIdx].landmarks[lmIdx].y = y;
    drawToCanvas(liveSpecimensRef.current);
  };

  const handleCanvasMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const drawDraft = drawBoxRef.current;
    const wasBoxDragging = boxDraggingRef.current !== null;
    const wasLmDragging  = draggingRef.current !== null;
    const wasBoxDrawing = drawDraft !== null;

    boxDraggingRef.current = null;
    draggingRef.current    = null;
    drawBoxRef.current = null;
    e.currentTarget.style.cursor = "crosshair";

    if (wasBoxDrawing && drawDraft) {
      const x1 = Math.min(drawDraft.start.x, drawDraft.current.x);
      const y1 = Math.min(drawDraft.start.y, drawDraft.current.y);
      const x2 = Math.max(drawDraft.start.x, drawDraft.current.x);
      const y2 = Math.max(drawDraft.start.y, drawDraft.current.y);
      const width = x2 - x1;
      const height = y2 - y1;
      const MIN_SIZE = 10;

      if (width >= MIN_SIZE && height >= MIN_SIZE) {
        const updated = JSON.parse(JSON.stringify(liveSpecimensRef.current || [])) as PredictedSpecimen[];
        const newSpecimen: PredictedSpecimen = {
          box: {
            left: x1,
            top: y1,
            width,
            height,
            right: x1 + width,
            bottom: y1 + height,
          },
          landmarks: [],
          num_landmarks: 0,
        };
        updated.push(newSpecimen);
        liveSpecimensRef.current = updated;
        setSelectedSpecimenIndex(updated.length - 1);
        setCorrectedSpecimensMap((prev) => {
          const next = new Map(prev);
          next.set(currentIndex, updated);
          return next;
        });
        markImageEdited(currentIndex);
        void persistReviewDraft(currentIndex, { specimens: updated, edited: true, saved: false });
        drawToCanvas(updated);
        toast.success("Added manual detection box.");
      } else {
        drawToCanvas(liveSpecimensRef.current || []);
      }
      return;
    }

    if (!wasBoxDragging && !wasLmDragging) return;

    // Commit corrected specimens (landmarks + boxes) to state
    const updated = JSON.parse(JSON.stringify(liveSpecimensRef.current)) as PredictedSpecimen[];
    setCorrectedSpecimensMap((prev) => {
      const next = new Map(prev);
      next.set(currentIndex, updated);
      return next;
    });
    if (wasLmDragging || wasBoxDragging) {
      markImageEdited(currentIndex);
      void persistReviewDraft(currentIndex, { specimens: updated, edited: true, saved: false });
    }
  };

  // Draw landmarks on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const currentImage = images[currentIndex];
    if (!currentImage) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const img = new Image();
    img.src = currentImage.url;
    imageRef.current = img;

    img.onload = () => {
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;

      // Use corrected specimens if available, otherwise derive from inference results
      const specimens =
        correctedSpecimensMap.get(currentIndex) ??
        getSpecimensFromResult(currentImage.results);
      liveSpecimensRef.current = JSON.parse(JSON.stringify(specimens));
      setSelectedSpecimenIndex((prev) => {
        if (specimens.length === 0) return null;
        if (prev === null || prev < 0 || prev >= specimens.length) return 0;
        return prev;
      });
      drawToCanvas(specimens);
    };
  }, [currentIndex, images, showBoundingBox, correctedSpecimensMap, drawToCanvas]);

  const currentImage = images[currentIndex];
  const currentSpecimens = getSpecimensForImageIndex(currentIndex);
  const currentSpecimenCount = currentSpecimens.length;
  const currentSam2Count = currentSpecimens.filter(
    (s) => s?.inference_metadata?.mask_source === "sam2"
  ).length;
  const currentRoughMaskCount = currentSpecimens.filter(
    (s) => s?.inference_metadata?.mask_source === "rough_otsu"
  ).length;
  const currentMaskOverlayCount = currentSpecimens.filter(
    (s) => Array.isArray(s?.mask_outline) && s.mask_outline.length >= 3
  ).length;
  const currentHasMaskOverlays = currentMaskOverlayCount > 0;
  const hasInferenceResults = images.some((img) => Boolean(img.results));
  const currentEdited = editedImageIndices.has(currentIndex);
  const currentSaved = savedImageIndices.has(currentIndex);
  const currentQueued = queuedImageIndices.has(currentIndex);
  const reviewActionsDisabled =
    isRunning ||
    isSavingCorrections ||
    isQueueingRetrain ||
    isAutoRetraining ||
    !currentImage?.results ||
    !inferenceSessionId;
  const selectedSpecimenResolvedIndex =
    currentSpecimenCount > 0
      ? Math.min(
          Math.max(selectedSpecimenIndex ?? 0, 0),
          currentSpecimenCount - 1
        )
      : null;
  const selectedSpecimen =
    selectedSpecimenResolvedIndex !== null
      ? currentSpecimens[selectedSpecimenResolvedIndex]
      : null;
  const selectedOrientation = resolveOrientationLabelFromBox(selectedSpecimen?.box);
  return (
    <div className="flex h-screen w-screen flex-col bg-background">
      {/* Header */}
      <div className="flex items-center justify-between border-b p-4">
        <div className="flex items-center gap-4">
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onNavigate("landing")}
            >
              <ArrowLeft className="h-5 w-5" />
            </Button>
          </motion.div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <Microscope className="h-5 w-5 text-primary" />
              <h1 className="text-lg font-bold">Run Inference</h1>
            </div>
            <div className="flex flex-wrap items-center gap-1 text-[10px] text-muted-foreground">
              {activeSpeciesId && (
                <span className="rounded bg-muted px-1.5 py-0.5">
                  Schema: {activeSpeciesId}
                </span>
              )}
              {selectedModelKey && (
                <span className="rounded bg-muted px-1.5 py-0.5">
                  Landmark: {selectedModelKey}
                </span>
              )}
              {inferenceSessionId && (
                <span className="rounded bg-blue-500/15 px-1.5 py-0.5 text-blue-600">
                  Inference Session: {inferenceSessionId}
                </span>
              )}
              {inferenceSessionManifest?.models?.landmark?.predictorType && (
                <span className="rounded bg-muted px-1.5 py-0.5">
                  Predictor: {String(inferenceSessionManifest.models.landmark.predictorType).toUpperCase()}
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {hasInferenceResults && (
            <div className="hidden items-center gap-2 lg:flex">
              <Button
                variant={showBoundingBox ? "default" : "outline"}
                size="sm"
                onClick={() => setShowBoundingBox(!showBoundingBox)}
                title="Toggle detected region box"
              >
                <Square className="mr-2 h-4 w-4" />
                {showBoundingBox ? "Boxes On" : "Boxes Off"}
              </Button>
              <Button
                variant={showMaskOverlay ? "default" : "outline"}
                size="sm"
                onClick={() => setShowMaskOverlay((prev) => !prev)}
                disabled={!currentHasMaskOverlays}
                title={
                  currentHasMaskOverlays
                    ? "Toggle segmentation mask overlay"
                    : "No mask outlines available for this image"
                }
              >
                <ImageIcon className="mr-2 h-4 w-4" />
                {showMaskOverlay ? "Masks On" : "Masks Off"}
              </Button>
              {currentQueued ? (
                <span className="rounded bg-indigo-500/15 px-2 py-1 text-xs font-semibold text-indigo-600">
                  Queued
                </span>
              ) : currentSaved ? (
                <span className="rounded bg-emerald-500/15 px-2 py-1 text-xs font-semibold text-emerald-600">
                  Finalized
                </span>
              ) : currentEdited ? (
                <span className="rounded bg-amber-500/15 px-2 py-1 text-xs font-semibold text-amber-600">
                  Edited
                </span>
              ) : null}
            </div>
          )}
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              variant="outline"
              onClick={handleRunDetection}
              disabled={isRunning || isAutoRetraining || !activeSpeciesId || images.length === 0}
            >
              {isRunning ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Square className="mr-2 h-4 w-4" />
                  Run Detection
                </>
              )}
            </Button>
          </motion.div>
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              onClick={handleRunInference}
              disabled={isRunning || isAutoRetraining || !activeSpeciesId || !selectedModelKey || images.length === 0}
            >
              {isRunning ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : isAutoRetraining ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Auto-retraining...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Landmark Inference
                </>
              )}
            </Button>
          </motion.div>
        </div>
      </div>

      {/* Inference progress bar */}
      {isRunning && inferProgress && (
        <div className="border-b px-4 py-2">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all duration-300"
              style={{ width: `${inferProgress.percent}%` }}
            />
          </div>
          <p className="mt-1 text-center text-xs text-muted-foreground">
            {STAGE_LABELS[inferProgress.stage] ?? inferProgress.stage}
          </p>
        </div>
      )}
      {isAutoRetraining && autoRetrainProgress && (
        <div className="border-b px-4 py-2">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-emerald-500 transition-all duration-300"
              style={{ width: `${autoRetrainProgress.percent}%` }}
            />
          </div>
          <p className="mt-1 text-center text-xs text-muted-foreground">
            Auto-retrain: {autoRetrainProgress.message}
          </p>
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div
          className="shrink-0 border-r bg-card"
          style={{ width: `${sidebarWidth}px` }}
        >
          <ScrollArea className="h-full">
            <div className="space-y-4 p-4">
              {/* Model Selection */}
              <motion.div
                variants={cardHover}
                initial="initial"
                whileHover="hover"
              >
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                      Model
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {loadingModels ? (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading...
                      </div>
                    ) : !activeSpeciesId ? (
                      <p className="text-xs text-muted-foreground">
                        Select an active session to load session-scoped models.
                      </p>
                    ) : models.length === 0 ? (
                      <p className="text-xs text-muted-foreground">
                        No trained models found. Train a model first.
                      </p>
                    ) : (
                      <div className="space-y-2">
                        <Label className="text-sm">Select Model</Label>
                        <select
                          value={selectedModelKey}
                          onChange={(e) => setSelectedModelKey(e.target.value)}
                          className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                        >
                          {models.map((model) => (
                            <option key={modelToKey(model)} value={modelToKey(model)}>
                              {model.name}
                              {" "}
                              ({predictorLabel(model.predictorType)})
                            </option>
                          ))}
                        </select>
                        {/* OBB detector readiness indicator */}
                        {obbDetectorReady === false && (
                          <div className="rounded-md border border-amber-500/40 bg-amber-500/5 px-3 py-2 mt-1">
                            <p className="text-[11px] font-semibold text-amber-700 dark:text-amber-400">
                              No OBB detector trained — orientation accuracy may be reduced. Train an OBB detector in the Training dialog.
                            </p>
                          </div>
                        )}
                        {obbDetectorReady === true && (
                          <p className="text-[11px] text-green-600 dark:text-green-400 mt-1">
                            ✓ OBB detector active — orientation from detector geometry.
                          </p>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>

              {/* Review Controls */}
              {hasInferenceResults && (
                <motion.div
                  variants={cardHover}
                  initial="initial"
                  whileHover="hover"
                >
                  <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                        Review
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex items-center justify-between rounded-md border border-border/70 px-2 py-1.5">
                        <span className="text-[11px] font-medium text-muted-foreground">
                          Auto-retrain
                        </span>
                        <div className="flex items-center gap-2">
                          <Switch
                            checked={autoRetrainEnabled}
                            onCheckedChange={setAutoRetrainEnabled}
                            disabled={reviewActionsDisabled || !selectedModelKey}
                          />
                          <span className="text-[10px] text-muted-foreground">
                            {retrainQueueCount}/{AUTO_RETRAIN_MIN_QUEUE}
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleSaveAllCorrections}
                          disabled={reviewActionsDisabled}
                          title="Persist all edited inference corrections across this page"
                          className="justify-start"
                        >
                          {isSavingCorrections && !isQueueingRetrain ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Save className="mr-2 h-4 w-4" />
                          )}
                          Save All Changes
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleQueueFinalizedForRetrain}
                          disabled={reviewActionsDisabled}
                          title="Queue all finalized images in this inference session for retraining"
                          className="justify-start"
                        >
                          {isQueueingRetrain || isAutoRetraining ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Save className="mr-2 h-4 w-4" />
                          )}
                          Queue Finalized for Retrain
                        </Button>
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <Button
                          variant={isDrawBoxMode ? "default" : "outline"}
                          size="sm"
                          onClick={() => {
                            setIsDrawBoxMode((prev) => !prev);
                            drawBoxRef.current = null;
                            drawToCanvas(currentSpecimens);
                          }}
                          disabled={reviewActionsDisabled}
                          title="Toggle manual bounding-box drawing mode"
                          className="justify-start"
                        >
                          <Square className="mr-2 h-4 w-4" />
                          {isDrawBoxMode ? "Drawing" : "Draw Box"}
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleDeleteSelectedSpecimen}
                          disabled={reviewActionsDisabled || currentSpecimenCount === 0}
                          title="Delete selected detection box and its landmarks"
                          className="justify-start text-destructive hover:bg-destructive/10 hover:text-destructive"
                        >
                          <Trash2 className="mr-2 h-4 w-4" />
                          Delete Box
                        </Button>
                      </div>

                      {currentSpecimenCount > 0 && selectedSpecimenResolvedIndex !== null && (
                        <div className="space-y-2 rounded-md border border-border/70 bg-background/70 p-2">
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] font-medium text-muted-foreground">
                              Selected Box {selectedSpecimenResolvedIndex + 1} / {currentSpecimenCount}
                            </span>
                            <div className="flex items-center gap-1">
                              <Button
                                variant="outline"
                                size="icon"
                                className="h-6 w-6"
                                disabled={selectedSpecimenResolvedIndex <= 0}
                                onClick={() =>
                                  setSelectedSpecimenIndex((prev) =>
                                    Math.max(0, (prev ?? 0) - 1)
                                  )
                                }
                              >
                                <ChevronLeft className="h-3 w-3" />
                              </Button>
                              <Button
                                variant="outline"
                                size="icon"
                                className="h-6 w-6"
                                disabled={selectedSpecimenResolvedIndex >= currentSpecimenCount - 1}
                                onClick={() =>
                                  setSelectedSpecimenIndex((prev) =>
                                    Math.min(
                                      currentSpecimenCount - 1,
                                      (prev ?? 0) + 1
                                    )
                                  )
                                }
                              >
                                <ChevronRight className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                          <div className="grid grid-cols-3 gap-1">
                            <Button
                              variant={selectedOrientation === "left" ? "default" : "outline"}
                              size="sm"
                              className="h-7 px-2 text-[11px]"
                              onClick={() =>
                                handleSetSpecimenOrientation(
                                  selectedSpecimenResolvedIndex,
                                  "left"
                                )
                              }
                            >
                              Left
                            </Button>
                            <Button
                              variant={selectedOrientation === "right" ? "default" : "outline"}
                              size="sm"
                              className="h-7 px-2 text-[11px]"
                              onClick={() =>
                                handleSetSpecimenOrientation(
                                  selectedSpecimenResolvedIndex,
                                  "right"
                                )
                              }
                            >
                              Right
                            </Button>
                            <Button
                              variant={selectedOrientation === "uncertain" ? "default" : "outline"}
                              size="sm"
                              className="h-7 px-2 text-[11px]"
                              onClick={() =>
                                handleSetSpecimenOrientation(
                                  selectedSpecimenResolvedIndex,
                                  "uncertain"
                                )
                              }
                            >
                              Uncertain
                            </Button>
                          </div>
                        </div>
                      )}

                      <Popover>
                        <PopoverTrigger asChild>
                          <Button variant="outline" size="sm" className="w-full justify-start">
                            <Download className="mr-2 h-4 w-4" />
                            Export Results
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent align="start" className="w-40 p-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="w-full justify-start"
                            onClick={() => handleExport("json")}
                          >
                            <FileJson className="mr-2 h-4 w-4" />
                            JSON
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="w-full justify-start"
                            onClick={() => handleExport("csv")}
                          >
                            <FileSpreadsheet className="mr-2 h-4 w-4" />
                            CSV
                          </Button>
                        </PopoverContent>
                      </Popover>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Images */}
              <motion.div
                variants={cardHover}
                initial="initial"
                whileHover="hover"
              >
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                        Images ({images.length})
                      </CardTitle>
                      <motion.div {...buttonHover} {...buttonTap}>
                        <Button size="sm" variant="outline" onClick={handleSelectImages}>
                          <Upload className="mr-1 h-3 w-3" />
                          Add
                        </Button>
                      </motion.div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {images.length === 0 ? (
                      <div className="flex flex-col items-center rounded-md border border-dashed p-6 text-center">
                        <ImageIcon className="mb-2 h-8 w-8 text-muted-foreground/50" />
                        <p className="text-xs text-muted-foreground">
                          No images added yet
                        </p>
                        <motion.div {...buttonHover} {...buttonTap} className="mt-2">
                          <Button size="sm" variant="outline" onClick={handleSelectImages}>
                            Select Images
                          </Button>
                        </motion.div>
                      </div>
                    ) : (
                      <motion.div
                        variants={staggerContainer}
                        initial="initial"
                        animate="animate"
                        className="space-y-2"
                      >
                        {images.map((img, idx) => (
                          <motion.div
                            key={img.path}
                            variants={staggerItem}
                            className={cn(
                              "flex items-center gap-2 rounded-md border p-2 text-xs",
                              idx === currentIndex
                                ? "border-primary bg-primary/5"
                                : "border-border/50",
                              idx !== currentIndex && img.results && !editedImageIndices.has(idx) && !savedImageIndices.has(idx) && "border-green-500/50",
                              idx !== currentIndex && editedImageIndices.has(idx) && "border-amber-500/60 bg-amber-500/5",
                              idx !== currentIndex && savedImageIndices.has(idx) && "border-emerald-500/60 bg-emerald-500/5",
                              img.error && "border-destructive/50"
                            )}
                            onClick={() => setCurrentIndex(idx)}
                          >
                            <div className="min-w-0 flex-1 truncate">
                              {img.name}
                            </div>
                            {img.results && (
                              <span className="shrink-0 text-green-500">
                                {getTotalLandmarks(img.results)} pts
                              </span>
                            )}
                            {img.results && queuedImageIndices.has(idx) && (
                              <span className="shrink-0 rounded bg-indigo-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-600">
                                Queued
                              </span>
                            )}
                            {img.results && !queuedImageIndices.has(idx) && savedImageIndices.has(idx) && (
                              <span className="shrink-0 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-600">
                                Finalized
                              </span>
                            )}
                            {img.results && !savedImageIndices.has(idx) && editedImageIndices.has(idx) && (
                              <span className="shrink-0 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-amber-600">
                                Edited
                              </span>
                            )}
                            {img.results && !savedImageIndices.has(idx) && !editedImageIndices.has(idx) && (
                              <span className="shrink-0 rounded bg-blue-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-blue-600">
                                Ready
                              </span>
                            )}
                            {img.error && (
                              <span className="shrink-0 text-destructive">Error</span>
                            )}
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 shrink-0"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleRemoveImage(idx);
                              }}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </motion.div>
                        ))}
                      </motion.div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </ScrollArea>
        </div>

        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize sidebar"
          title="Drag to resize sidebar"
          className={cn(
            "w-1 shrink-0 cursor-col-resize bg-border/60 transition-colors hover:bg-primary/50",
            isResizingSidebar && "bg-primary/70"
          )}
          onMouseDown={handleSidebarResizeStart}
        />

        {/* Main content area */}
        <div className="relative flex flex-1 flex-col items-center justify-center overflow-hidden bg-muted/30 p-4">
          {images.length === 0 ? (
            <div className="text-center">
              <Microscope className="mx-auto mb-4 h-16 w-16 text-muted-foreground/50" />
              <h2 className="text-lg font-semibold">No images to display</h2>
              <p className="mt-2 max-w-sm text-sm text-muted-foreground">
                Add images and select a model to run inference.
              </p>
            </div>
          ) : (
            <>
              {/* Canvas Ã¢â‚¬â€ drag landmarks to correct them */}
              <div className="relative flex-1 overflow-hidden">
                <canvas
                  ref={canvasRef}
                  className="max-h-full max-w-full object-contain"
                  style={{
                    display: "block",
                    margin: "auto",
                    cursor: isDrawBoxMode || currentSpecimenCount > 0 ? "crosshair" : "default",
                  }}
                  onMouseDown={handleCanvasMouseDown}
                  onMouseMove={handleCanvasMouseMove}
                  onMouseUp={handleCanvasMouseUp}
                  onMouseLeave={handleCanvasMouseUp}
                />
              </div>

              {/* Navigation */}
              <div className="mt-4 flex items-center gap-4">
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
                    disabled={currentIndex === 0}
                  >
                    <ChevronLeft className="h-5 w-5" />
                  </Button>
                </motion.div>
                <span className="text-sm text-muted-foreground">
                  {currentIndex + 1} / {images.length}
                </span>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      setCurrentIndex(Math.min(images.length - 1, currentIndex + 1))
                    }
                    disabled={currentIndex === images.length - 1}
                  >
                    <ChevronRight className="h-5 w-5" />
                  </Button>
                </motion.div>
              </div>

              {/* Current image info */}
              {currentImage && (
                <div className="mt-2 text-center text-xs text-muted-foreground">
                  <p className="font-medium">{currentImage.name}</p>
                  {currentImage.error && (
                    <p className="text-destructive">{currentImage.error}</p>
                  )}
                  {currentImage.results && (
                    <>
                      <p className="text-green-500">
                        Found {getTotalLandmarks(currentImage.results)} landmark(s)
                        {currentSpecimenCount > 1
                          ? ` across ${currentSpecimenCount} specimens`
                          : ""}
                      </p>
                      <div className="mt-1 flex flex-wrap items-center justify-center gap-1.5">
                        {currentSam2Count > 0 ? (
                          <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-600">
                            SAM2: {currentSam2Count}/{currentSpecimenCount}
                          </span>
                        ) : currentRoughMaskCount > 0 ? (
                          <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700">
                            Mask Fallback: Otsu ({currentRoughMaskCount})
                          </span>
                        ) : (
                          <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-semibold text-muted-foreground">
                            Mask Source: unavailable
                          </span>
                        )}
                        <span className="rounded bg-blue-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-blue-600">
                          Overlay: {currentMaskOverlayCount}
                        </span>
                      </div>
                      {currentSpecimenCount > 0 && (
                        <p className="text-muted-foreground/70">
                          Drag landmarks and bounding box corners, then save review to include this image in retraining.
                        </p>
                      )}
                      {isDrawBoxMode && (
                        <p className="text-emerald-600">
                          Draw-box mode is on. Drag on the image to add a new detection box.
                        </p>
                      )}
                      {currentSpecimenCount > 0 && (
                        <p className="text-muted-foreground/70">
                          Use the Review panel in the sidebar to set orientation, add/delete boxes, and save corrections.
                        </p>
                      )}
                      {currentQueued && (
                        <p className="font-medium text-indigo-600">Queued for retraining. Run training when ready.</p>
                      )}
                      {!autoRetrainEnabled && retrainQueueCount > 0 && (
                        <p className="text-muted-foreground/70">
                          Auto-retrain is off. Queued corrections will be used in your next manual training run.
                        </p>
                      )}
                      {!currentQueued && currentSaved && (
                        <p className="font-medium text-emerald-600">Finalized in session for retraining.</p>
                      )}
                      {!currentSaved && currentEdited && (
                        <p className="font-medium text-amber-600">Edited locally. Save to persist corrections.</p>
                      )}
                      {currentImage.results.image_dimensions && (
                        <p className="text-muted-foreground/70">
                          Image: {currentImage.results.image_dimensions.width} x {currentImage.results.image_dimensions.height}px
                        </p>
                      )}
                      {currentSpecimenCount > 1 ? (
                        <p className="text-muted-foreground/70">
                          Detection boxes: {currentSpecimenCount}
                        </p>
                      ) : currentSpecimenCount === 1 && currentSpecimens[0]?.box ? (
                        <p className="text-muted-foreground/70">
                          Detection box: ({Math.round(currentSpecimens[0].box.left)}, {Math.round(currentSpecimens[0].box.top)}){" -> "}
                          ({Math.round(currentSpecimens[0].box.left + currentSpecimens[0].box.width)}, {Math.round(currentSpecimens[0].box.top + currentSpecimens[0].box.height)})
                        </p>
                      ) : null}
                      {currentSpecimenCount > 0 && (
                        <details className="mx-auto mt-1 w-full max-w-3xl rounded border border-border/60 bg-background/50 px-2 py-1 text-left text-[10px] text-muted-foreground/80">
                          <summary className="cursor-pointer select-none text-[11px] font-medium text-muted-foreground">
                            Inference metadata (advanced)
                          </summary>
                          <div className="mt-1 space-y-0.5">
                            {currentSpecimens.map((specimen, idx) => {
                              const meta = specimen?.inference_metadata;
                              if (!meta) return null;
                              const pcaAngle =
                                typeof meta.pca_angle === "number"
                                  ? meta.pca_angle
                                  : meta.pca_rotation;
                              const dirConf =
                                typeof meta.direction_confidence === "number"
                                  ? meta.direction_confidence
                                  : meta.inferred_direction_confidence;
                              const detectorHint = meta.detector_hint_orientation ?? "none";
                              const detectorHintSource = meta.detector_hint_source ?? "n/a";
                              const canonicalFlip =
                                typeof meta.canonical_flip_applied === "boolean"
                                  ? (meta.canonical_flip_applied ? "yes" : "no")
                                  : "n/a";
                              return (
                                <p key={`inference-meta-${idx}`}>
                                  #{idx + 1}: hint {detectorHint} ({detectorHintSource}) | mask {meta.mask_source ?? "none"} | pca{" "}
                                  {typeof pcaAngle === "number" ? `${pcaAngle.toFixed(1)}°` : "n/a"} | canonical-flip {canonicalFlip} | dir{" "}
                                  {meta.direction_source ?? "n/a"}
                                  {typeof dirConf === "number" ? ` (${dirConf.toFixed(2)})` : ""} | flip{" "}
                                  {meta.was_flipped ? "yes" : "no"}
                                  {meta.orientation_warning?.code ? ` | warn ${meta.orientation_warning.code}` : ""}
                                </p>
                              );
                            })}
                          </div>
                        </details>
                      )}
                    </>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default InferencePage;


