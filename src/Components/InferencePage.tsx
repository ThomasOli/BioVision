import React, { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  Microscope,
  Upload,
  Play,
  Save,
  Plus,
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
import { useDispatch, useSelector } from "react-redux";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { Input } from "@/Components/ui/input";
import { Label } from "@/Components/ui/label";
import { ScrollArea } from "@/Components/ui/scroll-area";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/Components/ui/popover";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/Components/ui/dialog";
import { staggerContainer, staggerItem, buttonHover, buttonTap, cardHover } from "@/lib/animations";
import { TrainedModel, AppView } from "@/types/Image";
import { setActiveSpecies, removeSpecies } from "@/state/speciesState/speciesSlice";
import type { AppDispatch, RootState } from "@/state/store";

const STAGE_LABELS: Record<string, string> = {
  checking_compatibility: "Checking compatibility...",
  preparing_worker: "Preparing landmark worker...",
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

interface InferencePageProps {
  onNavigate: (view: AppView) => void;
  initialModel?: string;
  hasActivatedSchemaThisRun?: boolean;
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
  obbCorners?: [number, number][];
  angle?: number;
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
    clamped_landmark_count?: number;
    inferenceSignature?: string;
  };
}

interface InferenceResult {
  image: string;
  landmarks?: PredictionLandmark[];
  detected_box?: DetectedBox;
  image_dimensions?: ImageDimensions;
  specimens?: PredictedSpecimen[];
  num_specimens?: number;
  boxSignature?: string;
  inferenceSignature?: string;
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
  }[];
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

interface PredictProgressState {
  percent: number;
  stage: string;
  currentIndex?: number;
  total?: number;
  imagePath?: string;
}

interface LocalInferenceSessionManifest {
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

interface InferenceSessionCard {
  speciesId: string;
  schemaName: string;
  schemaImageCount: number;
  schemaUpdatedAt: string;
  exists: boolean;
  inferenceSessionId?: string;
  displayName?: string;
  createdAt?: string;
  updatedAt?: string;
  migratedFrom?: string;
}

interface SchemaOption {
  id: string;
  name: string;
}

type PredictorKey = "dlib" | "cnn";

const roundBoxNumber = (value: number | undefined) => Math.round(Number(value) || 0);

function getStableBoxSignature(
  imagePath: string,
  boxes: Array<DetectedBox | {
    left: number;
    top: number;
    width: number;
    height: number;
    right?: number;
    bottom?: number;
    obbCorners?: [number, number][];
    angle?: number;
    class_id?: number;
  }>
): string {
  const normalized = (boxes || [])
    .map((box) => {
      const left = roundBoxNumber(box.left);
      const top = roundBoxNumber(box.top);
      const width = roundBoxNumber(box.width);
      const height = roundBoxNumber(box.height);
      const right = roundBoxNumber(box.right ?? left + width);
      const bottom = roundBoxNumber(box.bottom ?? top + height);
      const obbCorners = Array.isArray(box.obbCorners)
        ? box.obbCorners.map(([x, y]) => [roundBoxNumber(x), roundBoxNumber(y)] as [number, number])
        : undefined;
      return {
        left,
        top,
        width,
        height,
        right,
        bottom,
        angle: typeof box.angle === "number" ? Number(box.angle.toFixed(4)) : undefined,
        class_id: Number.isFinite(Number(box.class_id)) ? Number(box.class_id) : undefined,
        obbCorners,
      };
    })
    .sort((a, b) =>
      a.top - b.top ||
      a.left - b.left ||
      (a.class_id ?? -1) - (b.class_id ?? -1) ||
      a.width - b.width ||
      a.height - b.height
    );
  return JSON.stringify({ imagePath, boxes: normalized });
}

function getInferenceSignature(
  selectedModelKey: string,
  predictorType: PredictorKey,
  boxSignature: string
): string {
  return JSON.stringify({
    selectedModelKey,
    predictorType,
    boxSignature,
  });
}

export const InferencePage: React.FC<InferencePageProps> = ({
  onNavigate,
  initialModel,
  hasActivatedSchemaThisRun = false,
}) => {
  const dispatch = useDispatch<AppDispatch>();
  const activeSpeciesId = useSelector((state: RootState) => state.species.activeSpeciesId);
  const allSpecies = useSelector((state: RootState) => state.species.species);
  const [selectedInferenceSpeciesId, setSelectedInferenceSpeciesId] = useState<string | null>(null);
  const [showInferenceHub, setShowInferenceHub] = useState(true);
  // effectiveSessionId: use the user-selected inference session, or fall back to the globally active one
  const effectiveSessionId = showInferenceHub
    ? null
    : (selectedInferenceSpeciesId ?? activeSpeciesId);
  const activeOrientationMode = useSelector((state: RootState) =>
    state.species.species.find(s => s.id === (selectedInferenceSpeciesId ?? state.species.activeSpeciesId))?.orientationPolicy?.mode
  );
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
  const [inferProgress, setInferProgress] = useState<PredictProgressState | null>(null);

  // HITL correction state
  const [correctedSpecimensMap, setCorrectedSpecimensMap] = useState<Map<number, PredictedSpecimen[]>>(new Map());
  const [editedImageIndices, setEditedImageIndices] = useState<Set<number>>(new Set());
  const [savedImageIndices, setSavedImageIndices] = useState<Set<number>>(new Set());
  const [reviewFinalizedImageIndices, setReviewFinalizedImageIndices] = useState<Set<number>>(new Set());
  const [committedImageIndices, setCommittedImageIndices] = useState<Set<number>>(new Set());
  const [isReviewStateHydrated, setIsReviewStateHydrated] = useState(false);
  const [selectedSpecimenIndex, setSelectedSpecimenIndex] = useState<number | null>(null);
  const [detectionRerunModelKey, setDetectionRerunModelKey] = useState<string | null>(null);
  const [inferenceRerunModelKey, setInferenceRerunModelKey] = useState<string | null>(null);
  const draggingRef = useRef<{ specIdx: number; lmIdx: number } | null>(null);
  const liveSpecimensRef = useRef<PredictedSpecimen[]>([]); // mutable during drag
  const reviewHydrationRequestRef = useRef(0);
  const pendingImageLoadRef = useRef<Promise<any> | null>(null);
  const incompatiblePreferenceNoticeRef = useRef<string | null>(null);
  const lastSelectedModelKeyRef = useRef<string>("");
  const sessionPreferredModelKeyRef = useRef<string>("");
  const preferredPredictorTypeRef = useRef<TrainedModel["predictorType"]>("cnn");
  const [isCommittingReview, setIsCommittingReview] = useState(false);
  const [inferenceSessionId, setInferenceSessionId] = useState<string>("");
  const [inferenceSessionManifest, setInferenceSessionManifest] = useState<LocalInferenceSessionManifest | null>(null);
  const [sessionPreferredModelKey, setSessionPreferredModelKey] = useState<string>("");
  const [hubSessions, setHubSessions] = useState<InferenceSessionCard[]>([]);
  const [loadingHubSessions, setLoadingHubSessions] = useState(false);
  const [openingInferenceSessionSpeciesId, setOpeningInferenceSessionSpeciesId] = useState<string | null>(null);
  const [createSessionDialogOpen, setCreateSessionDialogOpen] = useState(false);
  const [createSessionSpeciesId, setCreateSessionSpeciesId] = useState<string>("");
  const [createSessionName, setCreateSessionName] = useState("");
  const [obbDetectorReady, setObbDetectorReady] = useState<boolean | null>(null);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);

  const schemaOptions = useMemo<SchemaOption[]>(() => {
    const merged = new Map<string, SchemaOption>();

    hubSessions.forEach((session) => {
      if (!session?.speciesId) return;
      merged.set(session.speciesId, {
        id: session.speciesId,
        name: session.schemaName || session.speciesId,
      });
    });

    allSpecies.forEach((species) => {
      if (!species?.id) return;
      const existing = merged.get(species.id);
      merged.set(species.id, {
        id: species.id,
        name: species.name || existing?.name || species.id,
      });
    });

    return Array.from(merged.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [allSpecies, hubSessions]);

  const hasKnownSchemas =
    hubSessions.length > 0 || allSpecies.length > 0 || hasActivatedSchemaThisRun;
  const isFirstTimeLocked = !loadingHubSessions && !hasKnownSchemas;

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
    setCommittedImageIndices((prev) => {
      const next = new Set(prev);
      next.delete(index);
      return next;
    });
  }, []);

  useEffect(() => {
    if (!createSessionDialogOpen) return;
    const existing = schemaOptions.some((schema) => schema.id === createSessionSpeciesId);
    if (existing) return;
    if (activeSpeciesId && schemaOptions.some((schema) => schema.id === activeSpeciesId)) {
      setCreateSessionSpeciesId(activeSpeciesId);
      return;
    }
    if (schemaOptions.length > 0) {
      setCreateSessionSpeciesId(schemaOptions[0].id);
    } else {
      setCreateSessionSpeciesId("");
    }
  }, [activeSpeciesId, createSessionDialogOpen, createSessionSpeciesId, schemaOptions]);

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

  const preferredPredictorType = useMemo<TrainedModel["predictorType"]>(() => {
    return (
      inferenceSessionManifest?.preferences?.lastUsedPredictorType ||
      inferenceSessionManifest?.models?.landmark?.predictorType ||
      "cnn"
    );
  }, [inferenceSessionManifest]);

  const stampInferenceResult = useCallback((
    result: InferenceResult,
    imagePath: string,
    boxes: DetectedBox[],
    options?: {
      selectedModelKey?: string;
      predictorType?: PredictorKey;
    }
  ): InferenceResult => {
    const boxSignature = getStableBoxSignature(imagePath, boxes);
    const inferenceSignature =
      options?.selectedModelKey && options?.predictorType
        ? getInferenceSignature(options.selectedModelKey, options.predictorType, boxSignature)
        : undefined;
    return {
      ...result,
      boxSignature,
      inferenceSignature,
      specimens: (result.specimens || []).map((specimen) => ({
        ...specimen,
        inference_metadata: {
          ...(specimen.inference_metadata || {}),
          inferenceSignature,
        },
      })),
    };
  }, []);

  const resolveInitialModelKey = useCallback((available: TrainedModel[]): string => {
    if (!available.length) return "";
    if (sessionPreferredModelKeyRef.current) {
      const exactPreferred = available.find((m) => modelToKey(m) === sessionPreferredModelKeyRef.current);
      if (exactPreferred) return modelToKey(exactPreferred);
    }
    const activeForPreferredPredictor = available.find(
      (m) => (m.predictorType ?? "dlib") === (preferredPredictorTypeRef.current ?? "dlib")
    );
    if (!initialModel && activeForPreferredPredictor) {
      return modelToKey(activeForPreferredPredictor);
    }
    if (!initialModel) return modelToKey(available[0]);
    const exact = available.find((m) => modelToKey(m) === initialModel);
    if (exact) return modelToKey(exact);
    const byName = available.filter((m) => m.name === initialModel);
    if (byName.length > 0) {
      const preferred = byName.find((m) => m.predictorType === "cnn") ?? byName[0];
      return modelToKey(preferred);
    }
    return modelToKey(available[0]);
  }, [initialModel, modelToKey]);

  useEffect(() => {
    sessionPreferredModelKeyRef.current = sessionPreferredModelKey;
  }, [sessionPreferredModelKey]);

  useEffect(() => {
    preferredPredictorTypeRef.current = preferredPredictorType;
  }, [preferredPredictorType]);

  const resetInferenceWorkspaceState = useCallback(() => {
    setImages((prev) => {
      prev.forEach((img) => {
        if (img?.url) URL.revokeObjectURL(img.url);
      });
      return [];
    });
    setCurrentIndex(0);
    setCorrectedSpecimensMap(new Map());
    setEditedImageIndices(new Set());
    setSavedImageIndices(new Set());
    setReviewFinalizedImageIndices(new Set());
    setCommittedImageIndices(new Set());
    setIsReviewStateHydrated(false);
    setSelectedSpecimenIndex(null);
    setDetectionRerunModelKey(null);
    setInferenceRerunModelKey(null);
    setInferProgress(null);
    setIsRunning(false);
    setIsDrawBoxMode(false);
    drawBoxRef.current = null;
  }, []);

  const refreshInferenceHubSessions = useCallback(async () => {
    setLoadingHubSessions(true);
    try {
      const result = await window.api.sessionListInferenceSessions();
      if (!result.ok || !Array.isArray(result.sessions)) {
        setHubSessions([]);
        if (result.error) toast.error(result.error);
        return;
      }
      setHubSessions(result.sessions as InferenceSessionCard[]);
    } catch (err) {
      console.error("Failed to load inference sessions:", err);
      setHubSessions([]);
      toast.error("Failed to load inference sessions.");
    } finally {
      setLoadingHubSessions(false);
    }
  }, []);

  const handleDeleteSession = useCallback(async (speciesId: string) => {
    setDeletingSessionId(speciesId);
    try {
      const result = await window.api.sessionDeleteSchemaSession(speciesId);
      if (!result.ok) {
        toast.error(result.error || "Failed to delete session.");
        return;
      }
      dispatch(removeSpecies(speciesId));
      await refreshInferenceHubSessions();
      toast.success("Session deleted.");
    } catch (err) {
      console.error("Failed to delete session:", err);
      toast.error("Failed to delete session.");
    } finally {
      setDeletingSessionId(null);
    }
  }, [dispatch, refreshInferenceHubSessions]);

  const openInferenceSessionForSchema = useCallback(
    async (speciesId: string) => {
      if (!speciesId) return;
      if (isFirstTimeLocked) {
        toast.error("Create or resume your first schema session in Annotate first.");
        return;
      }
      setOpeningInferenceSessionSpeciesId(speciesId);
      try {
        const res = await window.api.sessionGetInferenceSession(speciesId);
        if (!res.ok) {
          toast.error(res.error || "Failed to open inference session.");
          return;
        }
        if (!res.exists || !res.inferenceSessionId || !res.manifest) {
          toast.error("Create an inference session for this schema first.");
          return;
        }
        dispatch(setActiveSpecies(speciesId));
        setSelectedInferenceSpeciesId(speciesId);
        setInferenceSessionId(res.inferenceSessionId);
        setInferenceSessionManifest(res.manifest as LocalInferenceSessionManifest);
        const preferredModel =
          res.manifest.preferences?.lastUsedLandmarkModelKey ||
          res.manifest.models?.landmark?.key ||
          "";
        setSessionPreferredModelKey(preferredModel);
        setSelectedModelKey("");
        // Pre-fetch images while the view is still transitioning so they're ready sooner
        pendingImageLoadRef.current = window.api.sessionLoadInferenceImagePaths(speciesId, res.inferenceSessionId);
        resetInferenceWorkspaceState();
        setShowInferenceHub(false);
      } catch (err) {
        console.error("Failed to open inference session:", err);
        toast.error("Failed to open inference session.");
      } finally {
        setOpeningInferenceSessionSpeciesId(null);
      }
    },
    [dispatch, isFirstTimeLocked, resetInferenceWorkspaceState]
  );

  const handleCreateInferenceSession = useCallback(async () => {
    if (isFirstTimeLocked) {
      toast.error("Create or resume your first schema session in Annotate first.");
      return;
    }
    const speciesId = createSessionSpeciesId.trim();
    const displayName = createSessionName.trim();
    if (!speciesId) {
      toast.error("Select a schema session.");
      return;
    }
    if (!displayName) {
      toast.error("Enter an inference session name.");
      return;
    }
    try {
      const res = await window.api.sessionCreateInferenceSession(speciesId, displayName);
      if (!res.ok) {
        toast.error(res.error || "Failed to create inference session.");
        return;
      }
      toast.success("Inference session created.");
      setCreateSessionDialogOpen(false);
      await refreshInferenceHubSessions();
      await openInferenceSessionForSchema(speciesId);
    } catch (err) {
      console.error("Failed to create inference session:", err);
      toast.error("Failed to create inference session.");
    }
  }, [
    createSessionName,
    createSessionSpeciesId,
    isFirstTimeLocked,
    openInferenceSessionForSchema,
    refreshInferenceHubSessions,
  ]);

  useEffect(() => {
    if (!showInferenceHub) return;
    void refreshInferenceHubSessions();
  }, [refreshInferenceHubSessions, showInferenceHub]);

  // Bounding-box drag/resize state
  type BoxDragMode = "move" | "resize-tl" | "resize-tr" | "resize-bl" | "resize-br" | "rotate";
  const boxDraggingRef = useRef<{
    specIdx: number;
    mode: BoxDragMode;
    startMouse: { x: number; y: number };
    startBox: {
      left: number; top: number; width: number; height: number;
      obbCorners?: [number, number][];
      cx?: number; cy?: number;
    };
  } | null>(null);
  const [isDrawBoxMode, setIsDrawBoxMode] = useState(false);
  const [drawDefaultOrientation, setDrawDefaultOrientation] = useState<"left" | "right">(() =>
    ((typeof window !== "undefined" && window.localStorage.getItem("bv_draw_default_orientation")) as "left" | "right") ?? "left"
  );
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

  const specimenHasValidBox = useCallback((specimen?: PredictedSpecimen | null): boolean => {
    return Boolean(
      specimen?.box &&
      Number(specimen.box.width) > 0 &&
      Number(specimen.box.height) > 0
    );
  }, []);

  const specimenHasLandmarks = useCallback((specimen?: PredictedSpecimen | null): boolean => {
    return Boolean(Array.isArray(specimen?.landmarks) && specimen.landmarks.length > 0);
  }, []);

  const hasSuccessfulDetectionForImageIndex = useCallback((index: number): boolean => {
    return getSpecimensForImageIndex(index).some((specimen) => specimenHasValidBox(specimen));
  }, [getSpecimensForImageIndex, specimenHasValidBox]);

  const hasSuccessfulLandmarksForImageIndex = useCallback((index: number): boolean => {
    return getSpecimensForImageIndex(index).some(
      (specimen) => specimenHasValidBox(specimen) && specimenHasLandmarks(specimen)
    );
  }, [getSpecimensForImageIndex, specimenHasLandmarks, specimenHasValidBox]);

  const eligibleDetectionIndices = useMemo(
    () => {
      if (selectedModelKey && detectionRerunModelKey === selectedModelKey) {
        return images.map((_, idx) => idx);
      }
      return images
        .map((_, idx) => idx)
        .filter((idx) => !hasSuccessfulDetectionForImageIndex(idx));
    },
    [images, hasSuccessfulDetectionForImageIndex, detectionRerunModelKey, selectedModelKey]
  );

  const eligibleInferenceIndices = useMemo(
    () => {
      const detectedIndices = images
        .map((_, idx) => idx)
        .filter((idx) => hasSuccessfulDetectionForImageIndex(idx));
      if (selectedModelKey && inferenceRerunModelKey === selectedModelKey) {
        return detectedIndices;
      }
      return detectedIndices.filter((idx) => !hasSuccessfulLandmarksForImageIndex(idx));
    },
    [images, hasSuccessfulDetectionForImageIndex, hasSuccessfulLandmarksForImageIndex, inferenceRerunModelKey, selectedModelKey]
  );

  useEffect(() => {
    lastSelectedModelKeyRef.current = selectedModelKey;
  }, [selectedModelKey]);

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
          obbCorners: Array.isArray(s.box.obbCorners) ? s.box.obbCorners : undefined,
          angle: typeof s.box.angle === "number" ? s.box.angle : undefined,
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
      if (!draftSpecimens.length) return inferenceSpecimens;
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

  const hydratePersistedReviewDrafts = useCallback(
    async (targetImages?: InferenceImage[]) => {
      const requestId = reviewHydrationRequestRef.current + 1;
      reviewHydrationRequestRef.current = requestId;
      setIsReviewStateHydrated(false);
      if (!activeSpeciesId || !inferenceSessionId) {
        if (reviewHydrationRequestRef.current === requestId) {
          setIsReviewStateHydrated(true);
        }
        return;
      }
      const sourceImages = targetImages ?? images;
      if (!sourceImages.length) {
        if (reviewHydrationRequestRef.current === requestId) {
          setIsReviewStateHydrated(true);
        }
        return;
      }

      try {
        const [draftResult, finalizedFilenames] = await Promise.all([
          window.api.sessionLoadInferenceReviewDrafts(activeSpeciesId, inferenceSessionId),
          loadPersistedFinalizedFilenames(),
        ]);
        if (reviewHydrationRequestRef.current !== requestId) return;
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
        const nextCommitted = new Set<number>();

        // Mutable copy so we can inject synthetic results for images that have drafts but no live results
        const mutableImages = [...sourceImages];
        let syntheticResultsInjected = false;

        mutableImages.forEach((img, idx) => {
          const draft =
            byPath.get(normalizePathForMatch(img.path)) ??
            byName.get((img.name || "").toLowerCase());
          const finalizedBySession = finalizedFilenames.has((img.name || "").toLowerCase());

          if (!draft) {
            if (finalizedBySession) {
              nextSaved.add(idx);
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
                obbCorners: s.box.obbCorners,
                angle: s.box.angle,
                orientation_override: s.box.orientation_override,
                orientation_hint: s.box.orientation_hint,
              },
              landmarks,
              num_landmarks: landmarks.length,
            };
          });

          const inferredSpecimens = getSpecimensFromResult(img.results);
          const liveInferenceSignature = img.results?.inferenceSignature;
          const liveBoxSignature = img.results?.boxSignature;
          const signaturesMatch = !img.results
            ? true
            : draft.inferenceSignature && liveInferenceSignature
              ? draft.inferenceSignature === liveInferenceSignature
              : draft.boxSignature && liveBoxSignature
                ? draft.boxSignature === liveBoxSignature
                : true;
          if (!signaturesMatch && img.results) {
            if (finalizedBySession) {
              nextSaved.add(idx);
            }
            return;
          }
          const mergedSpecimens = signaturesMatch
            ? mergeDraftLandmarksWithInference(
                convertedDraftSpecimens,
                inferredSpecimens
              )
            : inferredSpecimens;
          const draftIsEdited = Boolean(draft.edited);
          const draftIsSaved = Boolean(draft.saved) && !draftIsEdited;

          nextCorrected.set(idx, mergedSpecimens);
          if ((finalizedBySession || draftIsSaved) && !draftIsEdited) nextSaved.add(idx);
          if (draft.reviewComplete) nextReviewFinalized.add(idx);
          if (draftIsEdited) nextEdited.add(idx);
          const updatedAtMs = Date.parse(String(draft.updatedAt || ""));
          const committedAtMs = Date.parse(String(draft.committedAt || ""));
          if (
            draft.committedAt &&
            Number.isFinite(updatedAtMs) &&
            Number.isFinite(committedAtMs) &&
            committedAtMs >= updatedAtMs
          ) {
            nextCommitted.add(idx);
          }

          // Synthesize an InferenceResult so the canvas renders saved boxes/landmarks
          // immediately without requiring a new backend call
          if (!img.results && mergedSpecimens.length > 0) {
            const syntheticResult = {
              specimens: mergedSpecimens,
              image: img.path,
              image_dimensions: { width: 0, height: 0 },
              num_landmarks: mergedSpecimens.reduce((s: number, sp: PredictedSpecimen) => s + sp.landmarks.length, 0),
              boxSignature: draft.boxSignature,
              inferenceSignature: draft.inferenceSignature,
            };
            mutableImages[idx] = { ...img, results: syntheticResult as InferenceResult };
            syntheticResultsInjected = true;
          }
        });

        if (reviewHydrationRequestRef.current !== requestId) return;
        if (syntheticResultsInjected) {
          setImages(mutableImages);
        }
        setCorrectedSpecimensMap(nextCorrected);
        setEditedImageIndices(nextEdited);
        setSavedImageIndices(nextSaved);
        setReviewFinalizedImageIndices(nextReviewFinalized);
        setCommittedImageIndices(nextCommitted);
      } finally {
        if (reviewHydrationRequestRef.current === requestId) {
          setIsReviewStateHydrated(true);
        }
      }
    },
    [
      activeSpeciesId,
      inferenceSessionId,
      images,
      loadPersistedFinalizedFilenames,
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
        reviewComplete?: boolean;
        committedAt?: string | null;
        landmarkModelKey?: string | null;
        landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose" | null;
        boxSignature?: string | null;
        inferenceSignature?: string | null;
        clear?: boolean;
      }
    ) => {
      if (!activeSpeciesId || !inferenceSessionId) return;
      const image = images[index];
      if (!image) return;

      const specimens = toDraftSpecimens(
        options?.specimens ?? getSpecimensForImageIndex(index)
      );
      const currentResult = image.results;

      const response = await window.api.sessionSaveInferenceReviewDraft(
        activeSpeciesId,
        inferenceSessionId,
        image.path,
        specimens,
        {
          filename: image.name,
          edited: options?.edited ?? editedImageIndices.has(index),
          saved: options?.saved ?? savedImageIndices.has(index),
          reviewComplete:
            options?.reviewComplete ?? reviewFinalizedImageIndices.has(index),
          committedAt: options?.committedAt,
          landmarkModelKey:
            options?.landmarkModelKey === null
              ? null
              : options?.landmarkModelKey ??
                (currentResult?.inferenceSignature
                  ? selectedModelKey || inferenceSessionManifest?.preferences?.lastUsedLandmarkModelKey
                  : undefined),
          landmarkPredictorType:
            options?.landmarkPredictorType === null
              ? null
              : options?.landmarkPredictorType ??
                getSelectedModel()?.predictorType ??
                inferenceSessionManifest?.preferences?.lastUsedPredictorType,
          boxSignature:
            options?.boxSignature === null
              ? null
              : options?.boxSignature ?? currentResult?.boxSignature,
          inferenceSignature:
            options?.inferenceSignature === null
              ? null
              : options?.inferenceSignature ?? currentResult?.inferenceSignature,
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
      getSelectedModel,
      getSpecimensForImageIndex,
      inferenceSessionManifest,
      images,
      reviewFinalizedImageIndices,
      savedImageIndices,
      selectedModelKey,
      toDraftSpecimens,
    ]
  );

  const saveCorrectionsForImage = useCallback(async (index: number): Promise<boolean> => {
    if (!activeSpeciesId || !inferenceSessionId) {
      toast.error("Inference session is not ready yet.");
      return false;
    }
    const image = images[index];
    if (!image?.results) return false;

    const specimensToSave = getSpecimensForImageIndex(index)
      .filter((s) => s?.box && s.box.width > 0 && s.box.height > 0);
    await persistReviewDraft(index, {
      specimens: specimensToSave,
      edited: false,
      saved: true,
      reviewComplete: reviewFinalizedImageIndices.has(index),
    });
    setSavedImageIndices((prev) => {
      const next = new Set(prev);
      next.add(index);
      return next;
    });
    setEditedImageIndices((prev) => {
      const next = new Set(prev);
      next.delete(index);
      return next;
    });
    return true;
  }, [
    activeSpeciesId,
    inferenceSessionId,
    images,
    getSpecimensForImageIndex,
    persistReviewDraft,
    reviewFinalizedImageIndices,
  ]);

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

    // class_id fallback: only used when no hint exists (toggle value as last resort)
    if (box.class_id === 0) return "left";
    if (box.class_id === 1) return "right";

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
    (specimenIndex: number, orientation: "left" | "right") => {
      const current = getSpecimensForImageIndex(currentIndex);
      if (specimenIndex < 0 || specimenIndex >= current.length) return;
      const updated = JSON.parse(JSON.stringify(current)) as PredictedSpecimen[];
      const target = updated[specimenIndex];
      if (!target?.box) return;

      target.box.orientation_override = orientation;
      target.box.class_id = orientation === "left" ? 0 : 1;
      target.box.orientation_hint = {
        orientation,
        confidence: 1.0,
        source: "user_review",
      };

      liveSpecimensRef.current = updated;
      setCorrectedSpecimensMap((prev) => {
        const next = new Map(prev);
        next.set(currentIndex, updated);
        return next;
      });
      markImageEdited(currentIndex);
      void persistReviewDraft(currentIndex, {
        specimens: updated,
        edited: true,
        saved: false,
        committedAt: null,
      });
    },
    [currentIndex, getSpecimensForImageIndex, markImageEdited, persistReviewDraft]
  );

  useEffect(() => {
    if (!activeSpeciesId || !inferenceSessionId || images.length === 0) return;
    void hydratePersistedReviewDrafts(images);
  }, [activeSpeciesId, inferenceSessionId, hydratePersistedReviewDrafts]);

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

    if (showBoundingBox && specimens.length > 0) {
      const handleR = Math.max(5, diagonal * 0.008);
      specimens.forEach((specimen, idx) => {
        const box = specimen.box;
        if (!box || box.width <= 0 || box.height <= 0) return;
        const color = palette[idx % palette.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = idx === selectedSpecimenIndex ? lineWidth * 3 : lineWidth * 2;
        ctx.setLineDash([10, 5]);
        const obbPts = Array.isArray(box.obbCorners) ? box.obbCorners : undefined;
        if (obbPts && obbPts.length === 4) {
          ctx.beginPath();
          ctx.moveTo(obbPts[0][0], obbPts[0][1]);
          for (let k = 1; k < 4; k++) ctx.lineTo(obbPts[k][0], obbPts[k][1]);
          ctx.closePath();
          ctx.stroke();
        } else {
          ctx.strokeRect(box.left, box.top, box.width, box.height);
        }
        ctx.setLineDash([]);

        // Orientation arrow inside box from OBB geometry.
        // Use the same resolver as labels/review controls so UI state and arrow never diverge.
        const orientationLabel = resolveOrientationLabelFromBox(specimen?.box);
        const arrowPts: [number, number][] =
          obbPts && obbPts.length === 4
            ? (obbPts as [number, number][])
            : ([
                [box.left, box.top],
                [box.left + box.width, box.top],
                [box.left + box.width, box.top + box.height],
                [box.left, box.top + box.height],
              ] as [number, number][]);
        if ((activeOrientationMode === "directional" || activeOrientationMode === "bilateral") && (orientationLabel === "left" || orientationLabel === "right")) {
          const [op0, op1, op2, op3] = arrowPts;
          const oIsLeft = orientationLabel === "left";
          const isBilateral = activeOrientationMode === "bilateral";
          // Canonical edge midpoints (from buildObbCorners order):
          // LEFT=cp3â†’cp0, RIGHT=cp1â†’cp2, TOP=cp0â†’cp1, BOTTOM=cp2â†’cp3
          let emX: number, emY: number;
          if (!isBilateral) {
            [emX, emY] = oIsLeft
              ? [(op3[0]+op0[0])/2, (op3[1]+op0[1])/2]
              : [(op1[0]+op2[0])/2, (op1[1]+op2[1])/2];
          } else {
            [emX, emY] = oIsLeft
              ? [(op0[0]+op1[0])/2, (op0[1]+op1[1])/2]
              : [(op2[0]+op3[0])/2, (op2[1]+op3[1])/2];
          }
          const ocx = (op0[0]+op1[0]+op2[0]+op3[0])/4;
          const ocy = (op0[1]+op1[1]+op2[1]+op3[1])/4;
          const outLen = Math.hypot(emX-ocx, emY-ocy) || 1;
          const oNx = (emX-ocx)/outLen, oNy = (emY-ocy)/outLen;
          const arrowLen = Math.min(Math.max(outLen * 0.6, 14), 40);
          const tipX = emX + oNx * 4, tipY = emY + oNy * 4;
          const tailX = emX - oNx * arrowLen, tailY = emY - oNy * arrowLen;
          const fullLen = Math.hypot(tipX-tailX, tipY-tailY);
          if (fullLen >= 24) {
            const oHSizeLen = Math.min(Math.max(fullLen * 0.20, 6), 10);
            const oHSizeW   = Math.min(Math.max(fullLen * 0.15, 5), 8);
            ctx.save();
            ctx.globalAlpha = 0.85;
            ctx.strokeStyle = color; ctx.fillStyle = color;
            ctx.lineWidth = Math.max(1.5, lineWidth * 0.85);
            ctx.beginPath();
            ctx.moveTo(tailX, tailY);
            ctx.lineTo(tipX - oNx*oHSizeLen, tipY - oNy*oHSizeLen);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(tipX, tipY);
            ctx.lineTo(tipX - oNx*oHSizeLen + (-oNy)*oHSizeW*0.5, tipY - oNy*oHSizeLen + oNx*oHSizeW*0.5);
            ctx.lineTo(tipX - oNx*oHSizeLen - (-oNy)*oHSizeW*0.5, tipY - oNy*oHSizeLen - oNx*oHSizeW*0.5);
            ctx.closePath(); ctx.fill();
            ctx.restore();
          }
        }

        ctx.fillStyle = color;
        ctx.font = `bold ${fontSize}px sans-serif`;
        // Compute tilt angle from OBB long axis, folded to [0ï¿½, 90ï¿½]
        let angleSuffix = "";
        const isInvariant = activeOrientationMode === "invariant";
        if (!isInvariant && obbPts && obbPts.length === 4) {
          const [ap0, ap1, , ap3] = obbPts;
          const adx1 = ap1[0]-ap0[0], ady1 = ap1[1]-ap0[1];
          const adx3 = ap3[0]-ap0[0], ady3 = ap3[1]-ap0[1];
          const [aldx, aldy] = Math.hypot(adx1,ady1) >= Math.hypot(adx3,ady3) ? [adx1,ady1] : [adx3,ady3];
          let adeg = Math.atan2(aldy, aldx) * 180 / Math.PI;
          adeg = ((adeg % 180) + 180) % 180;
          if (adeg > 90) adeg = 180 - adeg;
          angleSuffix = ` \u00B7 ${Math.round(adeg)}\u00B0`;
        }
        const isVectorOrUnset = !activeOrientationMode || activeOrientationMode === "directional" || activeOrientationMode === "bilateral";
        const orientationSuffix = isVectorOrUnset
          ? ` \u00B7 ${orientationLabel}${angleSuffix}`
          : angleSuffix; // axial: angle only; invariant: empty (angleSuffix is "")
        const label = idx === selectedSpecimenIndex
          ? `Specimen ${idx + 1} (selected)${orientationSuffix}`
          : `Specimen ${idx + 1}${orientationSuffix}`;
        ctx.fillText(label, box.left + 5, box.top - 5);

        // Corner resize handles
        const corners = obbPts && obbPts.length === 4
          ? obbPts.map(([x, y]) => ({ x, y }))
          : [
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

        // Rotation handle ï¿½ circle above box top-center
        const rotHandleR = Math.max(7, diagonal * 0.009);
        const rotHandleX = box.left + box.width / 2;
        const rotHandleY = Math.max(box.top - 30, rotHandleR + 4);
        ctx.save();
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(rotHandleX, box.top);
        ctx.lineTo(rotHandleX, rotHandleY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(rotHandleX, rotHandleY, rotHandleR, 0, Math.PI * 2);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.lineWidth = lineWidth * 1.5;
        ctx.strokeStyle = color;
        ctx.stroke();
        ctx.restore();
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
        // Real-time orientation arrow in the preview box
        const previewObbPts: [number,number][] = [[x,y],[x+w,y],[x+w,y+h],[x,y+h]];
        const [dp0,dp1,dp2,dp3] = previewObbPts;
        const dLen01 = Math.hypot(dp1[0]-dp0[0], dp1[1]-dp0[1]);
        const dLen12 = Math.hypot(dp2[0]-dp1[0], dp2[1]-dp1[1]);
        let dMidA: [number,number], dMidB: [number,number];
        if (dLen01 >= dLen12) {
          dMidA = [(dp0[0]+dp3[0])/2, (dp0[1]+dp3[1])/2];
          dMidB = [(dp1[0]+dp2[0])/2, (dp1[1]+dp2[1])/2];
        } else {
          dMidA = [(dp0[0]+dp1[0])/2, (dp0[1]+dp1[1])/2];
          dMidB = [(dp2[0]+dp3[0])/2, (dp2[1]+dp3[1])/2];
        }
        const dIsLeft = drawDefaultOrientation === "left";
        const [dLeftEnd, dRightEnd] = dMidA[0] <= dMidB[0] ? [dMidA, dMidB] : [dMidB, dMidA];
        const dHead = dIsLeft ? dLeftEnd : dRightEnd;
        const dTail = dIsLeft ? dRightEnd : dLeftEnd;
        const dAxisLen = Math.hypot(dHead[0]-dTail[0], dHead[1]-dTail[1]) || 1;
        if (dAxisLen >= 24) {
          const dNx = (dHead[0]-dTail[0])/dAxisLen, dNy = (dHead[1]-dTail[1])/dAxisLen;
          const arrowLen  = Math.min(Math.max(dAxisLen * 0.25, 14), 32);
          const dHSizeLen = Math.min(Math.max(dAxisLen * 0.10, 6), 10);
          const dHSizeW   = Math.min(Math.max(dAxisLen * 0.07, 5), 8);
          const tipX = dHead[0], tipY = dHead[1];
          const tailX = dHead[0] - dNx * arrowLen, tailY = dHead[1] - dNy * arrowLen;
          ctx.strokeStyle = "rgba(34, 197, 94, 0.95)";
          ctx.fillStyle = "rgba(34, 197, 94, 0.95)";
          ctx.lineWidth = Math.max(1.5, lineWidth * 0.85);
          ctx.beginPath();
          ctx.moveTo(tailX, tailY);
          ctx.lineTo(tipX - dNx*dHSizeLen, tipY - dNy*dHSizeLen);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(tipX, tipY);
          ctx.lineTo(tipX - dNx*dHSizeLen + (-dNy)*dHSizeW*0.5, tipY - dNy*dHSizeLen + dNx*dHSizeW*0.5);
          ctx.lineTo(tipX - dNx*dHSizeLen - (-dNy)*dHSizeW*0.5, tipY - dNy*dHSizeLen - dNx*dHSizeW*0.5);
          ctx.closePath(); ctx.fill();
        }
        ctx.restore();
      }
    }
  }, [showBoundingBox, selectedSpecimenIndex, resolveOrientationLabelFromBox, drawDefaultOrientation, activeOrientationMode]);

  // Sync effectiveSessionId with the globally active species on first mount
  useEffect(() => {
    if (activeSpeciesId && selectedInferenceSpeciesId === null) {
      setSelectedInferenceSpeciesId(activeSpeciesId);
    }
  }, [activeSpeciesId, selectedInferenceSpeciesId]);

  // Load models on mount (or when effective session changes)
  useEffect(() => {
    const loadModels = async () => {
      setLoadingModels(true);
      if (!effectiveSessionId) {
        setModels([]);
        setSelectedModelKey("");
        setLoadingModels(false);
        return;
      }
      try {
        const result = await window.api.listModels({
          speciesId: effectiveSessionId,
          activeOnly: false,
          includeDeprecated: true,
        });
        if (result.ok && result.models) {
          const allModels = result.models.filter(
            (m) => (m.predictorType ?? "dlib") === "dlib" || m.predictorType === "cnn"
          );
          const available = allModels.filter((m) => {
            if (m.status && m.status !== "active") return false;
            if (m.predictorType === "cnn" && m.compatible === false) return false;
            return true;
          });
          if (sessionPreferredModelKey) {
            const persisted = allModels.find((m) => modelToKey(m) === sessionPreferredModelKey);
            if (persisted?.predictorType === "cnn" && persisted.compatible === false) {
              const message =
                persisted.reason ||
                "This CNN model predates the heatmap-head format and must be retrained.";
              if (incompatiblePreferenceNoticeRef.current !== sessionPreferredModelKey) {
                incompatiblePreferenceNoticeRef.current = sessionPreferredModelKey;
                toast.error(message);
              }
            }
          }
          setModels(available);
          setSelectedModelKey((prev) => {
            if (prev && available.some((m) => modelToKey(m) === prev)) {
              return prev;
            }
            return resolveInitialModelKey(available);
          });
          if (available.length === 0 && allModels.some((m) => m.predictorType === "cnn" && m.compatible === false)) {
            toast.error("No compatible CNN landmark model is available. Retrain the CNN model to use inference.");
          }
        }
      } catch (err) {
        console.error("Failed to load models:", err);
        toast.error("Failed to load models");
      } finally {
        setLoadingModels(false);
      }
    };
    loadModels();
  }, [effectiveSessionId, modelToKey, resolveInitialModelKey]);

  useEffect(() => {
    if (!initialModel || models.length === 0) return;
    const exact = models.find((m) => modelToKey(m) === initialModel);
    if (exact) {
      setSelectedModelKey(modelToKey(exact));
      return;
    }
    const byName = models.filter((m) => m.name === initialModel);
    if (byName.length > 0) {
      const preferred = byName.find((m) => m.predictorType === "cnn") ?? byName[0];
      setSelectedModelKey(modelToKey(preferred));
    }
  }, [initialModel, models, modelToKey]);

  useEffect(() => {
    if (!activeSpeciesId || !inferenceSessionId || !selectedModelKey) return;
    const selectedModel = getSelectedModel();
    if (!selectedModel) return;
    void window.api.sessionUpdateInferenceSessionPreferences(
      activeSpeciesId,
      inferenceSessionId,
      {
        preferences: {
          lastUsedLandmarkModelKey: selectedModelKey,
          lastUsedPredictorType: selectedModel.predictorType ?? "dlib",
        },
      }
    ).then((response) => {
      if (response?.ok && response.manifest) {
        setInferenceSessionManifest(response.manifest as LocalInferenceSessionManifest);
        setSessionPreferredModelKey(selectedModelKey);
      }
    });
  }, [activeSpeciesId, getSelectedModel, inferenceSessionId, selectedModelKey]);

  // Auto-load persisted image list when session opens and no images are loaded yet
  useEffect(() => {
    if (!activeSpeciesId || !inferenceSessionId || images.length > 0) return;
    (async () => {
      try {
        setIsReviewStateHydrated(false);
        const loadPromise = pendingImageLoadRef.current ?? window.api.sessionLoadInferenceImagePaths(activeSpeciesId, inferenceSessionId);
        pendingImageLoadRef.current = null;
        const res = await loadPromise;
        if (!res.ok || !res.images || res.images.length === 0) {
          setIsReviewStateHydrated(true);
          return;
        }
        const loaded: InferenceImage[] = res.images.map((f) => {
          const bytes = Uint8Array.from(atob(f.data), (c) => c.charCodeAt(0));
          const url = URL.createObjectURL(new Blob([bytes], { type: f.mimeType }));
          return { path: f.path, name: f.name, url };
        });
        setImages(loaded);
        await hydratePersistedReviewDrafts(loaded);
      } catch {
        setIsReviewStateHydrated(true);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSpeciesId, inferenceSessionId]);

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
        const existingPaths = new Set(images.map((img) => img.path));
        const deduped = newImages.filter((img) => !existingPaths.has(img.path));
        if (deduped.length < newImages.length) {
          toast.info(`${newImages.length - deduped.length} duplicate image(s) skipped.`);
        }
        const combinedImages = [...images, ...deduped];
        setImages(combinedImages);
        await hydratePersistedReviewDrafts(combinedImages);
        if (inferenceSessionId) {
          void window.api.sessionSaveInferenceImagePaths(
            activeSpeciesId,
            inferenceSessionId,
            combinedImages.map((img) => ({ path: img.path, name: img.name }))
          );
        }
        if (deduped.length > 0) toast.success(`Added ${deduped.length} image(s)`);
      }
    } catch (err) {
      console.error("Failed to select images:", err);
      toast.error("Failed to select images");
    }
  };

  const handleRemoveImage = (index: number) => {
    // Clear persisted draft so boxes/landmarks don't reappear if the same image is re-added later.
    void persistReviewDraft(index, { clear: true });
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
      if (activeSpeciesId && inferenceSessionId) {
        void window.api.sessionSaveInferenceImagePaths(
          activeSpeciesId,
          inferenceSessionId,
          next.map((img) => ({ path: img.path, name: img.name }))
        );
      }
      return next;
    });
    setCorrectedSpecimensMap((prev) => shiftIndicesInMap(prev, index));
    setEditedImageIndices((prev) => shiftIndicesInSet(prev, index));
    setSavedImageIndices((prev) => shiftIndicesInSet(prev, index));
    setReviewFinalizedImageIndices((prev) => shiftIndicesInSet(prev, index));
    setCommittedImageIndices((prev) => shiftIndicesInSet(prev, index));
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
    const targetIndices = eligibleDetectionIndices;
    if (targetIndices.length === 0) {
      toast.message("No eligible images for detection. Only unseen or failed images are processed.");
      return;
    }

    setIsRunning(true);
    setIsDrawBoxMode(false);
    setShowMaskOverlay(false);
    drawBoxRef.current = null;
    setInferProgress({ percent: 0, stage: "detecting" });

    let successCount = 0;
    let errorCount = 0;
    const updatedImages = [...images];
    try {
      for (let processed = 0; processed < targetIndices.length; processed++) {
        const i = targetIndices[processed];
        const img = images[i];
        setInferProgress({
          percent: Math.round((processed / Math.max(targetIndices.length, 1)) * 100),
          stage: "detecting",
        });
        try {
          const detectResult = await window.api.detectSpecimens(img.path, {
            speciesId: activeSpeciesId,
          });
          if (detectResult.ok && Array.isArray(detectResult.boxes)) {
            const boxes = (detectResult.boxes as Array<DetectedBox & { obbCorners?: [number, number][]; angle?: number }>)
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
                obbCorners: Array.isArray(b.obbCorners) ? b.obbCorners as [number, number][] : undefined,
                angle: typeof b.angle === "number" ? b.angle : undefined,
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
            const stampedDetectionResult = stampInferenceResult(
              detectionResult,
              img.path,
              boxes
            );
            updatedImages[i] = { ...updatedImages[i], results: stampedDetectionResult, error: undefined };
            successCount++;
            // Auto-save detected boxes as draft so they persist across sessions
            await persistReviewDraft(i, {
              specimens: stampedDetectionResult.specimens as PredictedSpecimen[],
              edited: false,
              saved: false,
              committedAt: null,
              boxSignature: stampedDetectionResult.boxSignature,
              inferenceSignature: null,
              landmarkModelKey: null,
              landmarkPredictorType: null,
            });
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
      if (selectedModelKey && detectionRerunModelKey === selectedModelKey) {
        setDetectionRerunModelKey(null);
      }
      if (selectedModelKey) {
        setInferenceRerunModelKey(selectedModelKey);
      }
      if (successCount > 0) {
        toast.success(`Detection complete: ${successCount} succeeded, ${errorCount} failed`);
      } else {
        toast.error("All eligible detections failed");
      }
    } finally {
      setInferProgress(null);
      setIsRunning(false);
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

    const hasAnyDetectionBoxes = images.some((_, idx) => hasSuccessfulDetectionForImageIndex(idx));
    if (!hasAnyDetectionBoxes) {
      toast.error("Run detection and keep at least one box before landmark inference.");
      return;
    }
    const targetIndices = eligibleInferenceIndices;
    if (targetIndices.length === 0) {
      toast.message("No eligible images for landmark inference. Only unseen or failed images are processed.");
      return;
    }

    setIsRunning(true);
    setInferProgress({ percent: 2, stage: "checking_compatibility", currentIndex: 0, total: targetIndices.length });

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
      setInferProgress(null);
      setIsRunning(false);
      return;
    }

    setInferProgress({ percent: 8, stage: "preparing_worker", currentIndex: 0, total: targetIndices.length });
    const unsubscribeProgress = window.api.onPredictProgress((data) => {
      setInferProgress(data);
    });

    const updatedImages = [...images];
    let successCount = 0;
    let errorCount = 0;
    try {
      const batchItems: Array<{
        batchIndex: number;
        imagePath: string;
        filename: string;
        boxes: Array<{
          left: number;
          top: number;
          width: number;
          height: number;
          right: number;
          bottom: number;
          obbCorners?: [number, number][];
          angle?: number;
          class_id?: number;
          orientation_hint?: {
            orientation?: "left" | "right";
            confidence?: number;
            source?: string;
          };
        }>;
      }> = [];

      for (let processed = 0; processed < targetIndices.length; processed++) {
        const i = targetIndices[processed];
        const img = images[i];
        const preInferenceSpecs = getSpecimensForImageIndex(i)
          .filter((s) => specimenHasValidBox(s));
        const acceptedBoxes = preInferenceSpecs.map((s) => ({
          left: Math.round(s.box.left),
          top: Math.round(s.box.top),
          width: Math.round(s.box.width),
          height: Math.round(s.box.height),
          right: Math.round(s.box.left + s.box.width),
          bottom: Math.round(s.box.top + s.box.height),
          ...(Array.isArray(s.box.obbCorners) && s.box.obbCorners.length === 4
            ? { obbCorners: s.box.obbCorners }
            : {}),
          ...(typeof s.box.angle === "number" ? { angle: s.box.angle } : {}),
          ...(s.box.class_id != null ? { class_id: s.box.class_id } : {}),
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
        batchItems.push({
          batchIndex: i,
          imagePath: img.path,
          filename: img.name,
          boxes: acceptedBoxes,
        });
      }

      if (batchItems.length > 0) {
        const result = await window.api.predictImagesBatch({
          speciesId: activeSpeciesId,
          modelName: selectedModel.name,
          predictorType: selectedPredictor,
          allowIncompatible,
          items: batchItems,
        });

        if (!result.ok || !Array.isArray(result.results)) {
          const batchError = result.error || "Landmark inference failed";
          for (const item of batchItems) {
            const targetImage = updatedImages[item.batchIndex];
            if (!targetImage) continue;
            updatedImages[item.batchIndex] = {
              ...targetImage,
              error: batchError,
            };
            errorCount++;
          }
        } else {
          for (const item of result.results) {
            const i = Number(item.batchIndex);
            const targetImage = updatedImages[i];
            const batchItem = batchItems.find((entry) => entry.batchIndex === i);
            if (!targetImage || !batchItem) continue;

            if (item.ok && item.data) {
              const boxSignature = getStableBoxSignature(targetImage.path, batchItem.boxes);
              const inferenceSignature = getInferenceSignature(
                selectedModelKey,
                selectedPredictor,
                boxSignature
              );
              const stampedResult = stampInferenceResult(
                item.data as InferenceResult,
                targetImage.path,
                batchItem.boxes,
                {
                  selectedModelKey,
                  predictorType: selectedPredictor,
                }
              );
              updatedImages[i] = { ...targetImage, results: stampedResult, error: undefined };
              successCount++;
              const specsToSave = (stampedResult.specimens ?? []) as unknown as PredictedSpecimen[];
              await persistReviewDraft(i, {
                specimens: specsToSave,
                edited: false,
                saved: false,
                committedAt: null,
                landmarkModelKey: selectedModelKey,
                landmarkPredictorType: selectedPredictor,
                boxSignature,
                inferenceSignature,
              });
            } else {
              updatedImages[i] = {
                ...targetImage,
                error: item.error || "Landmark inference failed",
              };
              errorCount++;
            }
          }
        }
      }

      setImages(updatedImages);
      await hydratePersistedReviewDrafts(updatedImages);
      if (selectedModelKey && inferenceRerunModelKey === selectedModelKey) {
        setInferenceRerunModelKey(null);
      }
      if (successCount > 0) {
        toast.success(`Landmark inference complete: ${successCount} succeeded, ${errorCount} failed`);
      } else {
        toast.error("All eligible landmark inferences failed");
      }
    } finally {
      unsubscribeProgress();
      setInferProgress(null);
      setIsRunning(false);
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

  const handleCommitReviewComplete = useCallback(async () => {
    if (!activeSpeciesId || !inferenceSessionId) {
      toast.error("Inference session is not ready yet.");
      return;
    }
    const reviewCompleteTargets = images
      .map((img, idx) => ({ img, idx }))
      .filter(({ img, idx }) => Boolean(img.results) && reviewFinalizedImageIndices.has(idx))
      .map(({ idx }) => idx);
    if (reviewCompleteTargets.length === 0) {
      toast.message("No review-complete images available to commit.");
      return;
    }

    setIsCommittingReview(true);
    try {
      // Flush only edited targets; untouched drafts must keep their original updatedAt
      // so commit idempotency can skip already-committed unchanged items.
      const editedTargets = reviewCompleteTargets.filter((idx) =>
        editedImageIndices.has(idx)
      );
      for (const idx of editedTargets) {
        await persistReviewDraft(idx, {
          specimens: getSpecimensForImageIndex(idx),
          edited: true,
          saved: savedImageIndices.has(idx),
          reviewComplete: true,
        });
      }
      const commitRes = await window.api.sessionCommitInferenceReview(
        activeSpeciesId,
        inferenceSessionId,
        { onlyReviewComplete: true }
      );
      if (!commitRes.ok) {
        toast.error(commitRes.error || "Failed to commit review-complete data.");
        return;
      }
      const committed = Number(commitRes.committed || 0);
      const skipped = Number(commitRes.skipped || 0);
      const failed = Number(commitRes.failed || 0);
      await hydratePersistedReviewDrafts(images);
      if (committed > 0) {
        toast.success(
          failed > 0
            ? `Committed ${committed} image(s), ${failed} failed, ${skipped} skipped.`
            : `Committed ${committed} image(s) to training data.`
        );
      } else if (skipped > 0) {
        toast.message(`No new changes to commit (${skipped} already up-to-date).`);
      } else {
        toast.error("No review data was committed.");
      }
    } finally {
      setIsCommittingReview(false);
    }
  }, [
    activeSpeciesId,
    inferenceSessionId,
    editedImageIndices,
    getSpecimensForImageIndex,
    hydratePersistedReviewDrafts,
    images,
    persistReviewDraft,
    reviewFinalizedImageIndices,
    savedImageIndices,
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
    void persistReviewDraft(currentIndex, {
      specimens: cloned,
      edited: true,
      saved: false,
      committedAt: null,
    });
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

  // Canvas coordinate mapping: CSS coordinates -> canvas pixel coordinates
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

    // 2) Box interactions: rotation handle, corner handles (resize), then box interior (move)
    if (showBoundingBox) {
      const rotHandleR = Math.max(7, diagonal * 0.009);

      // 2a) Rotation handle detection (priority over corner handles)
      for (let si = 0; si < specs.length; si++) {
        const box = specs[si].box;
        if (!box || box.width <= 0 || box.height <= 0) continue;
        const rhx = box.left + box.width / 2;
        const rhy = Math.max(box.top - 30, rotHandleR + 4);
        if (Math.sqrt((x - rhx) ** 2 + (y - rhy) ** 2) <= rotHandleR + 4) {
          setSelectedSpecimenIndex(si);
          const rawObb = Array.isArray(box.obbCorners) && box.obbCorners.length === 4
            ? box.obbCorners as [number, number][]
            : [[box.left, box.top], [box.left + box.width, box.top],
               [box.left + box.width, box.top + box.height], [box.left, box.top + box.height]] as [number, number][];
          const cx = rawObb.reduce((s, p) => s + p[0], 0) / 4;
          const cy = rawObb.reduce((s, p) => s + p[1], 0) / 4;
          boxDraggingRef.current = {
            specIdx: si,
            mode: "rotate",
            startMouse: { x, y },
            startBox: {
              left: box.left, top: box.top, width: box.width, height: box.height,
              obbCorners: rawObb.map(p => [...p] as [number, number]),
              cx, cy,
            },
          };
          e.currentTarget.style.cursor = "grab";
          return;
        }
      }

      for (let si = 0; si < specs.length; si++) {
        const box = specs[si].box;
        if (!box || box.width <= 0 || box.height <= 0) continue;

        // Use OBB corner positions when available (matching drawToCanvas)
        const obbPts = Array.isArray(box.obbCorners) && box.obbCorners.length === 4
          ? box.obbCorners as [number, number][]
          : null;
        const cornerModes: BoxDragMode[] = ["resize-tl", "resize-tr", "resize-br", "resize-bl"];
        const corners: { cx: number; cy: number; mode: BoxDragMode }[] = obbPts
          ? obbPts.map(([px, py], i) => ({ cx: px, cy: py, mode: cornerModes[i] }))
          : [
              { cx: box.left,             cy: box.top,              mode: "resize-tl" },
              { cx: box.left + box.width, cy: box.top,              mode: "resize-tr" },
              { cx: box.left + box.width, cy: box.top + box.height, mode: "resize-br" },
              { cx: box.left,             cy: box.top + box.height, mode: "resize-bl" },
            ];

        for (const corner of corners) {
          if (Math.abs(x - corner.cx) <= boxHitRadius && Math.abs(y - corner.cy) <= boxHitRadius) {
            setSelectedSpecimenIndex(si);
            boxDraggingRef.current = {
              specIdx: si,
              mode: corner.mode,
              startMouse: { x, y },
              startBox: {
                left: box.left, top: box.top, width: box.width, height: box.height,
                obbCorners: obbPts ? obbPts.map(p => [...p] as [number, number]) : undefined,
              },
            };
            const cursors: Record<BoxDragMode, string> = {
              "resize-tl": "nw-resize", "resize-tr": "ne-resize",
              "resize-bl": "sw-resize", "resize-br": "se-resize",
              move: "move", rotate: "grab",
            };
            e.currentTarget.style.cursor = cursors[corner.mode];
            return;
          }
        }

        // Box interior -> move
        if (
          x >= box.left && x <= box.left + box.width &&
          y >= box.top  && y <= box.top  + box.height
        ) {
          setSelectedSpecimenIndex(si);
          boxDraggingRef.current = {
            specIdx: si,
            mode: "move",
            startMouse: { x, y },
            startBox: {
              left: box.left, top: box.top, width: box.width, height: box.height,
              obbCorners: obbPts ? obbPts.map(p => [...p] as [number, number]) : undefined,
            },
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

      if (mode === "rotate") {
        const { cx = 0, cy = 0, obbCorners: initCorners } = startBox;
        if (initCorners && initCorners.length === 4) {
          const startAngle = Math.atan2(startMouse.y - cy, startMouse.x - cx);
          const currentAngle = Math.atan2(y - cy, x - cx);
          const delta = currentAngle - startAngle;
          const cos = Math.cos(delta);
          const sin = Math.sin(delta);
          const newCorners: [number, number][] = initCorners.map(([px, py]) => {
            const ddx = px - cx;
            const ddy = py - cy;
            return [cx + ddx * cos - ddy * sin, cy + ddx * sin + ddy * cos];
          });
          box.obbCorners = newCorners;
          const xs = newCorners.map(p => p[0]);
          const ys = newCorners.map(p => p[1]);
          box.left = Math.min(...xs);  box.top = Math.min(...ys);
          box.right = Math.max(...xs); box.bottom = Math.max(...ys);
          box.width = box.right - box.left; box.height = box.bottom - box.top;
        }
      } else if (mode === "move") {
        box.left  = startBox.left  + dx;
        box.top   = startBox.top   + dy;
        if (startBox.obbCorners) {
          box.obbCorners = startBox.obbCorners.map(([px, py]) => [px + dx, py + dy] as [number, number]);
        }
      } else if (mode === "resize-tl" || mode === "resize-tr" || mode === "resize-bl" || mode === "resize-br") {
        const modeToIdx: Record<string, number> = { "resize-tl": 0, "resize-tr": 1, "resize-br": 2, "resize-bl": 3 };
        const dragIdx = modeToIdx[mode];
        if (startBox.obbCorners && startBox.obbCorners.length === 4) {
          // OBB-preserving resize: keep opposite corner fixed, project mouse onto local axes
          const corners: [number, number][] = startBox.obbCorners.map(p => [p[0], p[1]]);
          const fixedIdx = (dragIdx + 2) % 4;
          const adj1Idx  = (dragIdx + 1) % 4;
          const adj2Idx  = (dragIdx + 3) % 4;
          const [fx, fy] = corners[fixedIdx];
          // Local axis vectors (from fixed corner to each adjacent corner)
          const [a1x, a1y] = corners[adj1Idx];
          const [a2x, a2y] = corners[adj2Idx];
          const u1x = a1x - fx, u1y = a1y - fy;
          const u2x = a2x - fx, u2y = a2y - fy;
          const len1 = Math.sqrt(u1x * u1x + u1y * u1y) || 1;
          const len2 = Math.sqrt(u2x * u2x + u2y * u2y) || 1;
          const n1x = u1x / len1, n1y = u1y / len1;
          const n2x = u2x / len2, n2y = u2y / len2;
          // Project mouse position onto each axis (relative to fixed corner)
          const mx = x - fx, my = y - fy;
          const proj1 = Math.max(MIN_SIZE, mx * n1x + my * n1y);
          const proj2 = Math.max(MIN_SIZE, mx * n2x + my * n2y);
          const newCorners: [number, number][] = [...corners];
          newCorners[adj1Idx] = [fx + n1x * proj1, fy + n1y * proj1];
          newCorners[adj2Idx] = [fx + n2x * proj2, fy + n2y * proj2];
          newCorners[dragIdx] = [fx + n1x * proj1 + n2x * proj2, fy + n1y * proj1 + n2y * proj2];
          newCorners[fixedIdx] = [fx, fy];
          box.obbCorners = newCorners;
          const xs = newCorners.map(p => p[0]);
          const ys = newCorners.map(p => p[1]);
          box.left = Math.min(...xs);  box.top = Math.min(...ys);
          box.right = Math.max(...xs); box.bottom = Math.max(...ys);
          box.width = box.right - box.left; box.height = box.bottom - box.top;
        } else {
          // Legacy non-OBB resize path
          box.obbCorners = undefined; box.angle = undefined;
          if (mode === "resize-tl") {
            const newW = Math.max(MIN_SIZE, startBox.width  - dx);
            const newH = Math.max(MIN_SIZE, startBox.height - dy);
            box.left  = startBox.left  + (startBox.width  - newW);
            box.top   = startBox.top   + (startBox.height - newH);
            box.width  = newW;
            box.height = newH;
          } else if (mode === "resize-tr") {
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
        }
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
            obbCorners: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            angle: 0,
            orientation_override: drawDefaultOrientation,
            orientation_hint: {
              orientation: drawDefaultOrientation,
              confidence: 1.0,
              source: "user_draw_default",
            },
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
        void persistReviewDraft(currentIndex, {
          specimens: updated,
          edited: true,
          saved: false,
          committedAt: null,
        });
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
      void persistReviewDraft(currentIndex, {
        specimens: updated,
        edited: true,
        saved: false,
        committedAt: null,
      });
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
    imageRef.current = img;

    // Must be attached before img.src — local file:// images load synchronously,
    // so attaching onload after src means the callback is never called for cached images.
    const onImageLoad = () => {
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

    img.onload = onImageLoad;
    img.src = currentImage.url;
    // For already-cached images img.complete is true synchronously; fire the handler manually.
    if (img.complete) onImageLoad();
  }, [currentIndex, images, showBoundingBox, correctedSpecimensMap, drawToCanvas]);

  const currentImage = images[currentIndex];
  const currentSpecimens = getSpecimensForImageIndex(currentIndex);
  const currentSpecimenCount = currentSpecimens.length;
  const hasInferenceResults = images.some((img) => Boolean(img.results));
  const eligibleDetectionCount = eligibleDetectionIndices.length;
  const eligibleInferenceCount = eligibleInferenceIndices.length;
  const detectionButtonTitle = !effectiveSessionId
    ? "Open an inference session first"
    : !isReviewStateHydrated
      ? "Restoring persisted review state..."
    : selectedModelKey && detectionRerunModelKey === selectedModelKey
      ? `Rerun detection for all ${images.length} image${images.length === 1 ? "" : "s"} after switching models`
    : eligibleDetectionCount > 0
      ? `Run detection on ${eligibleDetectionCount} unseen or failed image${eligibleDetectionCount === 1 ? "" : "s"}`
      : "All images already have detection results";
  const inferenceButtonTitle = !effectiveSessionId
    ? "Open an inference session first"
    : !isReviewStateHydrated
      ? "Restoring persisted review state..."
    : !selectedModelKey
      ? "Select a model first"
      : inferenceRerunModelKey === selectedModelKey
        ? `Rerun landmark inference for ${eligibleInferenceCount} detected image${eligibleInferenceCount === 1 ? "" : "s"} with the newly selected model`
      : eligibleInferenceCount > 0
        ? `Run landmark inference on ${eligibleInferenceCount} unseen or failed image${eligibleInferenceCount === 1 ? "" : "s"}`
        : "All detected images already have landmark results";
  const currentEdited = isReviewStateHydrated && editedImageIndices.has(currentIndex);
  const currentSaved = isReviewStateHydrated && !currentEdited && savedImageIndices.has(currentIndex);
  const currentCommitted = isReviewStateHydrated && committedImageIndices.has(currentIndex);
  const reviewActionsDisabled =
    isRunning ||
    isSavingCorrections ||
    isCommittingReview ||
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
  const hasWorkspaceData =
    images.length > 0 ||
    editedImageIndices.size > 0 ||
    savedImageIndices.size > 0 ||
    correctedSpecimensMap.size > 0;
  const handleReturnToInferenceHub = () => {
    if (
      hasWorkspaceData &&
      !window.confirm(
        "Return to Inference Sessions? Unsaved in-memory canvas state for this view will be reset."
      )
    ) {
      return;
    }
    setShowInferenceHub(true);
    setSelectedInferenceSpeciesId(null);
    setInferenceSessionId("");
    setInferenceSessionManifest(null);
    setSessionPreferredModelKey("");
    setSelectedModelKey("");
    resetInferenceWorkspaceState();
    void refreshInferenceHubSessions();
  };

  if (showInferenceHub) {
    return (
      <div className="flex h-screen w-screen flex-col bg-background">
        <div className="flex items-center justify-between border-b p-4">
          <div className="flex items-center gap-4">
            <motion.div {...buttonHover} {...buttonTap}>
              <Button variant="ghost" size="icon" onClick={() => onNavigate("landing")}>
                <ArrowLeft className="h-5 w-5" />
              </Button>
            </motion.div>
            <div className="flex items-center gap-2">
              <Microscope className="h-5 w-5 text-primary" />
              <h1 className="text-lg font-bold">Inference Sessions</h1>
            </div>
          </div>
          <Button
            onClick={() => {
              if (isFirstTimeLocked) {
                toast.error("Create or resume your first schema session in Annotate first.");
                return;
              }
              if (schemaOptions.length === 0) {
                toast.error("No schema sessions available. Create one in Annotate first.");
                return;
              }
              const fallbackSpeciesId =
                (activeSpeciesId && schemaOptions.some((schema) => schema.id === activeSpeciesId)
                  ? activeSpeciesId
                  : schemaOptions[0]?.id) || "";
              setCreateSessionSpeciesId(fallbackSpeciesId);
              setCreateSessionName("");
              setCreateSessionDialogOpen(true);
            }}
            disabled={isFirstTimeLocked || schemaOptions.length === 0}
          >
            <Plus className="mr-2 h-4 w-4" />
            Create Inference Session
          </Button>
        </div>

        <div className="flex-1 overflow-auto p-6">
          {isFirstTimeLocked && (
            <Card className="mx-auto mb-6 max-w-2xl border-amber-500/40 bg-amber-500/5">
              <CardHeader>
                <CardTitle className="text-base">Annotate First</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <p>
                  Inference is available after you create or resume your first schema session in Annotate.
                </p>
                <Button onClick={() => onNavigate("workspace")} variant="outline">
                  Go to Annotate
                </Button>
              </CardContent>
            </Card>
          )}

          {loadingHubSessions ? (
            <div className="flex items-center justify-center py-16 text-sm text-muted-foreground">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Loading inference sessions...
            </div>
          ) : hubSessions.length === 0 ? (
            <Card className="mx-auto max-w-xl border-dashed">
              <CardContent className="py-10 text-center text-sm text-muted-foreground">
                No schema sessions found. Create and open a schema in Annotate first.
              </CardContent>
            </Card>
          ) : (
            <motion.div
              variants={staggerContainer}
              initial="initial"
              animate="animate"
              className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
            >
              {hubSessions.map((session) => {
                const opening = openingInferenceSessionSpeciesId === session.speciesId;
                return (
                  <motion.div key={session.speciesId} variants={staggerItem}>
                    <motion.div variants={cardHover} initial="initial" whileHover="hover" className="h-full">
                      <Card className="h-full border-border/50 bg-card/50 backdrop-blur-sm">
                        <CardHeader className="pb-2">
                          <div className="flex items-start justify-between gap-2">
                            <CardTitle className="truncate text-sm font-semibold">
                              {session.schemaName}
                            </CardTitle>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 shrink-0 text-muted-foreground hover:text-destructive"
                              disabled={deletingSessionId === session.speciesId}
                              onClick={() => void handleDeleteSession(session.speciesId)}
                              title="Delete session"
                            >
                              {deletingSessionId === session.speciesId ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                              ) : (
                                <Trash2 className="h-3.5 w-3.5" />
                              )}
                            </Button>
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <p className="text-xs text-muted-foreground">
                            Schema ID: <span className="font-mono">{session.speciesId}</span>
                          </p>
                          <p className="text-xs text-muted-foreground">
                            Images: {session.schemaImageCount}
                          </p>
                          <p className="truncate text-xs text-muted-foreground">
                            Inference Session: {session.exists ? session.displayName || "default" : "Not created"}
                          </p>
                          <Button
                            className="w-full"
                            variant={session.exists ? "default" : "outline"}
                            disabled={opening || isFirstTimeLocked}
                            onClick={() => {
                              if (session.exists) {
                                void openInferenceSessionForSchema(session.speciesId);
                              } else {
                                setCreateSessionSpeciesId(session.speciesId);
                                setCreateSessionName(session.schemaName);
                                setCreateSessionDialogOpen(true);
                              }
                            }}
                          >
                            {opening ? (
                              <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Opening...
                              </>
                            ) : session.exists ? (
                              "Open Session"
                            ) : (
                              "Create Session"
                            )}
                          </Button>
                        </CardContent>
                      </Card>
                    </motion.div>
                  </motion.div>
                );
              })}
            </motion.div>
          )}
        </div>

        <Dialog open={createSessionDialogOpen} onOpenChange={setCreateSessionDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Inference Session</DialogTitle>
              <DialogDescription>
                Create the single inference session for a schema. This session stores draft review state and commit history.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3">
              <div className="space-y-1">
                <Label htmlFor="inference-schema-select">Schema Session</Label>
                <select
                  id="inference-schema-select"
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                  value={createSessionSpeciesId}
                  onChange={(e) => setCreateSessionSpeciesId(e.target.value)}
                >
                  {schemaOptions.map((schema) => (
                    <option key={schema.id} value={schema.id}>
                      {schema.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-1">
                <Label htmlFor="inference-session-name">Session Name</Label>
                <Input
                  id="inference-session-name"
                  value={createSessionName}
                  onChange={(e) => setCreateSessionName(e.target.value)}
                  placeholder="e.g. Fish Morphometrics Review"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateSessionDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={() => void handleCreateInferenceSession()}>
                Create
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    );
  }

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
              {effectiveSessionId && (
                <span className="rounded bg-muted px-1.5 py-0.5">
                  Schema: {effectiveSessionId}
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
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              variant="outline"
              onClick={handleReturnToInferenceHub}
            >
              Switch Session
            </Button>
          </motion.div>
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
              {currentCommitted ? (
                <span className="rounded bg-indigo-500/15 px-2 py-1 text-xs font-semibold text-indigo-600">
                  Committed
                </span>
              ) : currentEdited ? (
                <span className="rounded bg-amber-500/15 px-2 py-1 text-xs font-semibold text-amber-600">
                  Edited
                </span>
              ) : currentSaved ? (
                <span className="rounded bg-emerald-500/15 px-2 py-1 text-xs font-semibold text-emerald-600">
                  Saved
                </span>
              ) : currentImage?.results && !isReviewStateHydrated ? (
                <span className="rounded bg-slate-500/15 px-2 py-1 text-xs font-semibold text-slate-600">
                  Restoring
                </span>
              ) : null}
            </div>
          )}
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              variant="outline"
              onClick={handleRunDetection}
              disabled={isRunning || !effectiveSessionId || !isReviewStateHydrated || images.length === 0 || eligibleDetectionCount === 0}
              title={detectionButtonTitle}
            >
              {isRunning ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Square className="mr-2 h-4 w-4" />
                  {eligibleDetectionCount > 0
                    ? `${selectedModelKey && detectionRerunModelKey === selectedModelKey ? "Rerun Detection" : "Run Detection"} (${eligibleDetectionCount})`
                    : "Run Detection"}
                </>
              )}
            </Button>
          </motion.div>
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              onClick={handleRunInference}
              disabled={isRunning || !effectiveSessionId || !isReviewStateHydrated || !selectedModelKey || images.length === 0 || eligibleInferenceCount === 0}
              title={inferenceButtonTitle}
            >
              {isRunning ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  {eligibleInferenceCount > 0
                    ? `${inferenceRerunModelKey === selectedModelKey ? "Rerun Landmark Inference" : "Run Landmark Inference"} (${eligibleInferenceCount})`
                    : "Run Landmark Inference"}
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
            {typeof inferProgress.currentIndex === "number" &&
            typeof inferProgress.total === "number" &&
            inferProgress.total > 0
              ? ` (${Math.min(inferProgress.currentIndex, inferProgress.total)} of ${inferProgress.total})`
              : ""}
            {inferProgress.imagePath
              ? ` - ${inferProgress.imagePath.split(/[\\/]/).pop()}`
              : ""}
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
                    ) : !effectiveSessionId ? (
                      <p className="text-xs text-muted-foreground">
                        Select a session to load session-scoped models.
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
                          onChange={(e) => {
                            const newKey = e.target.value;
                            if (newKey !== selectedModelKey) {
                              setSelectedSpecimenIndex(null);
                              setShowMaskOverlay(false);
                              setDetectionRerunModelKey(newKey);
                              setInferenceRerunModelKey(newKey);
                            }
                            setSelectedModelKey(newKey);
                          }}
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
                              No OBB detector trained - orientation accuracy may be reduced. Train an OBB detector in the Training dialog.
                            </p>
                          </div>
                        )}
                        {obbDetectorReady === true && (
                          <p className="text-[11px] text-green-600 dark:text-green-400 mt-1">
                            OBB detector active - orientation from detector geometry.
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
                          Current Image Review
                        </span>
                        <span
                          className={cn(
                            "rounded px-1.5 py-0.5 text-[10px] font-semibold",
                            reviewFinalizedImageIndices.has(currentIndex)
                              ? "bg-emerald-500/15 text-emerald-600"
                              : "bg-amber-500/15 text-amber-600"
                          )}
                        >
                          {reviewFinalizedImageIndices.has(currentIndex)
                            ? "Complete"
                            : "In Progress"}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={reviewActionsDisabled || !currentImage?.results}
                          className="justify-start"
                          onClick={() => {
                            if (reviewFinalizedImageIndices.has(currentIndex)) return;
                            setReviewFinalizedImageIndices((prev) => {
                              const next = new Set(prev);
                              next.add(currentIndex);
                              return next;
                            });
                            void persistReviewDraft(currentIndex, {
                              specimens: getSpecimensForImageIndex(currentIndex),
                              edited: editedImageIndices.has(currentIndex),
                              saved: savedImageIndices.has(currentIndex),
                              reviewComplete: true,
                            });
                          }}
                        >
                          Mark Review Complete
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={reviewActionsDisabled || !currentImage?.results}
                          className="justify-start"
                          onClick={() => {
                            if (!reviewFinalizedImageIndices.has(currentIndex)) return;
                            setReviewFinalizedImageIndices((prev) => {
                              const next = new Set(prev);
                              next.delete(currentIndex);
                              return next;
                            });
                            setCommittedImageIndices((prev) => {
                              const next = new Set(prev);
                              next.delete(currentIndex);
                              return next;
                            });
                            void persistReviewDraft(currentIndex, {
                              specimens: getSpecimensForImageIndex(currentIndex),
                              edited: editedImageIndices.has(currentIndex),
                              saved: savedImageIndices.has(currentIndex),
                              reviewComplete: false,
                              committedAt: null,
                            });
                          }}
                        >
                          Mark In Progress
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleSaveAllCorrections}
                          disabled={reviewActionsDisabled}
                          title="Persist edited review drafts in this inference session"
                          className="justify-start"
                        >
                          {isSavingCorrections && !isCommittingReview ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Save className="mr-2 h-4 w-4" />
                          )}
                          Save All Changes
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleCommitReviewComplete}
                          disabled={reviewActionsDisabled}
                          title="Commit review-complete images into schema training data"
                          className="justify-start"
                        >
                          {isCommittingReview ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Save className="mr-2 h-4 w-4" />
                          )}
                          Commit to Training Data
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
                        {isDrawBoxMode && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              const next = drawDefaultOrientation === "left" ? "right" : "left";
                              setDrawDefaultOrientation(next);
                              window.localStorage.setItem("bv_draw_default_orientation", next);
                              if (
                                selectedSpecimenResolvedIndex !== null &&
                                selectedSpecimenResolvedIndex >= 0 &&
                                selectedSpecimenResolvedIndex < currentSpecimenCount
                              ) {
                                handleSetSpecimenOrientation(selectedSpecimenResolvedIndex, next);
                              }
                            }}
                            title="Toggle head direction (selected box and new boxes)"
                            className="justify-start text-xs"
                          >
                            {drawDefaultOrientation === "left" ? "\u2190 Head" : "Head \u2192"}
                          </Button>
                        )}
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
                          <div className="grid grid-cols-2 gap-1">
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
                              idx !== currentIndex && !editedImageIndices.has(idx) && savedImageIndices.has(idx) && "border-emerald-500/60 bg-emerald-500/5",
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
                            {img.results && committedImageIndices.has(idx) && (
                              <span className="shrink-0 rounded bg-indigo-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-600">
                                Committed
                              </span>
                            )}
                            {img.results && !isReviewStateHydrated && (
                              <span className="shrink-0 rounded bg-slate-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-slate-600">
                                Restoring
                              </span>
                            )}
                            {img.results && isReviewStateHydrated && !committedImageIndices.has(idx) && editedImageIndices.has(idx) && (
                              <span className="shrink-0 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-amber-600">
                                Edited
                              </span>
                            )}
                            {img.results && isReviewStateHydrated && !committedImageIndices.has(idx) && !editedImageIndices.has(idx) && savedImageIndices.has(idx) && (
                              <span className="shrink-0 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-600">
                                Saved
                              </span>
                            )}
                            {img.results && isReviewStateHydrated && !savedImageIndices.has(idx) && !editedImageIndices.has(idx) && (
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
            !isReviewStateHydrated ? (
              <div className="text-center">
                <Loader2 className="mx-auto mb-4 h-12 w-12 animate-spin text-muted-foreground/50" />
                <h2 className="text-lg font-semibold">Loading session…</h2>
                <p className="mt-2 max-w-sm text-sm text-muted-foreground">
                  Restoring images and review history.
                </p>
              </div>
            ) : (
              <div className="text-center">
                <Microscope className="mx-auto mb-4 h-16 w-16 text-muted-foreground/50" />
                <h2 className="text-lg font-semibold">No images to display</h2>
                <p className="mt-2 max-w-sm text-sm text-muted-foreground">
                  Add images and select a model to run inference.
                </p>
              </div>
            )
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
                      {currentSpecimenCount > 0 && (
                        <p className="text-muted-foreground/70">
                          Drag landmarks and bounding box corners, then save review to include this image in training-data commit.
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
                      {currentCommitted && (
                        <p className="font-medium text-indigo-600">Committed to schema training data.</p>
                      )}
                      {!currentCommitted && currentEdited && (
                        <p className="font-medium text-amber-600">Edited locally. Save to persist corrections.</p>
                      )}
                      {!currentCommitted && !currentEdited && currentSaved && (
                        <p className="font-medium text-emerald-600">Saved in inference session.</p>
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
                                  {typeof pcaAngle === "number" ? `${pcaAngle.toFixed(1)}ï¿½` : "n/a"} | canonical-flip {canonicalFlip} | dir{" "}
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
