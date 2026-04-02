import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Trash2, ZoomIn, Loader2, Sparkles, PencilRuler, XCircle, Save, CheckCircle2, LockOpen } from "lucide-react";
import { toast } from "sonner";

import ImageLabeler from "./ImageLabeler";
import { DetectionMode } from "./DetectionModeSelector";
import MagnifiedImageLabeler from "./MagnifiedZoomLabeler";
import { UndoRedoClearContext } from "./UndoRedoClearContext";

import { AppDispatch, RootState } from "../state/store";
import {
  clearFinalizePhaseForImages,
  isFinalizeInFlight as isSharedFinalizeInFlight,
  isTerminalFinalizeState,
  normalizeFinalizeFilename,
  removeFile,
  setFinalizePhaseForImage,
  setImageFinalized,
  setImagesUnfinalized,
  updateBoxes,
} from "../state/filesState/fileSlice";
import {
  AnnotatedImage,
  BoundingBox,
  FinalizeFailureDetail,
  FinalizePhaseMetadata,
  ObbDetectionSettings,
  OrientationPolicy,
} from "../types/Image";
import { updateSpecies } from "@/state/speciesState/speciesSlice";
import { normalizeObbDetectionSettings } from "@/lib/obbDetectorSettings";

import { Button } from "@/Components/ui/button";
import { Card } from "@/Components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/Components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/Components/ui/tooltip";
import { buttonHover, buttonTap } from "@/lib/animations";

interface ImageLabelerCarouselProps {
  color: string;
  opacity: number;
  isSwitchOn: boolean;
  detectionMode?: DetectionMode;
  obbDetectionSettings?: ObbDetectionSettings;
  className?: string;
  samEnabled?: boolean;
}

const ImageLabelerCarousel: React.FC<ImageLabelerCarouselProps> = ({
  color,
  opacity,
  isSwitchOn,
  detectionMode = "manual",
  obbDetectionSettings,
  className = "",
  samEnabled = false,
}) => {
  const {
    images,
    boxes,
    selectedBoxId,
    setSelectedImage,
    setBoxesFromSuperAnnotation,
    flipAllBoxOrientations,
    selectBox,
    deleteBox,
  } = useContext(UndoRedoClearContext);
  const dispatch = useDispatch<AppDispatch>();

  // Get Redux state directly to avoid sync issues during deletion
  const reduxFileArray = useSelector((state: RootState) => state.files.fileArray);

  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isMagnified, setIsMagnified] = useState<boolean>(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [autoDetectProgress, setAutoDetectProgress] = useState<{ message: string; percent: number } | null>(null);
  const [isAutoBoxCorrection, setIsAutoBoxCorrection] = useState(false);
  const [isBulkFinalizing, setIsBulkFinalizing] = useState(false);
  const [bulkFinalizeProgress, setBulkFinalizeProgress] = useState<{ current: number; total: number } | null>(null);
  const [isUnfinalizingCurrent, setIsUnfinalizingCurrent] = useState(false);
  const [isBulkUnfinalizing, setIsBulkUnfinalizing] = useState(false);
  const [finalizedBoxSignatureByImageId, setFinalizedBoxSignatureByImageId] = useState<Record<number, string>>({});
  const [annotationHydratedIds, setAnnotationHydratedIds] = useState<Set<number>>(new Set());
  const [sessionOrientationPolicy, setSessionOrientationPolicy] = useState<OrientationPolicy | undefined>(undefined);
  const resolvedDetectionSettings = useMemo(
    () => normalizeObbDetectionSettings(obbDetectionSettings),
    [obbDetectionSettings]
  );
  const annotationLoadInFlight = useRef<Set<number>>(new Set());
  const recentlyUnfinalizedIds = useRef<Set<number>>(new Set());

  // Use Context images for display, but check Redux for "has images" to avoid race conditions
  const totalImages = images.length;
  const hasAnyImages = reduxFileArray.length > 0 || images.length > 0;

  // Clamp index when images array changes
  useEffect(() => {
    if (totalImages === 0) {
      setCurrentIndex(0);
      setSelectedImage(0);
      return;
    }
    if (currentIndex >= totalImages) {
      const newIndex = totalImages - 1;
      setCurrentIndex(newIndex);
      setSelectedImage(newIndex);
    }
  }, [totalImages, currentIndex, setSelectedImage]);

  useEffect(() => {
    setSelectedImage(currentIndex);
  }, [currentIndex, setSelectedImage]);

  // Get current image safely
  const current = useMemo(() => {
    if (totalImages === 0) return null;
    const safeIndex = Math.min(Math.max(currentIndex, 0), totalImages - 1);
    return images[safeIndex] ?? null;
  }, [images, currentIndex, totalImages]);
  const reduxImagesById = useMemo(
    () => new Map(reduxFileArray.map((image) => [image.id, image])),
    [reduxFileArray]
  );

  const hasAnnotations = Boolean(current?.boxes?.length);
  const hasFinalizableBoxes = Boolean(
    current?.boxes?.some((b) => b.width > 0 && b.height > 0)
  );
  // Get speciesId from current image
  const activeSpeciesId = current?.speciesId;

  const currentBoxSignature = useMemo(() => {
    const curBoxes = current?.boxes ?? [];
    const reduced = curBoxes
      .filter((b) => b.width > 0 && b.height > 0)
      .map((b) => ({
        left: Math.round(b.left),
        top: Math.round(b.top),
        width: Math.round(b.width),
        height: Math.round(b.height),
        ...(Array.isArray(b.obbCorners) && b.obbCorners.length === 4
          ? {
              obbCorners: b.obbCorners.map((point) => [
                Number(Number(point?.[0] || 0).toFixed(3)),
                Number(Number(point?.[1] || 0).toFixed(3)),
              ]),
            }
          : {}),
        ...(typeof b.angle === "number" && Number.isFinite(b.angle)
          ? { angle: Number(Number(b.angle).toFixed(3)) }
          : {}),
        ...(typeof b.class_id === "number" && Number.isFinite(b.class_id)
          ? { class_id: Math.round(Number(b.class_id)) }
          : {}),
        ...(b.orientation_override ? { orientation_override: b.orientation_override } : {}),
      }))
      .sort((a, b) =>
        a.left - b.left ||
        a.top - b.top ||
        a.width - b.width ||
        a.height - b.height
      );
    return JSON.stringify(reduced);
  }, [current?.boxes]);

  const getResolvedImage = useCallback(
    (image: AnnotatedImage | null | undefined) => {
      if (!image) return null;
      return reduxImagesById.get(image.id) ?? image;
    },
    [reduxImagesById]
  );

  const isTerminalFinalizeSuccess = useCallback(
    (image: AnnotatedImage | null | undefined) => {
      const resolved = getResolvedImage(image);
      return Boolean(resolved && isTerminalFinalizeState(resolved));
    },
    [getResolvedImage]
  );

  const isFinalizeInFlight = useCallback(
    (image: AnnotatedImage | null | undefined) => Boolean(image && isSharedFinalizeInFlight(getResolvedImage(image) ?? image)),
    [getResolvedImage]
  );

  // isCurrentFinalized: true if the user finalized this session OR the session was restored from disk as finalized.
  // Belt-and-suspenders: also check Redux directly by image id in case the context sync missed the update.
  const isCurrentFinalizedInRedux = useMemo(() => {
    if (!current) return false;
    const reduxImage = reduxFileArray.find((f) => f.id === current.id);
    return Boolean(reduxImage?.isFinalized);
  }, [reduxFileArray, current]);

  const currentHasTerminalSuccess = isTerminalFinalizeSuccess(current);
  const isCurrentFinalized = current
    ? (!isFinalizeInFlight(current) && (currentHasTerminalSuccess || Boolean(current.isFinalized) || isCurrentFinalizedInRedux))
    : false;
  const isCurrentFinalizing = current ? isFinalizeInFlight(current) : false;

  const isFinalizePending = Boolean(
    current &&
    !isCurrentFinalized &&
    !isCurrentFinalizing &&
    hasFinalizableBoxes &&
    finalizedBoxSignatureByImageId[current.id] !== currentBoxSignature
  );

  const finalizationCandidates = useMemo(
    () =>
      images.map((img) => ({
        id: img.id,
        speciesId: img.speciesId,
        filename: img.filename,
        isFinalized: Boolean(img.isFinalized),
      })),
    [images]
  );
  const finalizationCandidatesRef = useRef(finalizationCandidates);
  useEffect(() => {
    finalizationCandidatesRef.current = finalizationCandidates;
  }, [finalizationCandidates]);

  // Re-hydrate finalized status from persisted session labels whenever this view mounts.
  // This keeps "Detection Finalized" state stable after leaving/re-entering workspace.
  useEffect(() => {
    if (!activeSpeciesId) return;
    let cancelled = false;

    const hydrateFinalized = async () => {
      try {
        const result = await window.api.sessionLoad(activeSpeciesId);
        if (cancelled || !result.ok || !result.images) return;

        const finalizedNames = new Set(
          result.images
            .filter((img) => Boolean(img.finalized))
            .map((img) => normalizeFinalizeFilename(img.filename))
        );
        if (finalizedNames.size === 0) return;

        const candidates = finalizationCandidatesRef.current;
        candidates.forEach((img) => {
          const localPhase = reduxImagesById.get(img.id)?.finalizePhase?.state;
          if (localPhase === "queued" || localPhase === "running") return;
          if (
            !recentlyUnfinalizedIds.current.has(img.id) &&
            finalizedNames.has(normalizeFinalizeFilename(img.filename)) &&
            !img.isFinalized
          ) {
            dispatch(setImageFinalized({ id: img.id }));
          }
        });
      } catch (err) {
        console.error("Failed to hydrate finalized image state:", err);
      }
    };

    void hydrateFinalized();
    return () => {
      cancelled = true;
    };
  }, [activeSpeciesId, dispatch, reduxImagesById]);

  useEffect(() => {
    if (detectionMode !== "auto") {
      setIsAutoBoxCorrection(false);
    }
  }, [detectionMode, current?.id]);

  // Lazy-load heavy box/landmark payload only for the active image.
  useEffect(() => {
    if (!current?.speciesId || !current?.filename) return;
    if (current.boxes?.length) {
      if (!annotationHydratedIds.has(current.id)) {
        setAnnotationHydratedIds((prev) => new Set([...prev, current.id]));
      }
      return;
    }
    if (annotationHydratedIds.has(current.id)) return;
    if (annotationLoadInFlight.current.has(current.id)) return;

    let cancelled = false;
    annotationLoadInFlight.current.add(current.id);

    void window.api
      .sessionLoadAnnotation(current.speciesId, current.filename)
      .then((result) => {
        if (cancelled || !result.ok) return;
        dispatch(updateBoxes({ id: current.id, boxes: result.boxes || [] }));
        if (result.finalized && !isFinalizeInFlight(current)) {
          dispatch(setImageFinalized({ id: current.id }));
        }
      })
      .catch((err) => {
        console.error("Failed to load session annotation:", err);
      })
      .finally(() => {
        annotationLoadInFlight.current.delete(current.id);
        if (!cancelled) {
          setAnnotationHydratedIds((prev) => new Set([...prev, current.id]));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [current, annotationHydratedIds, dispatch, isFinalizeInFlight]);

  const handleDeleteImage = useCallback(() => {
    if (!current) return;

    const idToDelete = current.id;
    const filenameToDelete = current.filename;
    const speciesId = current.speciesId;

    // Close dialog first
    setConfirmOpen(false);

    // Delete from session on disk if in a session
    if (speciesId && filenameToDelete) {
      window.api.sessionDeleteImage(speciesId, filenameToDelete).catch((err) =>
        console.error("Failed to delete image from session:", err)
      );
    }

    // Use Redux state for accurate index calculation (it updates synchronously)
    const reduxIndex = reduxFileArray.findIndex(img => img.id === idToDelete);
    const reduxTotal = reduxFileArray.length;
    const newTotal = reduxTotal - 1;

    if (newTotal > 0) {
      const newIndex = reduxIndex >= newTotal ? newTotal - 1 : reduxIndex;
      setCurrentIndex(newIndex);
      setSelectedImage(newIndex);
    } else {
      setCurrentIndex(0);
      setSelectedImage(0);
    }

    // Dispatch the deletion
    dispatch(removeFile(idToDelete));
  }, [current, reduxFileArray, dispatch, setSelectedImage]);

  const ensureImageBoxesLoaded = useCallback(
    async (image: AnnotatedImage | null): Promise<BoundingBox[]> => {
      if (!image?.speciesId || !image?.filename) return image?.boxes || [];
      if ((image.boxes || []).length > 0) return image.boxes || [];
      if (annotationLoadInFlight.current.has(image.id)) return image.boxes || [];

      annotationLoadInFlight.current.add(image.id);
      try {
        const result = await window.api.sessionLoadAnnotation(image.speciesId, image.filename);
        if (!result.ok) return image.boxes || [];
        const loadedBoxes = result.boxes || [];
        dispatch(updateBoxes({ id: image.id, boxes: loadedBoxes }));
        if (result.finalized && !isFinalizeInFlight(image)) {
          dispatch(setImageFinalized({ id: image.id }));
        }
        setAnnotationHydratedIds((prev) => new Set([...prev, image.id]));
        return loadedBoxes;
      } catch (err) {
        console.error("Failed to load session annotation:", err);
        return image.boxes || [];
      } finally {
        annotationLoadInFlight.current.delete(image.id);
      }
    },
    [dispatch, isFinalizeInFlight]
  );

  const getMatchingImageIds = useCallback((speciesId?: string, filename?: string): number[] => {
    const safeSpeciesId = String(speciesId || "").trim();
    const safeFilename = normalizeFinalizeFilename(filename);
    return images
      .filter(
        (img) =>
          String(img.speciesId || "").trim() === safeSpeciesId &&
          normalizeFinalizeFilename(img.filename) === safeFilename
      )
      .map((img) => img.id);
  }, [images]);

  const setFinalizePhase = useCallback((
    speciesId: string,
    filename: string,
    phase: FinalizePhaseMetadata,
    isFinalized?: boolean
  ) => {
    dispatch(setFinalizePhaseForImage({
      speciesId,
      filename,
      finalizePhase: phase,
      ...(typeof isFinalized === "boolean" ? { isFinalized } : {}),
    }));
  }, [dispatch]);

  const markTerminalFinalizeSuccess = useCallback((
    speciesId: string,
    filename: string,
    phaseState: Omit<FinalizePhaseMetadata, "state" | "updatedAt"> & {
      state: "saved" | "already_finalized" | "finalized_without_segments";
    }
  ): number[] => {
    const matchingIds = getMatchingImageIds(speciesId, filename);
    setFinalizePhase(speciesId, filename, {
      state: phaseState.state,
      signature: phaseState.signature,
      reason: phaseState.reason,
      expectedCount: phaseState.expectedCount,
      savedCount: phaseState.savedCount,
      details: phaseState.details,
      updatedAt: new Date().toISOString(),
    }, true);
    if (phaseState.signature) {
      setFinalizedBoxSignatureByImageId((prev) => {
        const next = { ...prev };
        matchingIds.forEach((id) => {
          next[id] = phaseState.signature as string;
        });
        return next;
      });
    }
    matchingIds.forEach((id) => dispatch(setImageFinalized({ id })));
    return matchingIds;
  }, [dispatch, getMatchingImageIds, setFinalizePhase]);

  const finalizeImageSegments = useCallback(async (
    image: AnnotatedImage | null
  ): Promise<{
    ok: boolean;
    speciesId?: string;
    filename?: string;
    signature?: string;
    state?: "queued" | "running" | "saved" | "already_finalized" | "finalized_without_segments";
    expectedCount?: number;
    savedCount?: number;
    details?: FinalizeFailureDetail[];
    error?: string;
  }> => {
    if (!image?.speciesId || !image?.filename) {
      return { ok: false, error: "No active image." };
    }
    const sourceBoxes = await ensureImageBoxesLoaded(image);

    const acceptedBoxes = (sourceBoxes || [])
      .filter((b) => b.width > 0 && b.height > 0)
      .map((b) => ({
        left: Math.round(b.left),
        top: Math.round(b.top),
        width: Math.round(b.width),
        height: Math.round(b.height),
        ...(b.orientation_override ? { orientation_override: b.orientation_override } : {}),
        ...(Array.isArray(b.obbCorners) && b.obbCorners.length === 4
          ? { obbCorners: b.obbCorners.map((point) => [Number(point[0]), Number(point[1])] as [number, number]) }
          : {}),
        ...(typeof b.angle === "number" && Number.isFinite(b.angle) ? { angle: Number(b.angle) } : {}),
        ...(typeof b.class_id === "number" && Number.isFinite(b.class_id) ? { class_id: Math.round(Number(b.class_id)) } : {}),
        ...(b.orientation_hint?.orientation
          ? {
              orientation_hint: {
                orientation: b.orientation_hint.orientation,
                ...(typeof b.orientation_hint.confidence === "number" && Number.isFinite(b.orientation_hint.confidence)
                  ? { confidence: Number(b.orientation_hint.confidence) }
                  : {}),
                ...(b.orientation_hint.source ? { source: b.orientation_hint.source } : {}),
              },
            }
          : {}),
        landmarks: (b.landmarks || [])
          .filter((lm) => Number.isFinite(Number(lm?.id)))
          .map((lm) => ({
            id: Number(lm.id),
            x: Number(lm.x),
            y: Number(lm.y),
            ...(lm.isSkipped ? { isSkipped: true } : {}),
          })),
      }));

    try {
      const result = await window.api.sessionFinalizeAcceptedBoxes(
        image.speciesId,
        image.filename,
        acceptedBoxes,
        image.path,
        samEnabled
      );
      if (
        result.ok &&
        result.signature &&
        (result.segmentQueueState === "queued" || result.segmentQueueState === "running")
      ) {
        setFinalizePhase(image.speciesId, image.filename, {
          state: result.segmentQueueState,
          signature: result.signature,
          updatedAt: new Date().toISOString(),
          reason: result.reason,
          expectedCount: result.expectedCount,
          savedCount: result.savedCount,
          details: result.details,
        });
        return {
          ok: true,
          speciesId: image.speciesId,
          filename: image.filename,
          signature: result.signature,
          state: result.segmentQueueState,
          expectedCount: result.expectedCount,
          savedCount: result.savedCount,
          details: result.details,
        };
      }
      if (
        result.ok &&
        result.signature &&
        (
          result.segmentQueueState === "saved" ||
          result.segmentQueueState === "already_finalized" ||
          result.segmentQueueState === "finalized_without_segments"
        )
      ) {
        markTerminalFinalizeSuccess(image.speciesId, image.filename, {
          state: result.segmentQueueState,
          signature: result.signature,
          reason: result.reason,
          expectedCount: result.expectedCount,
          savedCount: result.savedCount,
          details: result.details,
        });
        return {
          ok: true,
          speciesId: image.speciesId,
          filename: image.filename,
          signature: result.signature,
          state: result.segmentQueueState,
          expectedCount: result.expectedCount,
          savedCount: result.savedCount,
          details: result.details,
        };
      }
      return {
        ok: false,
        speciesId: image.speciesId,
        filename: image.filename,
        error: result.error || result.reason || "Segment finalization was not queued.",
      };
    } catch (err) {
      console.error("Failed to finalize accepted boxes:", err);
      return {
        ok: false,
        speciesId: image.speciesId,
        filename: image.filename,
        error: err instanceof Error ? err.message : "Failed to finalize accepted boxes.",
      };
    }
  }, [ensureImageBoxesLoaded, markTerminalFinalizeSuccess, samEnabled, setFinalizePhase]);

  const finalizeCurrentImageSegments = useCallback(async () => {
    return finalizeImageSegments(current);
  }, [current, finalizeImageSegments]);

  const clearFinalizeSignatureLocalState = useCallback((imageIds: number[]) => {
    if (!imageIds.length) return;
    const idSet = new Set(imageIds);
    setFinalizedBoxSignatureByImageId((prev) => {
      const next = { ...prev };
      idSet.forEach((id) => {
        delete next[id];
      });
      return next;
    });
  }, []);

  const clearFinalizeLocalState = useCallback((imageIds: number[]) => {
    if (!imageIds.length) return;
    clearFinalizeSignatureLocalState(imageIds);
    dispatch(clearFinalizePhaseForImages({ ids: imageIds }));
  }, [clearFinalizeSignatureLocalState, dispatch]);

  const summarizeFinalizeReason = useCallback((
    reason?: string,
    details?: FinalizeFailureDetail[]
  ): string | undefined => {
    if (details && details.length > 0) {
      const firstFailure = details.find((detail) => detail.status === "failed" && detail.reason);
      if (firstFailure?.reason) {
        return firstFailure.reason.replace(/_/g, " ");
      }
    }
    if (!reason) return undefined;
    return reason.replace(/^partial_segment_save:/, "partial segment save: ").replace(/_/g, " ");
  }, []);

  const waitForSegmentSaveOutcome = useCallback(async (
    speciesId: string,
    filename: string,
    signature?: string,
    options?: { bulkSequential?: boolean }
  ) => {
    const isTerminal = (state: string) =>
      state === "saved" ||
      state === "already_finalized" ||
      state === "finalized_without_segments" ||
      state === "skipped" ||
      state === "failed";
    try {
      const initial = await window.api.sessionGetSegmentSaveStatus(speciesId, filename);
      if (
        initial.ok &&
        initial.status &&
        (!signature || initial.status.signature === signature) &&
        isTerminal(initial.status.state)
      ) {
        return { speciesId, filename, ...initial.status };
      }
    } catch {
      // fall through to event wait
    }

    return await new Promise<{
      speciesId: string;
      filename: string;
      state: "idle" | "queued" | "running" | "saved" | "already_finalized" | "finalized_without_segments" | "skipped" | "failed";
      signature?: string;
      updatedAt: string;
      reason?: string;
      expectedCount?: number;
      savedCount?: number;
      details?: FinalizeFailureDetail[];
    }>((resolve) => {
      let resolved = false;
      let pollCancelled = false;
      const finish = (payload: {
        speciesId: string;
        filename: string;
        state: "idle" | "queued" | "running" | "saved" | "already_finalized" | "finalized_without_segments" | "skipped" | "failed";
        signature?: string;
        updatedAt: string;
        reason?: string;
        expectedCount?: number;
        savedCount?: number;
        details?: FinalizeFailureDetail[];
      }) => {
        if (resolved) return;
        resolved = true;
        pollCancelled = true;
        unsub();
        resolve(payload);
      };
      const unsub = window.api.onSegmentSaveStatus((data) => {
        if (
          data.speciesId === speciesId &&
          data.filename === filename &&
          (!signature || data.signature === signature) &&
          isTerminal(data.state)
        ) {
          finish(data);
        }
      });
      const timeoutMs = options?.bulkSequential ? 30 * 60_000 : 120_000;
      const timeoutAt = Date.now() + timeoutMs;
      const poll = async () => {
        while (!pollCancelled) {
          await new Promise((pollResolve) => setTimeout(pollResolve, 2000));
          if (pollCancelled) return;
          try {
            const status = await window.api.sessionGetSegmentSaveStatus(speciesId, filename);
            if (status.ok && status.status && (!signature || status.status.signature === signature)) {
              if (isTerminal(status.status.state)) {
                finish({ speciesId, filename, ...status.status });
                return;
              }
              if (options?.bulkSequential && (status.status.state === "queued" || status.status.state === "running")) {
                continue;
              }
            }
          } catch {
            // keep polling until timeout
          }
          if (Date.now() >= timeoutAt) {
            finish({
              speciesId,
              filename,
              state: "failed",
              updatedAt: new Date().toISOString(),
              reason: "finalize_timeout_waiting_for_backend_terminal_status",
            });
            return;
          }
        }
      };
      void poll();
    });
  }, []);

  useEffect(() => {
    const unsub = window.api.onSegmentSaveStatus((data) => {
      const matchingIds = getMatchingImageIds(data.speciesId, data.filename);
      if (matchingIds.length === 0) return;
      const localPhase = reduxFileArray.find((image) => matchingIds.includes(image.id))?.finalizePhase;
      if (
        localPhase?.signature &&
        data.signature &&
        localPhase.signature !== data.signature &&
        (localPhase.state === "queued" || localPhase.state === "running")
      ) {
        return;
      }

      if (data.state === "queued" || data.state === "running") {
        setFinalizePhase(data.speciesId, data.filename, {
          state: data.state,
          signature: data.signature,
          updatedAt: data.updatedAt,
          reason: data.reason,
          expectedCount: data.expectedCount,
          savedCount: data.savedCount,
          details: data.details,
        });
        return;
      }

      if (
        data.state === "saved" ||
        data.state === "already_finalized" ||
        data.state === "finalized_without_segments"
      ) {
        markTerminalFinalizeSuccess(data.speciesId, data.filename, {
          state: data.state,
          signature: data.signature,
          reason: data.reason,
          expectedCount: data.expectedCount,
          savedCount: data.savedCount,
          details: data.details,
        });
        return;
      }

      if (data.state === "failed" || data.state === "skipped") {
        setFinalizePhase(data.speciesId, data.filename, {
          state: "failed",
          signature: data.signature,
          updatedAt: data.updatedAt,
          reason: data.reason,
          expectedCount: data.expectedCount,
          savedCount: data.savedCount,
          details: data.details,
        });
        clearFinalizeSignatureLocalState(matchingIds);
        dispatch(setImagesUnfinalized({ ids: matchingIds }));
      }
    });
    return () => {
      unsub();
    };
  }, [clearFinalizeSignatureLocalState, dispatch, getMatchingImageIds, markTerminalFinalizeSuccess, reduxFileArray, setFinalizePhase]);

  const handleUnfinalizeCurrentImage = useCallback(async () => {
    if (!current?.speciesId || !current?.filename) return;
    setIsUnfinalizingCurrent(true);
    try {
      const filenameLower = String(current.filename || "").toLowerCase();
      const matchingIds = images
        .filter(
          (img) =>
            String(img.speciesId || "") === String(current.speciesId || "") &&
            String(img.filename || "").toLowerCase() === filenameLower
        )
        .map((img) => img.id);
      const imagePath = current.path ?? current.diskPath;
      const result = await window.api.sessionUnfinalizeImage(
        current.speciesId,
        current.filename,
        imagePath
      );
      if (!result.ok) {
        toast.error(result.error || "Failed to unfinalize image.");
        return;
      }

      const idsToClear = matchingIds.length > 0 ? matchingIds : [current.id];
      idsToClear.forEach((id) => recentlyUnfinalizedIds.current.add(id));
      clearFinalizeLocalState(idsToClear);
      dispatch(setImagesUnfinalized({ ids: idsToClear }));

      if (!result.removedFromList && !result.hadFinalizedDetection) {
        toast.info("Image was already unfinalized.");
      } else {
        toast.success("Image unfinalized. Detection editing is enabled again.");
      }
    } catch (err) {
      console.error("Failed to unfinalize image:", err);
      toast.error("Failed to unfinalize image.");
    } finally {
      setIsUnfinalizingCurrent(false);
    }
  }, [clearFinalizeLocalState, current, dispatch, images]);

  const finalizableImages = useMemo(() => {
    return images.filter((img) => {
      const hasValidBoxes =
        (img.boxes || []).some((b) => b.width > 0 && b.height > 0) ||
        Boolean(img.hasBoxes);
      if (!hasValidBoxes || !img.speciesId || !img.filename) return false;
      const finalized =
        !isFinalizeInFlight(img) &&
        (isTerminalFinalizeSuccess(img) || Boolean(img.isFinalized));
      const finalizing = isFinalizeInFlight(img);
      return !finalized && !finalizing;
    });
  }, [images, isFinalizeInFlight, isTerminalFinalizeSuccess]);

  const finalizableImageCount = finalizableImages.length;
  const finalizedImages = useMemo(() => {
    return images.filter((img) => {
      if (!img.speciesId || !img.filename) return false;
      return !isFinalizeInFlight(img) && (isTerminalFinalizeSuccess(img) || Boolean(img.isFinalized));
    });
  }, [images, isFinalizeInFlight, isTerminalFinalizeSuccess]);
  const finalizedImageCount = finalizedImages.length;

  const handleUnfinalizeAllFinalized = useCallback(async () => {
    if (!activeSpeciesId || finalizedImages.length === 0) {
      toast.info("No finalized images to unfinalize.");
      return;
    }

    setIsBulkUnfinalizing(true);
    try {
      const filenames = finalizedImages
        .map((img) => String(img.filename || "").trim())
        .filter((name) => name.length > 0);
      const result = await window.api.sessionUnfinalizeImages(activeSpeciesId, filenames);
      if (!result.ok && result.succeeded === 0) {
        toast.error(result.error || "Failed to unfinalize finalized images.");
        return;
      }

      const failedFilenames = new Set(
        (result.errors || []).map((e) => String(e.filename || "").toLowerCase())
      );
      const succeededIds = finalizedImages
        .filter((img) => !failedFilenames.has(String(img.filename || "").toLowerCase()))
        .map((img) => img.id);

      succeededIds.forEach((id) => recentlyUnfinalizedIds.current.add(id));
      clearFinalizeLocalState(succeededIds);
      dispatch(setImagesUnfinalized({ ids: succeededIds }));

      if (result.failed === 0) {
        toast.success(`Unfinalized ${result.succeeded} image(s).`);
      } else {
        toast.warning(`Unfinalized ${result.succeeded} image(s), ${result.failed} failed.`);
      }
    } catch (err) {
      console.error("Failed to unfinalize finalized images:", err);
      toast.error("Failed to unfinalize finalized images.");
    } finally {
      setIsBulkUnfinalizing(false);
    }
  }, [activeSpeciesId, clearFinalizeLocalState, dispatch, finalizedImages]);

  const handleNext = useCallback(() => {
    if (totalImages <= 1) return;
    const newIndex = (currentIndex + 1) % totalImages;
    setCurrentIndex(newIndex);
    setSelectedImage(newIndex);
  }, [totalImages, currentIndex, setSelectedImage]);

  const handlePrev = useCallback(() => {
    if (totalImages <= 1) return;
    const newIndex = (currentIndex - 1 + totalImages) % totalImages;
    setCurrentIndex(newIndex);
    setSelectedImage(newIndex);
  }, [totalImages, currentIndex, setSelectedImage]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") handleNext();
      else if (e.key === "ArrowLeft") handlePrev();
    },
    [handleNext, handlePrev]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const toggleMagnifiedView = () => setIsMagnified((prev) => !prev);

  // Handle AI auto-detection via the session OBB detector and optional SAM2 refinement.
  const handleAutoDetect = useCallback(async () => {
    const promptClass = className.trim();
    if (!current || !current.path) return;

    setIsDetecting(true);
    setAutoDetectProgress({ message: "Starting...", percent: 0 });

    // Subscribe to progress events
    const unsubscribe = window.api.onSuperAnnotateProgress((data) => {
      setAutoDetectProgress({ message: data.message, percent: data.percent });
    });

    try {
      const result = await window.api.superAnnotate(current.path, promptClass, undefined, {
        samEnabled,
        maxObjects: resolvedDetectionSettings.maxObjects,
        detectionMode: "auto",
        detectionPreset: resolvedDetectionSettings.detectionPreset,
        conf: resolvedDetectionSettings.conf,
        nmsIou: resolvedDetectionSettings.nmsIou,
        imgsz: resolvedDetectionSettings.imgsz,
        useOrientationHint: true,
      }, current.speciesId);

      if (result.ok && Array.isArray(result.objects) && result.objects.length > 0) {
        setBoxesFromSuperAnnotation(result.objects);
        setIsAutoBoxCorrection(false);
        toast.success(`Detected ${result.objects.length} object${result.objects.length > 1 ? "s" : ""}.`);
      } else if (result.ok) {
        toast.info("No objects detected. Try a different class prompt or detection goal.");
      } else {
        toast.error(result.error || "Auto-detect failed.");
      }

      // Signal that models are loaded so capability badges update
      window.dispatchEvent(new CustomEvent("super-annotator-ready"));
    } catch (err) {
      console.error("Failed to auto-detect:", err);
      toast.error("Auto-detect request failed.");
    } finally {
      unsubscribe();
      setIsDetecting(false);
      setAutoDetectProgress(null);
    }
  }, [current, className, resolvedDetectionSettings, samEnabled, setBoxesFromSuperAnnotation]);

  const handleAutoDetectOrFinalize = useCallback(async () => {
    if (isFinalizePending) {
      const request = await finalizeCurrentImageSegments();
      if (!request.ok || !request.speciesId || !request.filename) {
        toast.error(request.error || "Failed to finalize accepted boxes.");
        return;
      }
      if (
        request.state === "saved" ||
        request.state === "already_finalized" ||
        request.state === "finalized_without_segments"
      ) {
        if (request.state === "finalized_without_segments") {
          toast.warning(
            summarizeFinalizeReason(undefined, request.details) ||
              "Detection finalized without segments for this image."
          );
        } else {
          toast.success("Finalized accepted boxes for this image.");
        }
        return;
      }
      const outcome = await waitForSegmentSaveOutcome(
        request.speciesId,
        request.filename,
        request.signature
      );
      if (outcome.state === "saved" || outcome.state === "already_finalized") {
        toast.success("Finalized accepted boxes for this image.");
      } else if (outcome.state === "finalized_without_segments") {
        toast.warning(
          summarizeFinalizeReason(outcome.reason, outcome.details) ||
            "Detection finalized without segments for this image."
        );
      } else {
        toast.error(
          summarizeFinalizeReason(outcome.reason, outcome.details) ||
            (outcome.expectedCount != null && outcome.savedCount != null
              ? `Saved ${outcome.savedCount}/${outcome.expectedCount} segments.`
              : "Failed to finalize accepted boxes.")
        );
      }
      return;
    }
    await handleAutoDetect();
  }, [finalizeCurrentImageSegments, handleAutoDetect, isFinalizePending, summarizeFinalizeReason, waitForSegmentSaveOutcome]);

  const handleFinalizeAllWithBoxes = useCallback(async () => {
    if (finalizableImageCount === 0) {
      toast.info("No remaining images with boxes to finalize.");
      return;
    }

    setIsBulkFinalizing(true);
    setBulkFinalizeProgress({ current: 0, total: finalizableImages.length });
    let successCount = 0;
    let failCount = 0;
    let detectionOnlyCount = 0;

    try {
      for (let index = 0; index < finalizableImages.length; index += 1) {
        const image = finalizableImages[index];
        setBulkFinalizeProgress({ current: index + 1, total: finalizableImages.length });
        const request = await finalizeImageSegments(image);
        if (
          request.ok &&
          (
            request.state === "saved" ||
            request.state === "already_finalized" ||
            request.state === "finalized_without_segments"
          )
        ) {
          successCount += 1;
          if (request.state === "finalized_without_segments") {
            detectionOnlyCount += 1;
          }
          continue;
        }

        if (
          request.ok &&
          request.speciesId &&
          request.filename &&
          (request.state === "queued" || request.state === "running")
        ) {
          const outcome = await waitForSegmentSaveOutcome(
            request.speciesId,
            request.filename,
            request.signature,
            { bulkSequential: true }
          );
          if (
            outcome.state === "saved" ||
            outcome.state === "already_finalized" ||
            outcome.state === "finalized_without_segments"
          ) {
            successCount += 1;
            if (outcome.state === "finalized_without_segments") {
              detectionOnlyCount += 1;
            }
          } else {
            failCount += 1;
          }
        } else {
          failCount += 1;
        }
      }
    } catch (err) {
      console.error("Finalize-all error:", err);
      setIsBulkFinalizing(false);
      setBulkFinalizeProgress(null);
      toast.error("Failed to finalize images with boxes.");
      return;
    }

    if (successCount === 0) {
      setIsBulkFinalizing(false);
      setBulkFinalizeProgress(null);
      toast.error("Failed to finalize images with boxes.");
      return;
    }

    setIsBulkFinalizing(false);
    setBulkFinalizeProgress(null);

    if (successCount > 0 && failCount === 0) {
      if (detectionOnlyCount > 0) {
        toast.warning(
          `Finalized ${successCount} image(s); ${detectionOnlyCount} finished without segments.`
        );
      } else {
        toast.success(`Finalized ${successCount} image(s) with accepted boxes.`);
      }
    } else if (successCount > 0) {
      toast.warning(`Finalized ${successCount} image(s), ${failCount} failed.`);
    } else {
      toast.error("Failed to finalize images with boxes.");
    }
  }, [finalizableImageCount, finalizableImages, finalizeImageSegments, waitForSegmentSaveOutcome]);

  const handleToggleAutoBoxCorrection = useCallback(() => {
    if (!isAutoBoxCorrection && selectedBoxId === null && boxes.length > 0) {
      selectBox(boxes[0].id);
    }
    setIsAutoBoxCorrection((prev) => !prev);
  }, [isAutoBoxCorrection, selectedBoxId, boxes, selectBox]);

  const handleDeleteSelectedAutoBox = useCallback(() => {
    if (selectedBoxId === null) {
      toast.info("Select a detected box first.");
      return;
    }
    const rejected = boxes.find((b) => b.id === selectedBoxId);
    if (current?.speciesId && current?.filename && rejected && rejected.source === "predicted") {
      window.api.sessionAddRejectedDetection(current.speciesId, current.filename, {
        left: rejected.left,
        top: rejected.top,
        width: rejected.width,
        height: rejected.height,
        confidence: rejected.confidence,
        className: rejected.className,
        detectionMethod: rejected.detectionMethod,
      }).catch((err) => console.error("Failed to persist rejected detection:", err));
    }
    deleteBox(selectedBoxId);
    toast.success("Selected detected box removed.");
  }, [selectedBoxId, boxes, current, deleteBox]);

  // Get active species schema from Redux for the zoomed view
  const activeSpeciesForSchema = useSelector((state: RootState) =>
    state.species.species.find((s) => s.id === state.species.activeSpeciesId)
  );
  const activeSchema = activeSpeciesForSchema
    ? {
        id: activeSpeciesForSchema.id,
        name: activeSpeciesForSchema.name,
        description: activeSpeciesForSchema.description || "",
        landmarks: activeSpeciesForSchema.landmarkTemplate,
      }
    : undefined;
  const activeSpeciesForSchemaId = activeSpeciesForSchema?.id;
  const reduxOrientationPolicy = activeSpeciesForSchema?.orientationPolicy;

  useEffect(() => {
    const speciesId = current?.speciesId || activeSpeciesForSchemaId;
    if (!speciesId) {
      setSessionOrientationPolicy(undefined);
      return;
    }

    let cancelled = false;
    window.api.sessionLoad(speciesId).then((result) => {
      if (cancelled || !result?.ok) return;
      const sessionPolicy = result.meta?.orientationPolicy;
      const sessionMode = sessionPolicy?.mode;
      if (
        sessionMode === "directional" ||
        sessionMode === "bilateral" ||
        sessionMode === "axial" ||
        sessionMode === "invariant"
      ) {
        setSessionOrientationPolicy(sessionPolicy);
        if (
          sessionPolicy &&
          (
            sessionMode !== reduxOrientationPolicy?.mode ||
            sessionPolicy.bilateralClassAxis !== reduxOrientationPolicy?.bilateralClassAxis
          )
        ) {
          dispatch(
            updateSpecies({
              id: speciesId,
              updates: {
                orientationPolicy: sessionPolicy,
              },
            })
          );
        }
      }
    }).catch(() => {
      if (!cancelled) {
        setSessionOrientationPolicy(undefined);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [activeSpeciesForSchemaId, current?.speciesId, dispatch, reduxOrientationPolicy?.bilateralClassAxis, reduxOrientationPolicy?.mode]);

  const activeOrientationPolicy = sessionOrientationPolicy ?? reduxOrientationPolicy;
  const activeOrientationMode = activeOrientationPolicy?.mode;
  const activeBilateralClassAxis = activeOrientationPolicy?.bilateralClassAxis;

  const exportCurrent = useCallback(async () => {
    if (!current || !current.boxes?.length) return;

    const dims = await new Promise<{ width: number; height: number } | null>(
      (resolve) => {
        const img = new Image();
        img.onload = () =>
          resolve({ width: img.naturalWidth, height: img.naturalHeight });
        img.onerror = () => resolve(null);
        img.src = current.url;
      }
    );

    const data = {
      imageURL: current.url,
      imageDimensions: dims,
      boxes: current.boxes.map((box) => ({
        left: box.left,
        top: box.top,
        width: box.width,
        height: box.height,
        landmarks: box.landmarks.map(({ x, y, id }) => ({
          x: Math.round(x),
          y: Math.round(y),
          id,
        })),
      })),
    };

    const jsonData = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonData], { type: "application/json" });
    const urlBlob = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = urlBlob;
    a.download = `labeled_data_${Date.now()}.json`;
    a.click();

    URL.revokeObjectURL(urlBlob);
  }, [current]);

  const exportAll = useCallback(() => {
    const data = images.map(({ id, url, boxes }) => ({
      id,
      url,
      boxes: boxes.map((box) => ({
        left: box.left,
        top: box.top,
        width: box.width,
        height: box.height,
        landmarks: box.landmarks.map(({ x, y, id }) => ({
          x: Math.round(x),
          y: Math.round(y),
          id,
        })),
      })),
    }));
    const jsonData = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonData], { type: "application/json" });
    const urlBlob = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = urlBlob;
    a.download = `all_labeled_data_${Date.now()}.json`;
    a.click();

    URL.revokeObjectURL(urlBlob);
  }, [images]);

  // Only show empty state if BOTH Redux and Context have no images
  if (!hasAnyImages) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <div className="flex flex-col items-center gap-4 rounded-xl border border-border/40 bg-card/50 backdrop-blur-sm p-10 text-center max-w-sm">
          <div className="rounded-xl bg-primary/10 p-4 ring-1 ring-primary/20">
            <ZoomIn className="h-8 w-8 text-primary" />
          </div>
          <div className="space-y-1.5">
            <h2 className="font-display text-base font-semibold text-foreground">No images loaded</h2>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Press{" "}
              <kbd className="rounded border border-border/60 bg-muted/70 px-1.5 py-0.5 font-mono text-[10px]">
                Ctrl+N
              </kbd>{" "}
              to upload images, or use the sidebar to begin labeling.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
      <Card className="flex h-full w-full min-h-0 min-w-0 flex-col gap-3 border-border/50 bg-card/50 p-4 backdrop-blur-sm">
        {/* Toolbar */}
        <div className="flex w-full flex-wrap items-center gap-1.5">
          {/* Image counter + filename */}
          <div className="flex items-center gap-2 mr-auto">
            <div className="flex items-center gap-1 rounded-md border border-border/50 bg-muted/50 px-2.5 py-1">
              <span className="font-mono text-xs font-semibold text-foreground">{currentIndex + 1}</span>
              <span className="font-mono text-[10px] text-muted-foreground/40">/</span>
              <span className="font-mono text-xs text-muted-foreground">{totalImages}</span>
            </div>
            {current?.filename && (
              <span className="hidden md:block truncate font-mono text-[11px] text-muted-foreground/55 max-w-[220px]">
                {current.filename}
              </span>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {/* Finalized badge — shown in any detection mode once finalized */}
            {isCurrentFinalizing ? (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/15 px-3 py-1 text-xs font-bold text-amber-300 ring-1 ring-amber-500/40">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Finalizing Segments
                </span>
                <span className="text-xs text-muted-foreground">Editing locked until segment save completes</span>
              </div>
            ) : isCurrentFinalized ? (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-bold text-emerald-400 ring-1 ring-emerald-500/40">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Detection Finalized
                </span>
                <span className="text-xs text-muted-foreground">Landmark mode only</span>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleUnfinalizeCurrentImage}
                    disabled={isUnfinalizingCurrent || !current}
                    className="font-bold"
                  >
                    {isUnfinalizingCurrent ? (
                      <>
                        <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                        Unfinalizing...
                      </>
                    ) : (
                      <>
                        <LockOpen className="mr-1.5 h-3.5 w-3.5" />
                        Unfinalize This Image
                      </>
                    )}
                  </Button>
                </motion.div>
              </div>
            ) : (detectionMode === "auto" || isFinalizePending) ? (
              /* Finalize shared across modes; auto-detect/correction controls in auto mode only */
              <>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant={isFinalizePending ? "default" : "outline"}
                    size="sm"
                    onClick={handleAutoDetectOrFinalize}
                    disabled={isDetecting || !current || isCurrentFinalizing}
                    className={isFinalizePending ? "font-bold bg-emerald-600 hover:bg-emerald-700 text-white border-0" : "font-bold"}
                  >
                    {isDetecting ? (
                      <>
                        <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                        {autoDetectProgress?.message || "Processing..."}
                      </>
                    ) : isFinalizePending ? (
                      <>
                        <Save className="mr-1.5 h-3.5 w-3.5" />
                        Finalize This Image
                      </>
                    ) : (
                      <>
                        <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                        Auto-Detect
                      </>
                    )}
                  </Button>
                </motion.div>

                {detectionMode === "auto" && boxes.length > 0 && (
                  <motion.div {...buttonHover} {...buttonTap}>
                    <Button
                      variant={isAutoBoxCorrection ? "default" : "outline"}
                      size="sm"
                      onClick={handleToggleAutoBoxCorrection}
                      disabled={isDetecting}
                      className="font-bold"
                    >
                      <PencilRuler className="mr-1.5 h-3.5 w-3.5" />
                      {isAutoBoxCorrection ? "Finish Box Correction" : "Correct Selected Box"}
                    </Button>
                  </motion.div>
                )}

                {detectionMode === "auto" && boxes.length > 0 && (
                  <motion.div {...buttonHover} {...buttonTap}>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleDeleteSelectedAutoBox}
                      disabled={isDetecting || selectedBoxId === null}
                      className="font-bold text-destructive hover:bg-destructive/10 hover:text-destructive"
                    >
                      <XCircle className="mr-1.5 h-3.5 w-3.5" />
                      Delete Selected Box
                    </Button>
                  </motion.div>
                )}

                {detectionMode === "manual" && !isCurrentFinalized && selectedBoxId !== null && (
                  <motion.div {...buttonHover} {...buttonTap}>
                    <Button variant="outline" size="sm" onClick={handleDeleteSelectedAutoBox}
                      className="font-bold text-destructive hover:bg-destructive/10 hover:text-destructive">
                      <XCircle className="mr-1.5 h-3.5 w-3.5" />
                      Delete Selected Box
                    </Button>
                  </motion.div>
                )}

              </>
            ) : null}

            <motion.div {...buttonHover} {...buttonTap}>
              <Button
                variant="outline"
                size="sm"
                onClick={handleFinalizeAllWithBoxes}
                disabled={isBulkFinalizing || isDetecting || finalizableImageCount === 0}
                className="font-bold"
                title={
                  finalizableImageCount > 0
                    ? "Finalize every image that currently has accepted boxes"
                    : "No remaining images with boxes to finalize"
                }
              >
                {isBulkFinalizing ? (
                  <>
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    {bulkFinalizeProgress
                      ? `Finalizing ${bulkFinalizeProgress.current} of ${bulkFinalizeProgress.total}...`
                      : "Finalizing..."}
                  </>
                ) : (
                  <>
                    <Save className="mr-1.5 h-3.5 w-3.5" />
                    Finalize All With Boxes ({finalizableImageCount})
                  </>
                )}
              </Button>
            </motion.div>

            {finalizedImageCount > 0 && (
              <motion.div {...buttonHover} {...buttonTap}>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleUnfinalizeAllFinalized}
                  disabled={isBulkUnfinalizing || isDetecting}
                  className="font-bold"
                  title="Remove finalized lock from all finalized images"
                >
                  {isBulkUnfinalizing ? (
                    <>
                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                      Unfinalizing...
                    </>
                  ) : (
                    <>
                      <LockOpen className="mr-1.5 h-3.5 w-3.5" />
                      Unfinalize All Finalized ({finalizedImageCount})
                    </>
                  )}
                </Button>
              </motion.div>
            )}

            {hasAnnotations && (
              <motion.div {...buttonHover} {...buttonTap}>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={exportCurrent}
                  className="font-bold"
                >
                  Export Current (JSON)
                </Button>
              </motion.div>
            )}

            <motion.div {...buttonHover} {...buttonTap}>
              <Button
                variant="outline"
                size="sm"
                onClick={exportAll}
                className="font-bold"
              >
                Export All (JSON)
              </Button>
            </motion.div>

            <Tooltip>
              <TooltipTrigger asChild>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={toggleMagnifiedView}
                    aria-label="Magnify"
                  >
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                </motion.div>
              </TooltipTrigger>
              <TooltipContent>Magnify</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setConfirmOpen(true)}
                    aria-label="Delete"
                    className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </motion.div>
              </TooltipTrigger>
              <TooltipContent>Delete</TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* Progress bar for auto-detection */}
        {autoDetectProgress && (
          <div className="w-full space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] text-muted-foreground">{autoDetectProgress.message}</span>
              <span className="font-mono text-[11px] text-primary">{autoDetectProgress.percent}%</span>
            </div>
            <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-300"
                style={{ width: `${autoDetectProgress.percent}%` }}
              />
            </div>
          </div>
        )}

        {detectionMode === "auto" && isAutoBoxCorrection && (
          <p className="text-xs text-muted-foreground">
            Correction mode: select a box, then drag/resize it. Drag on empty area to redraw selected box.
          </p>
        )}

        {/* Image area */}
        <div className="relative flex flex-1 min-h-0 min-w-0 items-center justify-center overflow-hidden rounded-lg border border-border/50 bg-muted/15">
          {/* Previous button */}
          <div className="absolute left-2 top-1/2 z-20 -translate-y-1/2">
            <motion.div {...buttonHover} {...buttonTap}>
              <Button
                variant="outline"
                size="icon"
                onClick={handlePrev}
                disabled={totalImages <= 1}
                aria-label="Previous"
                className="h-9 w-9 bg-background/80 backdrop-blur-md border-border/50 shadow-lg disabled:opacity-30"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
            </motion.div>
          </div>

          {/* Image display - simple conditional rendering */}
          {current ? (
            <div key={current.id} className="relative h-full w-full min-h-0 min-w-0">
              <ImageLabeler
                key={current.id}
                imageURL={current.url}
                color={color}
                opacity={opacity}
                mode={isSwitchOn}
                detectionMode={detectionMode}
                autoCorrectionMode={!isCurrentFinalized && detectionMode === "auto" && isAutoBoxCorrection}
                imagePath={current.path ?? undefined}
                samEnabled={samEnabled}
                hideSegmentOutlines={isCurrentFinalized}
                lockBoxes={isCurrentFinalized || isCurrentFinalizing}
                orientationMode={activeOrientationMode}
                bilateralClassAxis={activeBilateralClassAxis}
                onFlipAll={flipAllBoxOrientations}
              />
              {/* Finalized overlay banner */}
              {isCurrentFinalized && (
                <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-center gap-1.5 bg-emerald-500/20 py-1.5 text-xs font-semibold text-emerald-300 backdrop-blur-sm">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Detection finalized — use landmark mode to annotate
                </div>
              )}
            </div>
          ) : (
            <div className="flex h-full w-full items-center justify-center text-muted-foreground">
              No image selected
            </div>
          )}

          {/* Next button */}
          <div className="absolute right-2 top-1/2 z-20 -translate-y-1/2">
            <motion.div {...buttonHover} {...buttonTap}>
              <Button
                variant="outline"
                size="icon"
                onClick={handleNext}
                disabled={totalImages <= 1}
                aria-label="Next"
                className="h-9 w-9 bg-background/80 backdrop-blur-md border-border/50 shadow-lg disabled:opacity-30"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </motion.div>
          </div>
        </div>

        {/* Delete confirmation dialog */}
        <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="font-bold">
                Delete this image?
              </DialogTitle>
              <DialogDescription>
                This will remove the current image and its labels from the
                session. This cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="gap-2 sm:gap-0">
              <Button
                variant="outline"
                onClick={() => setConfirmOpen(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDeleteImage}
              >
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {current && (
          <MagnifiedImageLabeler
            imageURL={current.url}
            color={color}
            opacity={opacity}
            open={isMagnified}
            onClose={toggleMagnifiedView}
            mode={isSwitchOn}
            schema={activeSchema}
            detectionMode={detectionMode}
            autoCorrectionMode={!isCurrentFinalized && detectionMode === "auto" && isAutoBoxCorrection}
            imagePath={current.path ?? undefined}
            samEnabled={samEnabled}
            hideSegmentOutlines={isCurrentFinalized}
            lockBoxes={isCurrentFinalized || isCurrentFinalizing}
            orientationMode={activeOrientationMode}
            bilateralClassAxis={activeBilateralClassAxis}
          />
        )}
      </Card>
  );
};

export default ImageLabelerCarousel;
