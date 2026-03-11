import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Trash2, ZoomIn, Loader2, Sparkles, PencilRuler, XCircle, Save, CheckCircle2, LockOpen } from "lucide-react";
import { toast } from "sonner";

import ImageLabeler from "./ImageLabeler";
import { DetectionMode, DetectionPreset } from "./DetectionModeSelector";
import MagnifiedImageLabeler from "./MagnifiedZoomLabeler";
import { UndoRedoClearContext } from "./UndoRedoClearContext";

import { AppDispatch, RootState } from "../state/store";
import {
  removeFile,
  updateBoxes,
  setImageFinalized,
  setImagesUnfinalized,
} from "../state/filesState/fileSlice";
import { AnnotatedImage, BoundingBox } from "../types/Image";

import { Button } from "@/Components/ui/button";
import { Card, CardContent } from "@/Components/ui/card";
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
  detectionPreset?: DetectionPreset;
  className?: string;
  samEnabled?: boolean;
}

const ImageLabelerCarousel: React.FC<ImageLabelerCarouselProps> = ({
  color,
  opacity,
  isSwitchOn,
  detectionMode = "manual",
  detectionPreset = "balanced",
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
  const [isUnfinalizingCurrent, setIsUnfinalizingCurrent] = useState(false);
  const [isBulkUnfinalizing, setIsBulkUnfinalizing] = useState(false);
  const [finalizedBoxSignatureByImageId, setFinalizedBoxSignatureByImageId] = useState<Record<number, string>>({});
  // Tracks which image IDs have been explicitly finalized by the user
  const [finalizedImageIds, setFinalizedImageIds] = useState<Set<number>>(new Set());
  const [finalizingImageIds, setFinalizingImageIds] = useState<Set<number>>(new Set());
  const [annotationHydratedIds, setAnnotationHydratedIds] = useState<Set<number>>(new Set());
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
      }))
      .sort((a, b) =>
        a.left - b.left ||
        a.top - b.top ||
        a.width - b.width ||
        a.height - b.height
      );
    return JSON.stringify(reduced);
  }, [current?.boxes]);

  // isCurrentFinalized: true if the user finalized this session OR the session was restored from disk as finalized.
  // Belt-and-suspenders: also check Redux directly by image id in case the context sync missed the update.
  const isCurrentFinalizedInRedux = useMemo(() => {
    if (!current) return false;
    const reduxImage = reduxFileArray.find((f) => f.id === current.id);
    return Boolean(reduxImage?.isFinalized);
  }, [reduxFileArray, current]);

  const isCurrentFinalized = current
    ? (finalizedImageIds.has(current.id) || Boolean(current.isFinalized) || isCurrentFinalizedInRedux)
    : false;
  const isCurrentFinalizing = current ? finalizingImageIds.has(current.id) : false;

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
            .map((img) => img.filename)
        );
        if (finalizedNames.size === 0) return;

        const candidates = finalizationCandidatesRef.current;
        setFinalizedImageIds((prev) => {
          const next = new Set(prev);
          candidates.forEach((img) => {
            if (!recentlyUnfinalizedIds.current.has(img.id) && finalizedNames.has(img.filename)) {
              next.add(img.id);
            }
          });
          return next;
        });

        candidates.forEach((img) => {
          if (
            !recentlyUnfinalizedIds.current.has(img.id) &&
            finalizedNames.has(img.filename) &&
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
  }, [activeSpeciesId, dispatch]);

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
        if (result.finalized) {
          dispatch(setImageFinalized({ id: current.id }));
          setFinalizedImageIds((prev) => new Set([...prev, current.id]));
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
  }, [current, annotationHydratedIds, dispatch]);

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
        if (result.finalized) {
          dispatch(setImageFinalized({ id: image.id }));
          setFinalizedImageIds((prev) => new Set([...prev, image.id]));
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
    [dispatch]
  );

  const getMatchingImageIds = useCallback((speciesId?: string, filename?: string): number[] => {
    const safeSpeciesId = String(speciesId || "");
    const safeFilename = String(filename || "").toLowerCase();
    return images
      .filter(
        (img) =>
          String(img.speciesId || "") === safeSpeciesId &&
          String(img.filename || "").toLowerCase() === safeFilename
      )
      .map((img) => img.id);
  }, [images]);

  const finalizeImageSegments = useCallback(async (
    image: AnnotatedImage | null
  ): Promise<{ ok: boolean; speciesId?: string; filename?: string; signature?: string; error?: string }> => {
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
        landmarks: (b.landmarks || [])
          .filter((lm) => Number.isFinite(Number(lm?.id)))
          .map((lm) => ({
            id: Number(lm.id),
            x: Number(lm.x),
            y: Number(lm.y),
            ...(lm.isSkipped ? { isSkipped: true } : {}),
          })),
      }));

    const imageId = image.id;
    try {
      const result = await window.api.sessionFinalizeAcceptedBoxes(
        image.speciesId,
        image.filename,
        acceptedBoxes,
        image.path
      );
      if (result.ok && result.segmentQueueState === "queued" && result.signature) {
        const matchingIds = getMatchingImageIds(image.speciesId, image.filename);
        setFinalizingImageIds((prev) => new Set([...prev, ...(matchingIds.length ? matchingIds : [imageId])]));
        return {
          ok: true,
          speciesId: image.speciesId,
          filename: image.filename,
          signature: result.signature,
        };
      }
      return {
        ok: false,
        speciesId: image.speciesId,
        filename: image.filename,
        error: result.error || "Segment finalization was not queued.",
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
  }, [ensureImageBoxesLoaded, getMatchingImageIds]);

  const finalizeCurrentImageSegments = useCallback(async (): Promise<{ ok: boolean; speciesId?: string; filename?: string; signature?: string; error?: string }> => {
    return finalizeImageSegments(current);
  }, [current, finalizeImageSegments]);

  const clearFinalizedLocalState = useCallback((imageIds: number[]) => {
    if (!imageIds.length) return;
    const idSet = new Set(imageIds);
    setFinalizedImageIds((prev) => {
      const next = new Set(prev);
      imageIds.forEach((id) => next.delete(id));
      return next;
    });
    setFinalizedBoxSignatureByImageId((prev) => {
      const next = { ...prev };
      idSet.forEach((id) => {
        delete next[id];
      });
      return next;
    });
  }, []);

  const clearFinalizingLocalState = useCallback((imageIds: number[]) => {
    if (!imageIds.length) return;
    setFinalizingImageIds((prev) => {
      const next = new Set(prev);
      imageIds.forEach((id) => next.delete(id));
      return next;
    });
  }, []);

  const waitForSegmentSaveOutcome = useCallback(async (
    speciesId: string,
    filename: string,
    signature?: string
  ) => {
    const isTerminal = (state: string) => state === "saved" || state === "skipped" || state === "failed";
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
      state: "idle" | "queued" | "running" | "saved" | "skipped" | "failed";
      signature?: string;
      updatedAt: string;
      reason?: string;
      expectedCount?: number;
      savedCount?: number;
    }>((resolve) => {
      const unsub = window.api.onSegmentSaveStatus((data) => {
        if (
          data.speciesId === speciesId &&
          data.filename === filename &&
          (!signature || data.signature === signature) &&
          isTerminal(data.state)
        ) {
          unsub();
          resolve(data);
        }
      });
      setTimeout(() => {
        unsub();
        resolve({
          speciesId,
          filename,
          state: "failed",
          updatedAt: new Date().toISOString(),
          reason: "timeout",
        });
      }, 120_000);
    });
  }, []);

  useEffect(() => {
    const unsub = window.api.onSegmentSaveStatus((data) => {
      const matchingIds = getMatchingImageIds(data.speciesId, data.filename);
      if (matchingIds.length === 0) return;

      if (data.state === "saved") {
        clearFinalizingLocalState(matchingIds);
        setFinalizedImageIds((prev) => new Set([...prev, ...matchingIds]));
        if (data.signature) {
          setFinalizedBoxSignatureByImageId((prev) => {
            const next = { ...prev };
            matchingIds.forEach((id) => {
              next[id] = data.signature as string;
            });
            return next;
          });
        }
        matchingIds.forEach((id) => dispatch(setImageFinalized({ id })));
        return;
      }

      if (data.state === "failed" || data.state === "skipped") {
        clearFinalizingLocalState(matchingIds);
        clearFinalizedLocalState(matchingIds);
        dispatch(setImagesUnfinalized({ ids: matchingIds }));
      }
    });
    return () => {
      unsub();
    };
  }, [clearFinalizedLocalState, clearFinalizingLocalState, dispatch, getMatchingImageIds]);

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
      clearFinalizedLocalState(idsToClear);
      clearFinalizingLocalState(idsToClear);
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
  }, [clearFinalizedLocalState, clearFinalizingLocalState, current, dispatch, images]);

  const finalizableImages = useMemo(() => {
    return images.filter((img) => {
      const hasValidBoxes =
        (img.boxes || []).some((b) => b.width > 0 && b.height > 0) ||
        Boolean(img.hasBoxes);
      if (!hasValidBoxes || !img.speciesId || !img.filename) return false;
      const finalized = Boolean(img.isFinalized) || finalizedImageIds.has(img.id);
      const finalizing = finalizingImageIds.has(img.id);
      return !finalized && !finalizing;
    });
  }, [images, finalizedImageIds, finalizingImageIds]);

  const finalizableImageCount = finalizableImages.length;
  const finalizedImages = useMemo(() => {
    return images.filter((img) => {
      if (!img.speciesId || !img.filename) return false;
      return Boolean(img.isFinalized) || finalizedImageIds.has(img.id);
    });
  }, [images, finalizedImageIds]);
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
      clearFinalizedLocalState(succeededIds);
      clearFinalizingLocalState(succeededIds);
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
  }, [activeSpeciesId, clearFinalizedLocalState, clearFinalizingLocalState, dispatch, finalizedImages]);

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
        maxObjects: 25,
        detectionMode: "auto",
        detectionPreset,
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
  }, [current, className, samEnabled, detectionPreset, setBoxesFromSuperAnnotation]);

  const handleAutoDetectOrFinalize = useCallback(async () => {
    if (isFinalizePending) {
      const queued = await finalizeCurrentImageSegments();
      if (!queued.ok || !queued.speciesId || !queued.filename) {
        toast.error(queued.error || "Failed to finalize accepted boxes.");
        return;
      }
      const outcome = await waitForSegmentSaveOutcome(
        queued.speciesId,
        queued.filename,
        queued.signature
      );
      if (outcome.state === "saved") {
        toast.success("Finalized accepted boxes for this image.");
      } else {
        toast.error(
          outcome.reason ||
            (outcome.expectedCount != null && outcome.savedCount != null
              ? `Saved ${outcome.savedCount}/${outcome.expectedCount} segments.`
              : "Failed to finalize accepted boxes.")
        );
      }
      return;
    }
    await handleAutoDetect();
  }, [handleAutoDetect, finalizeCurrentImageSegments, isFinalizePending, waitForSegmentSaveOutcome]);

  const handleFinalizeAllWithBoxes = useCallback(async () => {
    if (finalizableImageCount === 0) {
      toast.info("No remaining images with boxes to finalize.");
      return;
    }

    setIsBulkFinalizing(true);
    const pendingRequests: Array<{ speciesId: string; filename: string; signature?: string }> = [];
    let failCount = 0;

    try {
      for (const image of finalizableImages) {
        const request = await finalizeImageSegments(image);
        if (request.ok && request.speciesId && request.filename) {
          pendingRequests.push({
            speciesId: request.speciesId,
            filename: request.filename,
            signature: request.signature,
          });
        } else {
          failCount += 1;
        }
      }
    } catch (err) {
      console.error("Finalize-all error:", err);
      setIsBulkFinalizing(false);
      toast.error("Failed to finalize images with boxes.");
      return;
    }

    if (pendingRequests.length === 0) {
      setIsBulkFinalizing(false);
      toast.error("Failed to finalize images with boxes.");
      return;
    }

    const outcomes = await Promise.all(
      pendingRequests.map((request) =>
        waitForSegmentSaveOutcome(request.speciesId, request.filename, request.signature)
      )
    );
    const successCount = outcomes.filter((outcome) => outcome.state === "saved").length;
    failCount += outcomes.length - successCount;

    setIsBulkFinalizing(false);

    if (successCount > 0 && failCount === 0) {
      toast.success(`Finalized ${successCount} image(s) with accepted boxes.`);
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
  const activeOrientationMode = activeSpeciesForSchema?.orientationPolicy?.mode;

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
      <Card className="w-full max-w-[900px] border-border/50 bg-card/50 backdrop-blur-sm">
        <CardContent className="p-6 text-center">
          <h2 className="mb-2 text-lg font-bold text-foreground">
            No images available.
          </h2>
          <p className="mx-auto max-w-[520px] text-sm text-muted-foreground">
            Press <strong>Ctrl+N</strong> to upload images, or use the left
            sidebar to begin labeling.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
      <Card className="flex h-full w-full min-h-0 min-w-0 flex-col gap-3 border-border/50 bg-card/50 p-4 backdrop-blur-sm">
        {/* Toolbar */}
        <div className="flex w-full flex-col gap-2 md:flex-row md:items-start md:justify-between">
          <div className="min-w-0 md:pr-3">
            <p className="text-sm font-bold text-foreground">
              Image {currentIndex + 1} / {totalImages}
            </p>
            <p className="text-xs text-muted-foreground">
              Use ← / → to navigate • Ctrl+N to add
            </p>
          </div>

          <div className="flex min-w-0 flex-1 flex-wrap items-center justify-start gap-2 md:justify-end">
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
                    Finalizing...
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
          <div className="w-full">
            <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
              <span>{autoDetectProgress.message}</span>
              <span>{autoDetectProgress.percent}%</span>
            </div>
            <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-300"
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
        <div className="relative flex flex-1 min-h-0 min-w-0 items-center justify-center overflow-hidden rounded-xl border bg-background">
          {/* Previous button */}
          <div className="absolute left-2 top-1/2 z-20 -translate-y-1/2">
            <motion.div {...buttonHover} {...buttonTap}>
              <Button
                variant="outline"
                size="icon"
                onClick={handlePrev}
                disabled={totalImages <= 1}
                aria-label="Previous"
                className="bg-background/90 backdrop-blur-sm shadow-md"
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
                className="bg-background/90 backdrop-blur-sm shadow-md"
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
          />
        )}
      </Card>
  );
};

export default ImageLabelerCarousel;
