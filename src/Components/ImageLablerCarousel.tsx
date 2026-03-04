import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Trash2, ZoomIn, Loader2, Sparkles, PencilRuler, XCircle, Save, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";

import ImageLabeler from "./ImageLabeler";
import { DetectionMode, DetectionPreset } from "./DetectionModeSelector";
import MagnifiedImageLabeler from "./MagnifiedZoomLabeler";
import { UndoRedoClearContext } from "./UndoRedoClearContext";

import { AppDispatch, RootState } from "../state/store";
import { removeFile, updateBoxes, setImageFinalized } from "../state/filesState/fileSlice";
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
  TooltipProvider,
} from "@/Components/ui/tooltip";
import { buttonHover, buttonTap } from "@/lib/animations";

interface ImageLabelerCarouselProps {
  color: string;
  opacity: number;
  isSwitchOn: boolean;
  detectionMode?: DetectionMode;
  confThreshold?: number;
  detectionPreset?: DetectionPreset;
  className?: string;
  samEnabled?: boolean;
}

const ImageLabelerCarousel: React.FC<ImageLabelerCarouselProps> = ({
  color,
  opacity,
  isSwitchOn,
  detectionMode = "manual",
  confThreshold = 0.5,
  detectionPreset = "balanced",
  className = "",
  samEnabled = false,
}) => {
  const {
    images,
    setImages,
    boxes,
    selectedBoxId,
    setSelectedImage,
    setBoxesFromSuperAnnotation,
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
  const [isSegmentingBoxes, setIsSegmentingBoxes] = useState(false);
  const [segmentProgress, setSegmentProgress] = useState<{ current: number; total: number } | null>(null);
  const [isBulkFinalizing, setIsBulkFinalizing] = useState(false);
  const [finalizedBoxSignatureByImageId, setFinalizedBoxSignatureByImageId] = useState<Record<number, string>>({});
  // Tracks which image IDs have been explicitly finalized by the user
  const [finalizedImageIds, setFinalizedImageIds] = useState<Set<number>>(new Set());
  const [annotationHydratedIds, setAnnotationHydratedIds] = useState<Set<number>>(new Set());
  const annotationLoadInFlight = useRef<Set<number>>(new Set());

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
  // Belt-and-suspenders: also check Redux directly by filename in case the context sync missed the update.
  const isCurrentFinalizedInRedux = useMemo(() => {
    if (!current?.filename || !current?.speciesId) return false;
    return reduxFileArray.some(
      (f) => f.filename === current.filename && f.speciesId === current.speciesId && Boolean(f.isFinalized)
    );
  }, [reduxFileArray, current?.filename, current?.speciesId]);

  const isCurrentFinalized = current
    ? (finalizedImageIds.has(current.id) || Boolean(current.isFinalized) || isCurrentFinalizedInRedux)
    : false;

  const isFinalizePending = Boolean(
    current &&
    !isCurrentFinalized &&
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

        setFinalizedImageIds((prev) => {
          const next = new Set(prev);
          finalizationCandidates.forEach((img) => {
            if (finalizedNames.has(img.filename)) {
              next.add(img.id);
            }
          });
          return next;
        });

        finalizationCandidates.forEach((img) => {
          if (finalizedNames.has(img.filename) && !img.isFinalized) {
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
  }, [activeSpeciesId, finalizationCandidates, dispatch]);

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

  const handleUpdateBoxes = useCallback(
    (id: number, boxes: BoundingBox[]) => {
      dispatch(updateBoxes({ id, boxes }));
    },
    [dispatch]
  );

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

  const finalizeImageSegments = useCallback(async (image: AnnotatedImage | null): Promise<boolean> => {
    if (!image?.speciesId || !image?.filename) return false;
    const sourceBoxes = await ensureImageBoxesLoaded(image);

    // Compute class_ids from placed landmarks via the backend, then persist them
    // so the training pipeline (prepare_dataset.py / export_obb_dataset) can use them.
    const classIdMap = new Map<number, number>();
    const boxesWithObb = (sourceBoxes || []).filter((b) => b.obbCorners && b.obbCorners.length === 4);
    if (boxesWithObb.length > 0 && image.speciesId) {
      try {
        const tagResult = await window.api.tagClassIds(image.speciesId, sourceBoxes || []);
        if (tagResult.ok && tagResult.taggedBoxes) {
          for (const { id, class_id } of tagResult.taggedBoxes) {
            classIdMap.set(id, class_id);
          }
          // Merge class_ids into boxes and save so payload.boxes has them on disk
          const updatedBoxes = (sourceBoxes || []).map((b) => ({
            ...b,
            ...(classIdMap.has(b.id) ? { class_id: classIdMap.get(b.id) } : {}),
          }));
          dispatch(updateBoxes({ id: image.id, boxes: updatedBoxes }));
          await window.api.sessionSaveAnnotations(image.speciesId, image.filename, updatedBoxes);
        }
      } catch (_) {
        // Non-fatal: class_id will default to 0 in training pipeline
      }
    }

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
    const sig = JSON.stringify(
      acceptedBoxes
        .map((b) => ({
          left: b.left,
          top: b.top,
          width: b.width,
          height: b.height,
        }))
        .slice()
        .sort((a, b) =>
          a.left - b.left ||
          a.top - b.top ||
          a.width - b.width ||
          a.height - b.height
        )
    );

    try {
      const result = await window.api.sessionFinalizeAcceptedBoxes(
        image.speciesId,
        image.filename,
        acceptedBoxes,
        image.path
      );
      if (result.ok) {
        setFinalizedBoxSignatureByImageId((prev) => ({ ...prev, [imageId]: sig }));
        setFinalizedImageIds((prev) => new Set([...prev, imageId]));
        dispatch(setImageFinalized({ id: imageId }));
        return true;
      }
      return false;
    } catch (err) {
      console.error("Failed to finalize accepted boxes:", err);
      return false;
    }
  }, [dispatch, ensureImageBoxesLoaded]);

  const finalizeCurrentImageSegments = useCallback(async (): Promise<boolean> => {
    return finalizeImageSegments(current);
  }, [current, finalizeImageSegments]);

  const finalizableImages = useMemo(() => {
    return images.filter((img) => {
      const hasValidBoxes =
        (img.boxes || []).some((b) => b.width > 0 && b.height > 0) ||
        Boolean(img.hasBoxes);
      if (!hasValidBoxes || !img.speciesId || !img.filename) return false;
      const finalized = Boolean(img.isFinalized) || finalizedImageIds.has(img.id);
      return !finalized;
    });
  }, [images, finalizedImageIds]);

  const finalizableImageCount = finalizableImages.length;
  const segmentableImageCount = useMemo(() => {
    return images.filter((img) => {
      const imgPath = img.path || img.diskPath;
      const hasValidBoxes =
        (img.boxes || []).some((b) => b.width > 0 && b.height > 0) ||
        Boolean(img.hasBoxes);
      return Boolean(imgPath) && hasValidBoxes;
    }).length;
  }, [images]);

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

  // Handle AI auto-detection via SuperAnnotator pipeline (YOLO-World + optional SAM2)
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
        confThreshold,
        samEnabled,
        maxObjects: 25,
        detectionMode: "auto",
        detectionPreset,
        useOrientationHint: true,
      }, current.speciesId);

      if (result.ok && Array.isArray(result.objects) && result.objects.length > 0) {
        setBoxesFromSuperAnnotation(result.objects);
        setIsAutoBoxCorrection(false);
        const isOpenCv = result.detection_method === "opencv" || result.detection_method === "opencv_fallback";
        if (isOpenCv) {
          toast.success(
            `Detected ${result.objects.length} object${result.objects.length > 1 ? "s" : ""} using OpenCV (YOLO unavailable).`
          );
        } else {
          toast.success(`Detected ${result.objects.length} object${result.objects.length > 1 ? "s" : ""}.`);
        }
      } else if (result.ok) {
        toast.info("No objects detected. Try a different class prompt or lower confidence.");
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
  }, [current, className, confThreshold, samEnabled, detectionPreset, setBoxesFromSuperAnnotation]);

  const handleAutoDetectOrFinalize = useCallback(async () => {
    if (isFinalizePending) {
      const ok = await finalizeCurrentImageSegments();
      if (ok) {
        toast.success("Finalized accepted boxes for this image.");
      } else {
        toast.error("Failed to finalize accepted boxes.");
      }
      return;
    }
    await handleAutoDetect();
  }, [isFinalizePending, finalizeCurrentImageSegments, handleAutoDetect]);

  const handleFinalizeAllWithBoxes = useCallback(async () => {
    if (finalizableImageCount === 0) {
      toast.info("No remaining images with boxes to finalize.");
      return;
    }

    setIsBulkFinalizing(true);
    let successCount = 0;
    let failCount = 0;
    try {
      for (const image of finalizableImages) {
        const ok = await finalizeImageSegments(image);
        if (ok) successCount += 1;
        else failCount += 1;
      }
    } finally {
      setIsBulkFinalizing(false);
    }

    if (successCount > 0 && failCount === 0) {
      toast.success(`Finalized ${successCount} image(s) with accepted boxes.`);
    } else if (successCount > 0) {
      toast.warning(`Finalized ${successCount} image(s), ${failCount} failed.`);
    } else {
      toast.error("Failed to finalize images with boxes.");
    }
  }, [finalizableImageCount, finalizableImages, finalizeImageSegments]);

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

  const handleSegmentAllBoxes = useCallback(async () => {
    if (!samEnabled) {
      toast.info("Enable SAM2 first, then run segmentation.");
      return;
    }
    const candidates = images
      .map((img) => {
        const imgPath = img.path || img.diskPath;
        const validBoxes = (img.boxes || []).filter((b) => b.width > 0 && b.height > 0);
        return {
          image: img,
          imagePath: imgPath,
          validBoxes,
        };
      })
      .filter((entry) => Boolean(entry.imagePath) && entry.validBoxes.length > 0);

    if (candidates.length === 0) {
      toast.info("No images with valid bounding boxes to segment.");
      return;
    }

    const totalBoxes = candidates.reduce((sum, entry) => sum + entry.validBoxes.length, 0);
    setIsSegmentingBoxes(true);
    setSegmentProgress({ current: 0, total: totalBoxes });
    let success = 0;
    let failed = 0;
    let processed = 0;
    let imagesUpdated = 0;
    const errorCounts = new Map<string, number>();
    const updatedBoxesByImageId = new Map<number, BoundingBox[]>();

    try {
      for (const entry of candidates) {
        const imagePath = entry.imagePath!;
        let localBoxes = (entry.image.boxes || []).map((b) => ({ ...b }));
        let imageHadUpdate = false;

        for (const box of entry.validBoxes) {
          processed += 1;
          setSegmentProgress({ current: processed, total: totalBoxes });

          const left = Math.round(box.left);
          const top = Math.round(box.top);
          const width = Math.max(1, Math.round(box.width));
          const height = Math.max(1, Math.round(box.height));

          try {
            const result = await window.api.resegmentBox(imagePath, [
              left,
              top,
              left + width,
              top + height,
            ]);
            if (result.ok && Array.isArray(result.maskOutline) && result.maskOutline.length > 2) {
              localBoxes = localBoxes.map((candidateBox) =>
                candidateBox.id === box.id
                  ? {
                      ...candidateBox,
                      maskOutline: result.maskOutline,
                    }
                  : candidateBox
              );
              success += 1;
              imageHadUpdate = true;
            } else {
              failed += 1;
              const reason = (result.error || "unknown_error").trim();
              errorCounts.set(reason, (errorCounts.get(reason) || 0) + 1);
            }
          } catch (err) {
            console.error(`SAM segmentation failed for box ${box.id}:`, err);
            failed += 1;
            const reason = err instanceof Error ? err.message : "request_failed";
            errorCounts.set(reason, (errorCounts.get(reason) || 0) + 1);
          }
        }

        if (imageHadUpdate) {
          updatedBoxesByImageId.set(entry.image.id, localBoxes);
          imagesUpdated += 1;
        }
      }

      if (updatedBoxesByImageId.size > 0) {
        setImages((prev) =>
          prev.map((img) => {
            const updatedBoxes = updatedBoxesByImageId.get(img.id);
            return updatedBoxes ? { ...img, boxes: updatedBoxes } : img;
          })
        );

        const persistOps: Promise<unknown>[] = [];
        for (const [imageId, updatedBoxes] of updatedBoxesByImageId.entries()) {
          dispatch(updateBoxes({ id: imageId, boxes: updatedBoxes }));
          const imageMeta = images.find((img) => img.id === imageId);
          if (imageMeta?.speciesId && imageMeta?.filename) {
            persistOps.push(
              window.api.sessionSaveAnnotations(imageMeta.speciesId, imageMeta.filename, updatedBoxes)
            );
          }
        }
        if (persistOps.length > 0) {
          await Promise.allSettled(persistOps);
        }
      }
    } finally {
      setIsSegmentingBoxes(false);
      setSegmentProgress(null);
    }

    if (success > 0 && failed === 0) {
      toast.success(
        `Segmented ${success} box${success === 1 ? "" : "es"} across ${imagesUpdated} image${imagesUpdated === 1 ? "" : "s"}.`
      );
    } else if (success > 0) {
      const topError = [...errorCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
      toast.warning(
        `Segmented ${success} box${success === 1 ? "" : "es"} across ${imagesUpdated} image${imagesUpdated === 1 ? "" : "s"}; ${failed} failed` +
          (topError ? ` (${topError})` : ".")
      );
    } else {
      const topError = [...errorCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
      toast.error(
        `SAM2 did not produce segments for the current boxes` +
          (topError ? `: ${topError}` : ".")
      );
    }
  }, [samEnabled, images, setImages, dispatch]);

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
    <TooltipProvider>
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
            {isCurrentFinalized ? (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-bold text-emerald-400 ring-1 ring-emerald-500/40">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Detection Finalized
                </span>
                <span className="text-xs text-muted-foreground">Landmark mode only</span>
              </div>
            ) : (detectionMode === "auto" || isFinalizePending) ? (
              /* Finalize shared across modes; auto-detect/correction controls in auto mode only */
              <>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant={isFinalizePending ? "default" : "outline"}
                    size="sm"
                    onClick={handleAutoDetectOrFinalize}
                    disabled={isDetecting || !current}
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

            {samEnabled && segmentableImageCount > 0 && (
              <motion.div {...buttonHover} {...buttonTap}>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSegmentAllBoxes}
                  disabled={isDetecting || isSegmentingBoxes}
                  className="font-bold"
                  title="Run SAM2 on every accepted box in all images"
                >
                  {isSegmentingBoxes ? (
                    <>
                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                      {segmentProgress
                        ? `Segmenting ${segmentProgress.current}/${segmentProgress.total}`
                        : "Segmenting..."}
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                      Segment All Boxes ({segmentableImageCount} img)
                    </>
                  )}
                </Button>
              </motion.div>
            )}

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
                onBoxesChange={(newBoxes) =>
                  handleUpdateBoxes(current.id, newBoxes)
                }
                color={color}
                opacity={opacity}
                mode={isSwitchOn}
                detectionMode={detectionMode}
                autoCorrectionMode={!isCurrentFinalized && detectionMode === "auto" && isAutoBoxCorrection}
                imagePath={current.path ?? undefined}
                samEnabled={samEnabled}
                hideSegmentOutlines={isCurrentFinalized}
                lockBoxes={isCurrentFinalized}
                orientationMode={activeOrientationMode}
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
            onBoxesChange={(newBoxes) =>
              handleUpdateBoxes(current.id, newBoxes)
            }
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
            lockBoxes={isCurrentFinalized}
            orientationMode={activeOrientationMode}
          />
        )}
      </Card>
    </TooltipProvider>
  );
};

export default ImageLabelerCarousel;
