import React, { useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Trash2, ZoomIn, Loader2, Sparkles, Crosshair, PencilRuler, XCircle } from "lucide-react";
import { toast } from "sonner";

import ImageLabeler from "./ImageLabeler";
import { DetectionMode, DetectionPreset } from "./DetectionModeSelector";
import MagnifiedImageLabeler from "./MagnifiedZoomLabeler";
import { UndoRedoClearContext } from "./UndoRedoClearContext";

import { AppDispatch, RootState } from "../state/store";
import { removeFile, updateBoxes } from "../state/filesState/fileSlice";
import { BoundingBox } from "../types/Image";

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
  const [isTrainingDetection, setIsTrainingDetection] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<{ message: string; percent: number } | null>(null);
  const [isAutoBoxCorrection, setIsAutoBoxCorrection] = useState(false);

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

  useEffect(() => {
    if (detectionMode !== "auto") {
      setIsAutoBoxCorrection(false);
    }
  }, [detectionMode, current?.id]);

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
    if (!promptClass) {
      toast.info("Enter an object class prompt first (for example: fish or leaf).");
      return;
    }

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
      }, current.speciesId);

      if (result.ok && Array.isArray(result.objects) && result.objects.length > 0) {
        setBoxesFromSuperAnnotation(result.objects);
        setIsAutoBoxCorrection(false);
        toast.success(`Detected ${result.objects.length} object${result.objects.length > 1 ? "s" : ""}.`);
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

  // Count images with bounding box annotations for YOLO training
  const annotatedImageCount = useMemo(() => {
    return images.filter((img) => (img.boxes || []).length > 0).length;
  }, [images]);

  const canTrainDetection = annotatedImageCount >= 10;

  // Get speciesId from current image
  const activeSpeciesId = current?.speciesId;

  // Handle YOLO detection model training
  const handleTrainDetection = useCallback(async () => {
    const promptClass = className.trim();
    if (!activeSpeciesId || !promptClass) {
      toast.info("Enter an object class name and ensure you're in an active session.");
      return;
    }
    if (!canTrainDetection) {
      toast.info(`Need at least 10 annotated images to train. Currently have ${annotatedImageCount}.`);
      return;
    }

    setIsTrainingDetection(true);
    setTrainingProgress({ message: "Starting training...", percent: 0 });

    const unsubscribe = window.api.onSuperAnnotateProgress((data) => {
      setTrainingProgress({ message: data.message, percent: data.percent });
    });

    try {
      const result = await window.api.trainYolo(activeSpeciesId, promptClass, undefined, detectionPreset);
      if (result.ok) {
        const mapSuffix = result.candidateMap50 != null ? ` mAP50=${(result.candidateMap50 * 100).toFixed(1)}%.` : "";
        toast.success(
          result.promoted
            ? `Detection model v${result.version ?? "?"} promoted.${mapSuffix}`
            : `Detection model v${result.version ?? "?"} kept as candidate (not promoted).${mapSuffix}`
        );
      } else {
        toast.error(result.error || "Detection training failed.");
      }
    } catch (err) {
      console.error("Detection training failed:", err);
      toast.error("Detection training request failed.");
    } finally {
      unsubscribe();
      setIsTrainingDetection(false);
      setTrainingProgress(null);
    }
  }, [activeSpeciesId, className, canTrainDetection, annotatedImageCount, detectionPreset]);

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
            {/* Manual mode has no detect button — user draws boxes by dragging */}

            {/* Auto-Detect button - auto mode (AI) */}
            {detectionMode === "auto" && (
              <>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleAutoDetect}
                    disabled={isDetecting || !current || !className.trim()}
                    className="font-bold"
                  >
                    {isDetecting ? (
                      <>
                        <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                        {autoDetectProgress?.message || "Processing..."}
                      </>
                    ) : (
                      <>
                        <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                        Auto-Detect
                      </>
                    )}
                  </Button>
                </motion.div>

                {boxes.length > 0 && (
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

                {boxes.length > 0 && (
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
              </>
            )}

            {/* Train Detection Model button — visible when 10+ images annotated */}
            {canTrainDetection && activeSpeciesId && className.trim() && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <motion.div {...buttonHover} {...buttonTap}>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleTrainDetection}
                      disabled={isTrainingDetection || isDetecting}
                      className="font-bold"
                    >
                      {isTrainingDetection ? (
                        <>
                          <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                          {trainingProgress?.message || "Training..."}
                        </>
                      ) : (
                        <>
                          <Crosshair className="mr-1.5 h-3.5 w-3.5" />
                          Train Detection
                        </>
                      )}
                    </Button>
                  </motion.div>
                </TooltipTrigger>
                <TooltipContent>
                  Fine-tune detection model on {annotatedImageCount} annotated images
                </TooltipContent>
              </Tooltip>
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

        {/* Progress bar for YOLO training */}
        {trainingProgress && (
          <div className="w-full">
            <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
              <span>{trainingProgress.message}</span>
              <span>{trainingProgress.percent}%</span>
            </div>
            <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500 rounded-full transition-all duration-300"
                style={{ width: `${trainingProgress.percent}%` }}
              />
            </div>
          </div>
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
            <div key={current.id} className="h-full w-full min-h-0 min-w-0">
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
                autoCorrectionMode={detectionMode === "auto" && isAutoBoxCorrection}
              />
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
            detectionMode={detectionMode}
            autoCorrectionMode={detectionMode === "auto" && isAutoBoxCorrection}
          />
        )}
      </Card>
    </TooltipProvider>
  );
};

export default ImageLabelerCarousel;
