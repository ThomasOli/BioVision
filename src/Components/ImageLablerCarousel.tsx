import React, { useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useDispatch } from "react-redux";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight, Trash2, ZoomIn } from "lucide-react";

import ImageLabeler from "./ImageLabeler";
import MagnifiedImageLabeler from "./MagnifiedZoomLabeler";
import { UndoRedoClearContext } from "./UndoRedoClearContext";

import { AppDispatch } from "../state/store";
import { removeFile, updateLabels } from "../state/filesState/fileSlice";

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
import { carouselImage, buttonHover, buttonTap, modalContent } from "@/lib/animations";

interface ImageLabelerCarouselProps {
  color: string;
  opacity: number;
  isSwitchOn: boolean;
}

const ImageLabelerCarousel: React.FC<ImageLabelerCarouselProps> = ({
  color,
  opacity,
  isSwitchOn,
}) => {
  const { images, setSelectedImage } = useContext(UndoRedoClearContext);
  const dispatch = useDispatch<AppDispatch>();

  const totalImages = images.length;

  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isMagnified, setIsMagnified] = useState<boolean>(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [direction, setDirection] = useState(0);

  useEffect(() => {
    if (totalImages === 0) {
      setCurrentIndex(0);
      return;
    }
    setCurrentIndex((prev) => Math.min(prev, totalImages - 1));
  }, [totalImages]);

  const current = useMemo(
    () => (totalImages ? images[currentIndex] : null),
    [images, currentIndex, totalImages]
  );
  const hasLandmarks = Boolean(current?.labels?.length);

  const handleUpdateLabels = useCallback(
    (id: number, labels: { x: number; y: number; id: number }[]) => {
      dispatch(updateLabels({ id, labels }));
    },
    [dispatch]
  );

  const handleDeleteImage = useCallback(
    (id: number) => {
      dispatch(removeFile(id));
      setCurrentIndex((prevIndex) => {
        const newTotal = totalImages - 1;
        if (newTotal <= 0) return 0;
        if (prevIndex >= newTotal) return newTotal - 1;
        return prevIndex;
      });
    },
    [dispatch, totalImages]
  );

  const handleNext = useCallback(() => {
    if (totalImages <= 1) return;
    setDirection(1);
    setCurrentIndex((prev) => (prev + 1) % totalImages);
    setSelectedImage((prev) => (prev + 1) % totalImages);
  }, [totalImages, setSelectedImage]);

  const handlePrev = useCallback(() => {
    if (totalImages <= 1) return;
    setDirection(-1);
    setCurrentIndex((prev) => (prev - 1 + totalImages) % totalImages);
    setSelectedImage((prev) => (prev - 1 + totalImages) % totalImages);
  }, [totalImages, setSelectedImage]);

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

  const exportCurrent = useCallback(async () => {
    if (!current || !current.labels?.length) return;

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
      points: current.labels.map(({ x, y, id }: { x: number; y: number; id: number }) => ({
        x: Math.round(x),
        y: Math.round(y),
        id,
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
    const data = images.map(({ id, url, labels }) => ({ id, url, labels }));
    const jsonData = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonData], { type: "application/json" });
    const urlBlob = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = urlBlob;
    a.download = `all_labeled_data_${Date.now()}.json`;
    a.click();

    URL.revokeObjectURL(urlBlob);
  }, [images]);

  if (totalImages === 0) {
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
        <div className="flex w-full items-center justify-between gap-2">
          <div className="min-w-0">
            <p className="text-sm font-bold text-foreground">
              Image {currentIndex + 1} / {totalImages}
            </p>
            <p className="text-xs text-muted-foreground">
              Use ← / → to navigate • Ctrl+N to add
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            {hasLandmarks && (
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

        {/* Image area */}
        <div className="relative flex flex-1 min-h-0 min-w-0 items-center justify-center overflow-hidden rounded-xl border bg-background">
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              variant="outline"
              size="icon"
              onClick={handlePrev}
              disabled={totalImages === 1}
              aria-label="Previous"
              className="absolute left-2 top-1/2 z-10 -translate-y-1/2 bg-background/90 backdrop-blur-sm"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
          </motion.div>

          <AnimatePresence mode="wait" custom={direction}>
            {current && (
              <motion.div
                key={current.id}
                custom={direction}
                variants={carouselImage}
                initial="enter"
                animate="center"
                exit="exit"
                className="h-full w-full min-h-0 min-w-0"
              >
                <ImageLabeler
                  imageURL={current.url}
                  onPointsChange={(newPoints) =>
                    handleUpdateLabels(current.id, newPoints)
                  }
                  color={color}
                  opacity={opacity}
                  mode={isSwitchOn}
                />
              </motion.div>
            )}
          </AnimatePresence>

          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              variant="outline"
              size="icon"
              onClick={handleNext}
              disabled={totalImages === 1}
              aria-label="Next"
              className="absolute right-2 top-1/2 z-10 -translate-y-1/2 bg-background/90 backdrop-blur-sm"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </motion.div>
        </div>

        {/* Delete confirmation dialog */}
        <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
          <DialogContent asChild>
            <motion.div
              variants={modalContent}
              initial="hidden"
              animate="visible"
              exit="exit"
            >
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
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    onClick={() => setConfirmOpen(false)}
                  >
                    Cancel
                  </Button>
                </motion.div>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="destructive"
                    onClick={() => {
                      if (current) handleDeleteImage(current.id);
                      setConfirmOpen(false);
                    }}
                  >
                    Delete
                  </Button>
                </motion.div>
              </DialogFooter>
            </motion.div>
          </DialogContent>
        </Dialog>

        {current && (
          <MagnifiedImageLabeler
            imageURL={current.url}
            onPointsChange={(newPoints) =>
              handleUpdateLabels(current.id, newPoints)
            }
            color={color}
            opacity={opacity}
            open={isMagnified}
            onClose={toggleMagnifiedView}
            mode={isSwitchOn}
          />
        )}
      </Card>
    </TooltipProvider>
  );
};

export default ImageLabelerCarousel;
