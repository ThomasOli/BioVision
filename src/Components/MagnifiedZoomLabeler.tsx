import React, { useRef, useCallback, useEffect, useContext, useState, useMemo } from "react";
import { motion } from "framer-motion";
import { X, Info } from "lucide-react";
import { Stage, Layer, Image as KonvaImage, Circle, Text } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";
import { LandmarkPlacementGuide } from "./LandmarkPlacementGuide";
import { Button } from "@/Components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogClose,
} from "@/Components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/Components/ui/tooltip";
import { modalContent } from "@/lib/animations";
import { BoundingBox, LandmarkSchema } from "../types/Image";
import Konva from "konva";
import { DEFAULT_SCHEMAS } from "@/data/defaultSchemas";

interface MagnifiedImageLabelerProps {
  imageURL: string;
  onBoxesChange: (boxes: BoundingBox[]) => void;
  color: string;
  opacity: number;
  open: boolean;
  onClose: () => void;
  mode: boolean; // View-only mode
  schema?: LandmarkSchema;
}

const MagnifiedImageLabeler: React.FC<MagnifiedImageLabelerProps> = ({
  imageURL,
  color,
  opacity,
  open,
  onClose,
  mode,
  schema,
}) => {
  const {
    addLandmark,
    boxes,
    selectedBoxId,
    addBox,
    selectBox,
    skipLandmark,
  } = useContext(UndoRedoClearContext);

  const [image, imageDimensions, imageError] = useImageLoader(imageURL);
  const stageRef = useRef<Konva.Stage>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  // Track if we just created a box to avoid double-adding landmarks
  const pendingBoxRef = useRef<{ x: number; y: number } | null>(null);

  // Show/hide landmark guide
  const [showGuide, setShowGuide] = useState(true);

  // User-controlled zoom
  const [userZoom, setUserZoom] = useState(1);
  const MIN_ZOOM = 0.5;
  const MAX_ZOOM = 5;

  // Drag/pan state for scrollable container
  const [isDragging, setIsDragging] = useState(false);
  const [isSpaceHeld, setIsSpaceHeld] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number; scrollLeft: number; scrollTop: number } | null>(null);

  // Use provided schema or default to fish morphometrics
  const activeSchema = useMemo(() => {
    return schema || DEFAULT_SCHEMAS.find(s => s.id === "fish-morphometrics") || DEFAULT_SCHEMAS[0];
  }, [schema]);

  // Get landmarks from the first box for the guide
  const selectedBoxLandmarks = useMemo(() => {
    if (boxes.length === 0) return [];
    const box = boxes[0];
    return box?.landmarks || [];
  }, [boxes]);

  const MAX_WIDTH = window.innerWidth * 0.9;
  const MAX_HEIGHT = window.innerHeight * 0.9;

  const calculateBaseScale = useCallback(() => {
    if (!imageDimensions) return 1;
    const widthScale = MAX_WIDTH / imageDimensions.width;
    const heightScale = MAX_HEIGHT / imageDimensions.height;
    return Math.min(widthScale, heightScale);
  }, [imageDimensions, MAX_WIDTH, MAX_HEIGHT]);

  const baseScale = calculateBaseScale();
  const scale = baseScale * userZoom;

  // Reset user zoom when image changes or dialog opens
  useEffect(() => {
    if (open) {
      setUserZoom(1);
      // Reset scroll position
      if (canvasContainerRef.current) {
        canvasContainerRef.current.scrollLeft = 0;
        canvasContainerRef.current.scrollTop = 0;
      }
    }
  }, [imageURL, open]);

  // Handle wheel zoom centered on cursor position (adjusts scroll for scrollable container)
  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();

    const container = canvasContainerRef.current;
    if (!container) return;

    // Get cursor position relative to container
    const containerRect = container.getBoundingClientRect();
    const cursorXInContainer = e.clientX - containerRect.left;
    const cursorYInContainer = e.clientY - containerRect.top;

    // Position in the content (accounting for scroll)
    const cursorXInContent = cursorXInContainer + container.scrollLeft;
    const cursorYInContent = cursorYInContainer + container.scrollTop;

    // Current and new scale
    const oldScale = scale;
    const zoomIntensity = e.ctrlKey ? 0.02 : 0.1;
    const direction = e.deltaY > 0 ? -1 : 1;

    const newUserZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM,
      userZoom + direction * zoomIntensity * userZoom
    ));
    const newScale = baseScale * newUserZoom;
    const scaleRatio = newScale / oldScale;

    // Calculate new scroll position to keep cursor point stationary
    const newScrollLeft = cursorXInContent * scaleRatio - cursorXInContainer;
    const newScrollTop = cursorYInContent * scaleRatio - cursorYInContainer;

    setUserZoom(newUserZoom);

    // Apply new scroll position after state update
    requestAnimationFrame(() => {
      container.scrollLeft = Math.max(0, newScrollLeft);
      container.scrollTop = Math.max(0, newScrollTop);
    });
  }, [userZoom, baseScale, scale]);
  const baseRadius = 1;

  const getScaledRadius = useCallback(() => {
    if (!imageDimensions) return baseRadius;
    const imageDiagonal = Math.sqrt(
      Math.pow(imageDimensions.width, 2) + Math.pow(imageDimensions.height, 2)
    );
    return Math.max(baseRadius, imageDiagonal * 0.002);
  }, [imageDimensions]);

  const getTextConfig = useCallback(() => {
    if (!imageDimensions) return { fontSize: 7 };
    const imageDiagonal = Math.sqrt(
      Math.pow(imageDimensions.width, 2) + Math.pow(imageDimensions.height, 2)
    );
    const fontSize = Math.max(6, imageDiagonal * 0.007);
    return { fontSize };
  }, [imageDimensions]);

  // When boxes change and we have a pending landmark to add
  useEffect(() => {
    if (pendingBoxRef.current && boxes.length > 0) {
      const pos = pendingBoxRef.current;
      pendingBoxRef.current = null;
      const targetBox = boxes[0];
      addLandmark(targetBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
    }
  }, [boxes, addLandmark]);

  const getPointerPosition = useCallback((e: KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    const pointerPosition = stage?.getPointerPosition();
    if (!pointerPosition || !stage) return null;

    const x = pointerPosition.x / scale;
    const y = pointerPosition.y / scale;
    return { x, y };
  }, [scale]);

  const isPointInBounds = useCallback(
    (x: number, y: number) => {
      if (!imageDimensions) return false;
      return x >= 0 && y >= 0 && x <= imageDimensions.width && y <= imageDimensions.height;
    },
    [imageDimensions]
  );

  // Click to add landmark - auto-creates box if needed
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      // Don't add landmarks if we're in drag mode or just finished dragging
      if (!image || !imageDimensions || mode || isSpaceHeld || isDragging) return;
      // Ignore middle mouse button clicks
      if (e.evt.button === 1) return;

      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;

      // Auto-create a default box if none exists
      if (boxes.length === 0) {
        // Store the position to add landmark after box is created
        pendingBoxRef.current = pos;
        addBox({
          left: 0,
          top: 0,
          width: imageDimensions.width,
          height: imageDimensions.height,
        });
        return;
      }

      // Use the first box
      const targetBox = boxes[0];

      // Auto-select the box if not selected
      if (selectedBoxId !== targetBox.id) {
        selectBox(targetBox.id);
      }

      addLandmark(targetBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
    },
    [image, imageDimensions, mode, isSpaceHeld, isDragging, getPointerPosition, isPointInBounds, boxes, selectedBoxId, addBox, selectBox, addLandmark]
  );

  // Note: Keyboard undo/redo is handled by ImageLabeler (parent component)
  // to avoid duplicate event handlers

  // Space key for drag mode
  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        setIsSpaceHeld(true);
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        setIsSpaceHeld(false);
        setIsDragging(false);
        dragStartRef.current = null;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [open]);

  // Handle drag/pan for scrollable container
  const handleMouseDown = useCallback((e: KonvaEventObject<MouseEvent>) => {
    const container = canvasContainerRef.current;
    if (!container) return;

    // Middle mouse button (button 1) or Space + left click enables drag
    if (e.evt.button === 1 || (isSpaceHeld && e.evt.button === 0)) {
      e.evt.preventDefault();
      setIsDragging(true);
      dragStartRef.current = {
        x: e.evt.clientX,
        y: e.evt.clientY,
        scrollLeft: container.scrollLeft,
        scrollTop: container.scrollTop,
      };
    }
  }, [isSpaceHeld]);

  const handleMouseMove = useCallback((e: KonvaEventObject<MouseEvent>) => {
    if (!isDragging || !dragStartRef.current) return;

    const container = canvasContainerRef.current;
    if (!container) return;

    const dx = e.evt.clientX - dragStartRef.current.x;
    const dy = e.evt.clientY - dragStartRef.current.y;

    container.scrollLeft = dragStartRef.current.scrollLeft - dx;
    container.scrollTop = dragStartRef.current.scrollTop - dy;
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      dragStartRef.current = null;
    }
  }, [isDragging]);

  // Also listen for mouseup on window in case mouse leaves stage
  useEffect(() => {
    if (!open) return;

    const handleWindowMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
        dragStartRef.current = null;
      }
    };
    window.addEventListener("mouseup", handleWindowMouseUp);
    return () => window.removeEventListener("mouseup", handleWindowMouseUp);
  }, [isDragging, open]);

  // Attach wheel event listener to canvas container for zoom
  useEffect(() => {
    const container = canvasContainerRef.current;
    if (!container || !open) return;

    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => container.removeEventListener("wheel", handleWheel);
  }, [handleWheel, open]);

  if (imageError) {
    return <div className="text-destructive">Error loading image.</div>;
  }

  // Track global landmark index for numbering
  let globalLandmarkIndex = 0;

  // Get the first box id for skip functionality
  const firstBoxId = boxes.length > 0 ? boxes[0].id : null;

  return (
    <TooltipProvider>
      <Dialog open={open} onOpenChange={(value) => !value && onClose()}>
        <DialogContent
          className="max-w-[95vw] max-h-[90vh] p-4 overflow-auto"
          hideCloseButton
          style={{
            width: imageDimensions ? Math.min(imageDimensions.width * scale + 320, window.innerWidth * 0.95) : "auto",
          }}
        >
          <motion.div
            variants={modalContent}
            initial="hidden"
            animate="visible"
            exit="exit"
          >
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant={showGuide ? "default" : "outline"}
                      size="sm"
                      onClick={() => setShowGuide(!showGuide)}
                      className="gap-1"
                    >
                      <Info className="h-4 w-4" />
                      {showGuide ? "Hide Guide" : "Show Guide"}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Toggle landmark placement guide</TooltipContent>
                </Tooltip>
              </div>
              <DialogClose asChild>
                <Button variant="ghost" size="icon" aria-label="Close Magnified View">
                  <X className="h-4 w-4" />
                </Button>
              </DialogClose>
            </div>

          {image && imageDimensions && (
            <div className="flex gap-4">
              {/* Canvas */}
              <div ref={canvasContainerRef} className="flex-1 min-w-0 overflow-auto">
                <Stage
                  width={imageDimensions.width * scale}
                  height={imageDimensions.height * scale}
                  onClick={handleCanvasClick}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  ref={stageRef}
                  style={{
                    border: "1px solid hsl(var(--border))",
                    backgroundColor: "hsl(var(--muted))",
                    cursor: isDragging ? "grabbing" : isSpaceHeld ? "grab" : mode ? "default" : "crosshair",
                    display: "block",
                    margin: "0 auto",
                    borderRadius: "8px",
                  }}
                >
              <Layer scaleX={scale} scaleY={scale}>
                <KonvaImage
                  image={image}
                  width={imageDimensions.width}
                  height={imageDimensions.height}
                />

                {/* Render landmarks (box is hidden since it covers full image) */}
                {boxes.map((box) => (
                  <React.Fragment key={box.id}>
                    {box.landmarks.map((point) => {
                      globalLandmarkIndex++;
                      // Skip rendering for skipped landmarks
                      if (point.isSkipped) return null;
                      return (
                        <React.Fragment key={point.id}>
                          <Circle
                            x={point.x}
                            y={point.y}
                            radius={getScaledRadius()}
                            fill={color}
                            opacity={opacity / 100}
                          />
                          <Text
                            x={point.x + 2.5}
                            y={point.y - 8}
                            text={globalLandmarkIndex.toString()}
                            fontSize={getTextConfig().fontSize}
                            fill={color}
                            opacity={opacity / 100}
                          />
                        </React.Fragment>
                      );
                    })}
                  </React.Fragment>
                ))}
              </Layer>
                </Stage>
              </div>

              {/* Landmark Placement Guide */}
              {showGuide && activeSchema && (
                <div className="w-64 shrink-0">
                  <LandmarkPlacementGuide
                    schema={activeSchema}
                    placedLandmarks={selectedBoxLandmarks}
                    onSkip={firstBoxId ? () => skipLandmark(firstBoxId) : undefined}
                  />
                </div>
              )}
            </div>
          )}
        </motion.div>
      </DialogContent>
    </Dialog>
    </TooltipProvider>
  );
};

export default MagnifiedImageLabeler;
