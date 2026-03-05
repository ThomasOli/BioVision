import React, { useRef, useCallback, useEffect, useContext, useState, useMemo } from "react";
import { motion } from "framer-motion";
import { X, Info } from "lucide-react";
import { Stage, Layer, Image as KonvaImage, Circle, Text, Rect, Line, Transformer, Arrow } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";
import { LandmarkPlacementGuide } from "./LandmarkPlacementGuide";
import { DetectionMode } from "./DetectionModeSelector";
import { Button } from "@/Components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogClose,
  DialogTitle,
  DialogDescription,
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

// ── OBB / Transformer helpers ─────────────────────────────────────────────
function getBoxKonvaParams(box: BoundingBox): {
  cx: number; cy: number; w: number; h: number; angleDeg: number;
} {
  if (box.obbCorners && box.obbCorners.length === 4) {
    const c = box.obbCorners;
    const cx = (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4;
    const cy = (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4;
    const w  = Math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]);
    const h  = Math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1]);
    const angleDeg = box.angle ??
      (Math.atan2(c[1][1] - c[0][1], c[1][0] - c[0][0]) * 180 / Math.PI);
    return { cx, cy, w: Math.max(1, w), h: Math.max(1, h), angleDeg };
  }
  return {
    cx: box.left + box.width / 2,
    cy: box.top  + box.height / 2,
    w: box.width,
    h: box.height,
    angleDeg: 0,
  };
}
function buildObbCorners(
  cx: number, cy: number, w: number, h: number, angleDeg: number,
): [number, number][] {
  const r = angleDeg * (Math.PI / 180);
  const cos = Math.cos(r), sin = Math.sin(r);
  const hw = w / 2, hh = h / 2;
  return [
    [cx + cos * (-hw) - sin * (-hh), cy + sin * (-hw) + cos * (-hh)],
    [cx + cos *   hw  - sin * (-hh), cy + sin *   hw  + cos * (-hh)],
    [cx + cos *   hw  - sin *   hh,  cy + sin *   hw  + cos *   hh ],
    [cx + cos * (-hw) - sin *   hh,  cy + sin * (-hw) + cos *   hh ],
  ] as [number, number][];
}
function cornersToAabb(
  corners: [number, number][], imgW: number, imgH: number,
): { left: number; top: number; width: number; height: number } {
  const xs = corners.map(c => c[0]);
  const ys = corners.map(c => c[1]);
  const left  = Math.round(Math.max(0,    Math.min(...xs)));
  const top   = Math.round(Math.max(0,    Math.min(...ys)));
  const right = Math.round(Math.min(imgW, Math.max(...xs)));
  const bot   = Math.round(Math.min(imgH, Math.max(...ys)));
  return { left, top, width: Math.max(12, right - left), height: Math.max(12, bot - top) };
}
// ─────────────────────────────────────────────────────────────────────────────

interface MagnifiedImageLabelerProps {
  imageURL: string;
  onBoxesChange: (boxes: BoundingBox[]) => void;
  color: string;
  opacity: number;
  open: boolean;
  onClose: () => void;
  mode: boolean; // View-only mode
  schema?: LandmarkSchema;
  detectionMode?: DetectionMode;
  autoCorrectionMode?: boolean;
  imagePath?: string;
  samEnabled?: boolean;
  hideSegmentOutlines?: boolean;
  lockBoxes?: boolean;
  orientationMode?: "directional" | "bilateral" | "axial" | "invariant";
}

const MagnifiedImageLabeler: React.FC<MagnifiedImageLabelerProps> = ({
  imageURL,
  color,
  opacity,
  open,
  onClose,
  mode,
  schema,
  detectionMode = "manual",
  autoCorrectionMode = false,
  imagePath,
  samEnabled = false,
  hideSegmentOutlines = false,
  lockBoxes = false,
  orientationMode,
}) => {
  const {
    addLandmark,
    boxes,
    selectedBoxId,
    addBox,
    selectBox,
    skipLandmark,
    updateBox,
  } = useContext(UndoRedoClearContext);

  const [resegmentingBoxId, setResegmentingBoxId] = useState<number | null>(null);

  const triggerResegment = useCallback(
    async (boxId: number, left: number, top: number, width: number, height: number) => {
      if (!samEnabled || !imagePath) return;
      setResegmentingBoxId(boxId);
      try {
        const result = await window.api.resegmentBox(imagePath, [left, top, left + width, top + height]);
        if (result.ok && result.maskOutline && result.maskOutline.length > 0) {
          updateBox(boxId, { maskOutline: result.maskOutline });
        }
      } catch (err) {
        console.error("SAM2 re-segmentation failed:", err);
      } finally {
        setResegmentingBoxId(null);
      }
    },
    [samEnabled, imagePath, updateBox]
  );

  const [image, imageDimensions, imageError] = useImageLoader(imageURL);
  const stageRef = useRef<Konva.Stage>(null);
  const selectedRectRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  // Track if we just created a box to avoid double-adding landmarks
  const pendingBoxRef = useRef<{ x: number; y: number } | null>(null);
  // Auto-select the most-recently drawn box so the Transformer appears immediately
  const pendingSelectRef = useRef<boolean>(false);
  useEffect(() => {
    if (pendingSelectRef.current && boxes.length > 0) {
      pendingSelectRef.current = false;
      selectBox(boxes[boxes.length - 1].id);
    }
  }, [boxes, selectBox]);

  // Drag-to-draw bounding box state
  const [isDrawingBox, setIsDrawingBox] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);
  const [isRedrawingSelected, setIsRedrawingSelected] = useState(false);
  const [drawDefaultOrientation, setDrawDefaultOrientation] = useState<"left" | "right">(() =>
    ((typeof window !== "undefined" && window.localStorage.getItem("bv_draw_default_orientation")) as "left" | "right") ?? "left"
  );

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

  // Keep one shared box set across manual/auto modes so switching modes
  // never hides or drops accepted boxes.
  const visibleBoxes = useMemo<BoundingBox[]>(() => {
    return boxes;
  }, [boxes]);

  // Get landmarks from the selected box for the guide
  const selectedBoxLandmarks = useMemo(() => {
    if (visibleBoxes.length === 0) return [];
    if (detectionMode === "auto" && selectedBoxId !== null) {
      const box = visibleBoxes.find(b => b.id === selectedBoxId);
      return box?.landmarks || [];
    }
    const box = visibleBoxes[0];
    return box?.landmarks || [];
  }, [visibleBoxes, detectionMode, selectedBoxId]);

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

  // Box stroke width based on image size
  const boxStrokeWidth = useMemo(() => {
    if (!imageDimensions) return 2;
    return Math.max(2, imageDimensions.width * 0.002);
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

  // Check if point is inside a box
  const findBoxAtPoint = useCallback(
    (x: number, y: number): BoundingBox | null => {
      // Give priority to the currently selected box so that clicking near
      // the edge of a selected box doesn't accidentally activate an adjacent one
      if (selectedBoxId !== null) {
        const selectedBox = visibleBoxes.find(b => b.id === selectedBoxId);
        if (
          selectedBox &&
          x >= selectedBox.left &&
          x <= selectedBox.left + selectedBox.width &&
          y >= selectedBox.top &&
          y <= selectedBox.top + selectedBox.height
        ) {
          return selectedBox;
        }
      }
      for (let i = visibleBoxes.length - 1; i >= 0; i--) {
        const box = visibleBoxes[i];
        if (
          x >= box.left &&
          x <= box.left + box.width &&
          y >= box.top &&
          y <= box.top + box.height
        ) {
          return box;
        }
      }
      return null;
    },
    [visibleBoxes, selectedBoxId]
  );

  // Click to add landmark - supports both manual and auto mode
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      // Don't add landmarks if we're in drag mode or just finished dragging
      if (!image || !imageDimensions || mode || isSpaceHeld || isDragging) return;
      // Ignore middle mouse button clicks
      if (e.evt.button === 1) return;

      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;

      // Auto mode: click to select box, click inside selected box to add landmark
      if (detectionMode === "auto") {
        const clickedBox = findBoxAtPoint(pos.x, pos.y);

        if (clickedBox) {
          if (selectedBoxId !== clickedBox.id) {
            selectBox(clickedBox.id);
            return;
          }
          if (!autoCorrectionMode) {
            addLandmark(clickedBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
          }
          return;
        }
        return;
      }

      // Manual mode: clicking inside an existing box adds a landmark to it
      if (detectionMode === "manual") {
        const clickedBox = findBoxAtPoint(pos.x, pos.y);
        if (clickedBox) {
          if (selectedBoxId !== clickedBox.id) {
            selectBox(clickedBox.id);
            return;
          }
          addLandmark(clickedBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
        }
        return;
      }
    },
    [image, imageDimensions, mode, isSpaceHeld, isDragging, getPointerPosition, isPointInBounds, selectedBoxId, selectBox, addLandmark, detectionMode, findBoxAtPoint, autoCorrectionMode]
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

  // Handle drag/pan for scrollable container + box drawing
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
      return;
    }

    // Manual mode: start drawing a box if not inside an existing box
    if (detectionMode === "manual" && !mode && !lockBoxes && !isSpaceHeld && e.evt.button === 0) {
      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;
      const clickedBox = findBoxAtPoint(pos.x, pos.y);
      if (!clickedBox) {
        setIsDrawingBox(true);
        setIsRedrawingSelected(false);
        setDrawStart(pos);
        setDrawCurrent(pos);
      }
      return;
    }

    // Auto mode correction: redraw selected box by dragging in empty area
    if (detectionMode === "auto" && autoCorrectionMode && !mode && !isSpaceHeld && e.evt.button === 0) {
      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;
      const clickedBox = findBoxAtPoint(pos.x, pos.y);
      if (!clickedBox && selectedBoxId !== null) {
        setIsDrawingBox(true);
        setIsRedrawingSelected(true);
        setDrawStart(pos);
        setDrawCurrent(pos);
      }
    }
  }, [isSpaceHeld, detectionMode, autoCorrectionMode, mode, lockBoxes, selectedBoxId, getPointerPosition, isPointInBounds, findBoxAtPoint]);

  const handleMouseMove = useCallback((e: KonvaEventObject<MouseEvent>) => {
    // Pan mode
    if (isDragging && dragStartRef.current) {
      const container = canvasContainerRef.current;
      if (!container) return;
      const dx = e.evt.clientX - dragStartRef.current.x;
      const dy = e.evt.clientY - dragStartRef.current.y;
      container.scrollLeft = dragStartRef.current.scrollLeft - dx;
      container.scrollTop = dragStartRef.current.scrollTop - dy;
      return;
    }

    // Box drawing mode
    if (isDrawingBox && drawStart) {
      const pos = getPointerPosition(e);
      if (!pos) return;
      const x = imageDimensions ? Math.max(0, Math.min(pos.x, imageDimensions.width)) : pos.x;
      const y = imageDimensions ? Math.max(0, Math.min(pos.y, imageDimensions.height)) : pos.y;
      setDrawCurrent({ x, y });
    }
  }, [isDragging, isDrawingBox, drawStart, getPointerPosition, imageDimensions]);

  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      dragStartRef.current = null;
      return;
    }

    if (isDrawingBox && drawStart && drawCurrent) {
      const left = Math.min(drawStart.x, drawCurrent.x);
      const top = Math.min(drawStart.y, drawCurrent.y);
      const width = Math.abs(drawCurrent.x - drawStart.x);
      const height = Math.abs(drawCurrent.y - drawStart.y);

      const minSize = imageDimensions ? Math.max(20, imageDimensions.width * 0.02) : 20;
      if (width >= minSize && height >= minSize) {
        if (detectionMode === "manual") {
          const bLeft = Math.round(left), bTop = Math.round(top);
          const bRight = bLeft + Math.round(width), bBottom = bTop + Math.round(height);
          addBox({
            left: bLeft,
            top: bTop,
            width: Math.round(width),
            height: Math.round(height),
            obbCorners: [[bLeft, bTop], [bRight, bTop], [bRight, bBottom], [bLeft, bBottom]],
            angle: 0,
            class_id: drawDefaultOrientation === "left" ? 0 : 1,
            source: "manual",
          });
          pendingSelectRef.current = true;
        } else if (
          detectionMode === "auto" &&
          autoCorrectionMode &&
          isRedrawingSelected &&
          selectedBoxId !== null
        ) {
          const rLeft = Math.round(left);
          const rTop = Math.round(top);
          const rWidth = Math.round(width);
          const rHeight = Math.round(height);
          updateBox(selectedBoxId, {
            left: rLeft,
            top: rTop,
            width: rWidth,
            height: rHeight,
            source: "corrected",
            confidence: undefined,
            maskOutline: undefined,
            detectionMethod: "human_corrected",
          });
          triggerResegment(selectedBoxId, rLeft, rTop, rWidth, rHeight);
        }
      }
    }

    setIsDrawingBox(false);
    setIsRedrawingSelected(false);
    setDrawStart(null);
    setDrawCurrent(null);
  }, [isDragging, isDrawingBox, drawStart, drawCurrent, imageDimensions, addBox, detectionMode, autoCorrectionMode, isRedrawingSelected, selectedBoxId, updateBox, triggerResegment, drawDefaultOrientation]);

  // Draw preview rect
  const drawPreview = useMemo(() => {
    if (!isDrawingBox || !drawStart || !drawCurrent) return null;
    return {
      x: Math.min(drawStart.x, drawCurrent.x),
      y: Math.min(drawStart.y, drawCurrent.y),
      width: Math.abs(drawCurrent.x - drawStart.x),
      height: Math.abs(drawCurrent.y - drawStart.y),
    };
  }, [isDrawingBox, drawStart, drawCurrent]);

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

  useEffect(() => {
    const tr = transformerRef.current;
    const node = selectedRectRef.current;
    if (!tr) return;

    const canTransform =
      !mode && !lockBoxes &&
      (detectionMode === "manual" || (detectionMode === "auto" && autoCorrectionMode));
    if (canTransform && node) {
      tr.nodes([node]);
      tr.getLayer()?.batchDraw();
      return;
    }

    tr.nodes([]);
    tr.getLayer()?.batchDraw();
  }, [mode, lockBoxes, detectionMode, autoCorrectionMode, selectedBoxId, visibleBoxes]);

  if (imageError) {
    return <div className="text-destructive">Error loading image.</div>;
  }

  // Gate orientation arrows on vector schemas (directional + bilateral)
  const showOrientationArrow = !orientationMode || orientationMode === "directional" || orientationMode === "bilateral";
  const isVectorSchema = showOrientationArrow; // same condition

  // Colors for boxes
  const getBoxColor = (isSelected: boolean) => {
    if (isSelected) return "#3b82f6"; // blue-500
    return "#6b7280"; // gray-500
  };

  // Get the first box id for skip functionality
  const skipBoxId = detectionMode === "auto"
    ? (selectedBoxId ?? (visibleBoxes.length > 0 ? visibleBoxes[0].id : null))
    : (visibleBoxes.length > 0 ? visibleBoxes[0].id : null);

  return (
    <TooltipProvider>
      <Dialog open={open} onOpenChange={(value) => !value && onClose()}>
        <DialogContent
          className="max-w-[95vw] max-h-[90vh] p-4 flex flex-col overflow-hidden"
          hideCloseButton
          style={{
            width: imageDimensions ? Math.min(imageDimensions.width * scale + 320, window.innerWidth * 0.95) : "auto",
          }}
        >
          <DialogTitle className="sr-only">Magnified Image Labeler</DialogTitle>
          <DialogDescription className="sr-only">
            Zoomed view for placing landmark annotations on the selected image.
          </DialogDescription>
          <motion.div
            variants={modalContent}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="flex flex-col flex-1 min-h-0"
          >
            <div className="flex justify-between items-center mb-2 shrink-0">
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
                {!mode && !lockBoxes && detectionMode === "manual" && isVectorSchema && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          const next = drawDefaultOrientation === "left" ? "right" : "left";
                          setDrawDefaultOrientation(next);
                          window.localStorage.setItem("bv_draw_default_orientation", next);
                          if (selectedBoxId !== null) {
                            const sel = boxes.find(b => b.id === selectedBoxId);
                            if (sel && sel.class_id !== undefined) {
                              updateBox(selectedBoxId, { class_id: sel.class_id === 0 ? 1 : 0 });
                            }
                          }
                        }}
                      >
                        {drawDefaultOrientation === "left" ? "\u2190 Head" : "Head \u2192"}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Toggle default head direction for new boxes</TooltipContent>
                  </Tooltip>
                )}
              </div>
              <DialogClose asChild>
                <Button variant="ghost" size="icon" aria-label="Close Magnified View">
                  <X className="h-4 w-4" />
                </Button>
              </DialogClose>
            </div>

          {image && imageDimensions && (
            <div className="flex gap-4 flex-1 min-h-0">
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

                {/* Draw preview rectangle during drag */}
                {drawPreview && (() => {
                  const { x: px, y: py, width: pw, height: ph } = drawPreview;
                  const arrowIsLeft = drawDefaultOrientation === "left";
                  const maxDim = Math.max(pw, ph);
                  const arrowLen = Math.min(Math.max(maxDim * 0.25, 14), 32);
                  const midY = py + ph / 2;
                  const tipX  = arrowIsLeft ? px : px + pw;
                  const tailX = arrowIsLeft ? px + arrowLen : px + pw - arrowLen;
                  return (
                    <>
                      <Rect
                        x={px} y={py}
                        width={pw} height={ph}
                        stroke="#3b82f6"
                        strokeWidth={boxStrokeWidth}
                        dash={[8, 4]}
                        fill="rgba(59, 130, 246, 0.08)"
                      />
                      {maxDim >= 24 && (
                        <Arrow
                          points={[tailX, midY, tipX, midY]}
                          pointerLength={Math.min(Math.max(maxDim * 0.10, 6), 10)}
                          pointerWidth={Math.min(Math.max(maxDim * 0.07, 5), 8)}
                          fill="#3b82f6"
                          stroke="#3b82f6"
                          strokeWidth={Math.max(1.5, boxStrokeWidth * 0.85)}
                          opacity={0.9}
                          listening={false}
                        />
                      )}
                    </>
                  );
                })()}

                {/* Render bounding boxes */}
                {visibleBoxes.map((box, index) => {
                  const isSelected = selectedBoxId === box.id;
                  const boxColor = getBoxColor(isSelected);
                  const isEditableSelected =
                    isSelected && !mode && !lockBoxes &&
                    (detectionMode === "manual" || (detectionMode === "auto" && autoCorrectionMode));
                  const kp = getBoxKonvaParams(box);
                  const hasObb = !!(box.obbCorners && box.obbCorners.length === 4);

                  return (
                    <React.Fragment key={`box-${box.id}`}>
                      {/* SAM2 mask polygon overlay */}
                      {!hideSegmentOutlines && box.maskOutline && box.maskOutline.length > 0 && (
                        <Line
                          points={box.maskOutline.flat()}
                          closed={true}
                          fill={isSelected ? "rgba(59, 130, 246, 0.15)" : "rgba(100, 100, 100, 0.1)"}
                          stroke={isSelected ? "#3b82f6" : "#6b7280"}
                          strokeWidth={boxStrokeWidth * 0.5}
                        />
                      )}
                      {/* OBB polygon outline */}
                      {hasObb && box.obbCorners!.length === 4 && (
                        <Line
                          points={box.obbCorners!.flat()}
                          closed={true}
                          stroke={boxColor}
                          strokeWidth={boxStrokeWidth}
                          dash={isSelected ? undefined : [8, 4]}
                          fill={isSelected ? "rgba(59, 130, 246, 0.08)" : "transparent"}
                          listening={!isEditableSelected}
                        />
                      )}
                      {/* Interactive rect — center-anchored for correct rotation pivot */}
                      <Rect
                        ref={isEditableSelected ? selectedRectRef : undefined}
                        x={kp.cx}
                        y={kp.cy}
                        width={kp.w}
                        height={kp.h}
                        offsetX={kp.w / 2}
                        offsetY={kp.h / 2}
                        rotation={kp.angleDeg}
                        stroke={hasObb ? "transparent" : boxColor}
                        strokeWidth={boxStrokeWidth}
                        dash={isSelected ? undefined : [10, 5]}
                        fill={isEditableSelected && hasObb ? "rgba(0,0,0,0.001)" : (isSelected && !hasObb ? "rgba(59, 130, 246, 0.1)" : "transparent")}
                        draggable={isEditableSelected}
                        onDragEnd={(e) => {
                          if (!isEditableSelected || !imageDimensions) return;
                          const node = e.target;
                          const newCx = node.x();
                          const newCy = node.y();
                          if (hasObb && box.obbCorners) {
                            const dx = newCx - kp.cx;
                            const dy = newCy - kp.cy;
                            const newCorners = box.obbCorners.map(
                              ([px, py]) => [px + dx, py + dy] as [number, number],
                            );
                            const aabb = cornersToAabb(newCorners, imageDimensions.width, imageDimensions.height);
                            updateBox(box.id, {
                              ...aabb,
                              obbCorners: newCorners,
                              angle: box.angle,
                              source: "corrected",
                              confidence: undefined,
                              maskOutline: undefined,
                              detectionMethod: "human_corrected",
                            });
                            triggerResegment(box.id, aabb.left, aabb.top, aabb.width, aabb.height);
                          } else {
                            const nextLeft = Math.round(Math.max(0, Math.min(newCx - kp.w / 2, imageDimensions.width - kp.w)));
                            const nextTop  = Math.round(Math.max(0, Math.min(newCy - kp.h / 2, imageDimensions.height - kp.h)));
                            updateBox(box.id, {
                              left: nextLeft,
                              top: nextTop,
                              source: "corrected",
                              confidence: undefined,
                              maskOutline: undefined,
                              detectionMethod: "human_corrected",
                            });
                            triggerResegment(box.id, nextLeft, nextTop, box.width, box.height);
                          }
                        }}
                        onTransformEnd={() => {
                          if (!isEditableSelected || !imageDimensions || !selectedRectRef.current) return;
                          const node = selectedRectRef.current;
                          const newCx    = node.x();
                          const newCy    = node.y();
                          const newW     = Math.max(12, kp.w * node.scaleX());
                          const newH     = Math.max(12, kp.h * node.scaleY());
                          const newAngle = node.rotation();
                          node.scaleX(1);
                          node.scaleY(1);
                          const newCorners = buildObbCorners(newCx, newCy, newW, newH, newAngle);
                          const aabb = cornersToAabb(newCorners, imageDimensions.width, imageDimensions.height);
                          const isRotated = Math.abs(newAngle) > 0.5;
                          updateBox(box.id, {
                            ...aabb,
                            ...(isRotated || hasObb
                              ? { angle: newAngle, obbCorners: newCorners }
                              : { angle: undefined, obbCorners: undefined }),
                            source: "corrected",
                            confidence: undefined,
                            maskOutline: undefined,
                            detectionMethod: "human_corrected",
                          });
                          triggerResegment(box.id, aabb.left, aabb.top, aabb.width, aabb.height);
                        }}
                      />
                      {/* Box number label */}
                      <Text
                        x={box.left + 5}
                        y={box.top + 5}
                        text={`#${index + 1}`}
                        fontSize={getTextConfig().fontSize * 1.2}
                        fill={boxColor}
                        fontStyle="bold"
                      />
                      {/* Confidence badge if available */}
                      {box.confidence !== undefined && (
                        <Text
                          x={box.left + 5}
                          y={box.top + 5 + getTextConfig().fontSize * 1.5}
                          text={`${(box.confidence * 100).toFixed(0)}%`}
                          fontSize={getTextConfig().fontSize * 0.9}
                          fill={boxColor}
                        />
                      )}
                      {/* SAM2 re-segmenting indicator */}
                      {resegmentingBoxId === box.id && (
                        <Text
                          x={box.left + 5}
                          y={box.top + box.height - getTextConfig().fontSize * 1.5 - 5}
                          text="⟳ Segmenting…"
                          fontSize={getTextConfig().fontSize * 0.9}
                          fill="#60a5fa"
                          fontStyle="bold"
                        />
                      )}
                      {/* Orientation arrow + tilt angle */}
                      {(() => {
                        const corners = box.obbCorners && box.obbCorners.length === 4
                          ? box.obbCorners as [number,number][]
                          : null;
                        if (!corners) return null;
                        if (orientationMode === "invariant") return null;
                        const [cp0, cp1, cp2, cp3] = corners;
                        // Tilt angle (schema-independent, only needs corners)
                        const adx1 = cp1[0]-cp0[0], ady1 = cp1[1]-cp0[1];
                        const adx3 = cp3[0]-cp0[0], ady3 = cp3[1]-cp0[1];
                        const [aldx, aldy] = Math.hypot(adx1,ady1) >= Math.hypot(adx3,ady3) ? [adx1,ady1] : [adx3,ady3];
                        let adeg = Math.atan2(aldy, aldx) * 180 / Math.PI;
                        adeg = ((adeg % 180) + 180) % 180;
                        if (adeg > 90) adeg = 180 - adeg;
                        // Arrow (vector schemas only)
                        const hasOrientation = box.class_id !== undefined;
                        let arrowEl = null;
                        if (showOrientationArrow && hasOrientation) {
                          const isLeft = box.class_id === 0;
                          const isRight = box.class_id === 1;
                          const cLen01 = Math.hypot(cp1[0]-cp0[0], cp1[1]-cp0[1]);
                          const cLen12 = Math.hypot(cp2[0]-cp1[0], cp2[1]-cp1[1]);
                          let cMidA: [number,number], cMidB: [number,number];
                          if (cLen01 >= cLen12) {
                            cMidA = [(cp0[0]+cp3[0])/2, (cp0[1]+cp3[1])/2];
                            cMidB = [(cp1[0]+cp2[0])/2, (cp1[1]+cp2[1])/2];
                          } else {
                            cMidA = [(cp0[0]+cp1[0])/2, (cp0[1]+cp1[1])/2];
                            cMidB = [(cp2[0]+cp3[0])/2, (cp2[1]+cp3[1])/2];
                          }
                          const [cLeftEnd, cRightEnd] = cMidA[0] <= cMidB[0] ? [cMidA, cMidB] : [cMidB, cMidA];
                          const cHead = (isLeft || !isRight) ? cLeftEnd : cRightEnd;
                          const cTail = (isLeft || !isRight) ? cRightEnd : cLeftEnd;
                          const cAxisLen = Math.hypot(cHead[0]-cTail[0], cHead[1]-cTail[1]) || 1;
                          if (cAxisLen >= 24) {
                            const cNx = (cHead[0]-cTail[0])/cAxisLen, cNy = (cHead[1]-cTail[1])/cAxisLen;
                            const arrowLen  = Math.min(Math.max(cAxisLen * 0.25, 14), 32);
                            const cHSizeLen = Math.min(Math.max(cAxisLen * 0.10, 6), 10);
                            const cHSizeW   = Math.min(Math.max(cAxisLen * 0.07, 5), 8);
                            arrowEl = (
                              <Arrow
                                points={[cHead[0] - cNx*arrowLen, cHead[1] - cNy*arrowLen, cHead[0], cHead[1]]}
                                pointerLength={cHSizeLen}
                                pointerWidth={cHSizeW}
                                fill={boxColor}
                                stroke={boxColor}
                                strokeWidth={Math.max(1.5, boxStrokeWidth * 0.85)}
                                opacity={isSelected ? 0.95 : 0.85}
                                listening={false}
                              />
                            );
                          }
                        }
                        return (
                          <>
                            {arrowEl}
                            <Text
                              x={box.left + 5}
                              y={box.top + 5 + getTextConfig().fontSize * 1.5 * (box.confidence !== undefined ? 2 : 1)}
                              text={`${Math.round(adeg)}\u00B0`}
                              fontSize={getTextConfig().fontSize * 0.9}
                              fill={boxColor}
                            />
                          </>
                        );
                      })()}
                    </React.Fragment>
                  );
                })}

                {!mode && !lockBoxes &&
                  (detectionMode === "manual" || (detectionMode === "auto" && autoCorrectionMode)) && (
                  <Transformer
                    ref={transformerRef}
                    rotateEnabled={true}
                    enabledAnchors={[
                      "top-left",
                      "top-center",
                      "top-right",
                      "middle-right",
                      "bottom-right",
                      "bottom-center",
                      "bottom-left",
                      "middle-left",
                    ]}
                    boundBoxFunc={(oldBox, newBox) => {
                      if (newBox.width < 12 || newBox.height < 12) return oldBox;
                      return newBox;
                    }}
                  />
                )}

                {/* Render landmarks */}
                {visibleBoxes.map((box) => {
                  const isBoxSelected = selectedBoxId === box.id;

                  return (
                    <React.Fragment key={`landmarks-${box.id}`}>
                      {box.landmarks.map((point, lmIndex) => {
                        if (point.isSkipped) return null;

                        // Dim landmarks of non-selected boxes
                        const landmarkOpacity = !isBoxSelected && visibleBoxes.length > 1
                          ? (opacity / 100) * 0.4
                          : opacity / 100;

                        return (
                          <React.Fragment key={`lm-${box.id}-${point.id}`}>
                            <Circle
                              x={point.x}
                              y={point.y}
                              radius={getScaledRadius()}
                              fill={color}
                              opacity={landmarkOpacity}
                            />
                            <Text
                              x={point.x + 2.5}
                              y={point.y - 8}
                              text={(lmIndex + 1).toString()}
                              fontSize={getTextConfig().fontSize}
                              fill={color}
                              opacity={landmarkOpacity}
                            />
                          </React.Fragment>
                        );
                      })}
                    </React.Fragment>
                  );
                })}
              </Layer>
                </Stage>
              </div>

              {/* Landmark Placement Guide */}
              {showGuide && activeSchema && (
                <div className="w-64 shrink-0 overflow-y-auto">
                  <LandmarkPlacementGuide
                    schema={activeSchema}
                    placedLandmarks={selectedBoxLandmarks}
                    onSkip={skipBoxId ? () => skipLandmark(skipBoxId) : undefined}
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
