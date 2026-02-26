import React, { useRef, useState, useCallback, useEffect, useContext, useMemo } from "react";
import { Stage, Layer, Image as KonvaImage, Circle, Text, Rect, Line, Transformer } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";
import { BoundingBox } from "../types/Image";
import { DetectionMode } from "./DetectionModeSelector";
import Konva from "konva";

// ── OBB / Transformer helpers ─────────────────────────────────────────────
/** Derive a center-anchored Konva Rect description from a BoundingBox. */
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
/** Build 4 OBB corner points from center, size, and rotation angle (degrees). */
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
/** Compute AABB of 4 corners, clamped to image dimensions. */
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

interface ImageLabelerProps {
  imageURL: string;
  onBoxesChange: (boxes: BoundingBox[]) => void;
  color: string;
  opacity: number;
  mode: boolean; // View-only mode
  detectionMode?: DetectionMode;
  autoCorrectionMode?: boolean;
  imagePath?: string;    // Disk path for SAM2 re-segmentation
  samEnabled?: boolean;  // Whether SAM2 is active — triggers auto re-segment on box resize
  hideSegmentOutlines?: boolean; // Hide SAM2 mask overlays (e.g. after finalize)
  lockBoxes?: boolean;           // Prevent drawing/adding new boxes (landmark-only mode)
}

const ImageLabeler: React.FC<ImageLabelerProps> = ({
  imageURL,
  onBoxesChange,
  color,
  opacity,
  mode,
  detectionMode = "manual",
  autoCorrectionMode = false,
  imagePath,
  samEnabled = false,
  hideSegmentOutlines = false,
  lockBoxes = false,
}) => {
  const {
    addLandmark,
    boxes,
    selectedBoxId,
    addBox,
    selectBox,
    updateBox,
    undo,
    redo,
  } = useContext(UndoRedoClearContext);

  // Track which box is currently being re-segmented by SAM2
  const [resegmentingBoxId, setResegmentingBoxId] = useState<number | null>(null);

  // Use refs to avoid re-running keyboard effect when undo/redo change
  const undoRef = useRef(undo);
  const redoRef = useRef(redo);
  useEffect(() => {
    undoRef.current = undo;
    redoRef.current = redo;
  }, [undo, redo]);

  const [image, imageDimensions, imageError] = useImageLoader(imageURL);
  const stageRef = useRef<Konva.Stage>(null);
  const selectedRectRef = useRef<Konva.Rect>(null);
  const transformerRef = useRef<Konva.Transformer>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerSize, setContainerSize] = useState<{ w: number; h: number }>({
    w: 800,
    h: 600,
  });

  const [scale, setScale] = useState(1);

  // Drag-to-draw bounding box state
  const [isDrawingBox, setIsDrawingBox] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);
  const [isRedrawingSelected, setIsRedrawingSelected] = useState(false);

  // Track if we just created a box to avoid double-adding landmarks
  const pendingBoxRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const el = containerRef.current;
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      setContainerSize({ w: rect.width, h: rect.height });
    });

    ro.observe(el);
    const rect = el.getBoundingClientRect();
    setContainerSize({ w: rect.width, h: rect.height });

    return () => ro.disconnect();
  }, []);

  const baseRadius = 3;
  const getScaledRadius = useCallback(() => {
    if (!imageDimensions) return baseRadius;
    return Math.max(baseRadius, imageDimensions.width * 0.003);
  }, [imageDimensions]);

  const getTextFontSize = useMemo(() => {
    if (!imageDimensions) return 7;
    const imageDiagonal = Math.sqrt(
      imageDimensions.width ** 2 + imageDimensions.height ** 2
    );
    return Math.max(7, imageDiagonal * 0.01);
  }, [imageDimensions]);

  // Box stroke width based on image size
  const boxStrokeWidth = useMemo(() => {
    if (!imageDimensions) return 2;
    return Math.max(2, imageDimensions.width * 0.002);
  }, [imageDimensions]);

  useEffect(() => {
    if (!imageDimensions) return;

    const widthScale = containerSize.w / imageDimensions.width;
    const heightScale = containerSize.h / imageDimensions.height;

    const next = Math.min(widthScale, heightScale) * 0.98;
    setScale(Math.min(next, 1));
  }, [imageDimensions, containerSize]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isMod = e.ctrlKey || e.metaKey; // Ctrl on Windows, Cmd on Mac
      if (isMod && e.key === "z") {
        e.preventDefault();
        undoRef.current();
      } else if (isMod && e.key === "y") {
        e.preventDefault();
        redoRef.current();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []); // Empty deps - uses refs for stable reference

  // When boxes change and we have a pending landmark to add
  useEffect(() => {
    if (pendingBoxRef.current && boxes.length > 0) {
      const pos = pendingBoxRef.current;
      pendingBoxRef.current = null;
      const targetBox = boxes[0];
      addLandmark(targetBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
    }
  }, [boxes, addLandmark]);

  // Auto-select the most-recently drawn box so the Transformer appears immediately
  const pendingSelectRef = useRef<boolean>(false);
  useEffect(() => {
    if (pendingSelectRef.current && boxes.length > 0) {
      pendingSelectRef.current = false;
      selectBox(boxes[boxes.length - 1].id);
    }
  }, [boxes, selectBox]);

  const getPointerPosition = useCallback((e: KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    const pointerPosition = stage?.getPointerPosition();
    if (!pointerPosition || !stage) return null;

    const x = (pointerPosition.x - stage.x()) / stage.scaleX();
    const y = (pointerPosition.y - stage.y()) / stage.scaleY();
    return { x, y };
  }, []);

  const isPointInBounds = useCallback(
    (x: number, y: number) => {
      if (!imageDimensions) return false;
      return x >= 0 && y >= 0 && x <= imageDimensions.width && y <= imageDimensions.height;
    },
    [imageDimensions]
  );

  // Re-run SAM2 on a corrected bounding box and update its maskOutline
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

  // Keep one shared box set across manual/auto modes so switching modes
  // never hides or drops accepted boxes.
  const visibleBoxes = useMemo<BoundingBox[]>(() => {
    return boxes;
  }, [boxes]);

  // Check if point is inside a box
  const findBoxAtPoint = useCallback(
    (x: number, y: number): BoundingBox | null => {
      // Check boxes in reverse order (top-most first)
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
    [visibleBoxes]
  );

  // Drag-to-draw handlers for manual mode
  const handleMouseDown = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (mode || lockBoxes) return;
      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;

      // Don't start drawing if clicking inside an existing visible box
      const clickedBox = findBoxAtPoint(pos.x, pos.y);
      if (detectionMode === "manual") {
        if (clickedBox) return;
        setIsDrawingBox(true);
        setIsRedrawingSelected(false);
        setDrawStart(pos);
        setDrawCurrent(pos);
        return;
      }

      // Auto mode correction: redraw selected box by dragging in empty area
      if (detectionMode === "auto" && autoCorrectionMode) {
        if (clickedBox) return;
        if (selectedBoxId === null) return;
        setIsDrawingBox(true);
        setIsRedrawingSelected(true);
        setDrawStart(pos);
        setDrawCurrent(pos);
      }
    },
    [mode, lockBoxes, detectionMode, autoCorrectionMode, selectedBoxId, getPointerPosition, isPointInBounds, findBoxAtPoint]
  );

  const handleMouseMove = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!isDrawingBox || !drawStart) return;
      const pos = getPointerPosition(e);
      if (!pos) return;

      // Clamp to image bounds
      const x = imageDimensions ? Math.max(0, Math.min(pos.x, imageDimensions.width)) : pos.x;
      const y = imageDimensions ? Math.max(0, Math.min(pos.y, imageDimensions.height)) : pos.y;
      setDrawCurrent({ x, y });
    },
    [isDrawingBox, drawStart, getPointerPosition, imageDimensions]
  );

  const handleMouseUp = useCallback(
    () => {
      if (!isDrawingBox || !drawStart || !drawCurrent) {
        setIsDrawingBox(false);
        setIsRedrawingSelected(false);
        setDrawStart(null);
        setDrawCurrent(null);
        return;
      }

      const left = Math.min(drawStart.x, drawCurrent.x);
      const top = Math.min(drawStart.y, drawCurrent.y);
      const width = Math.abs(drawCurrent.x - drawStart.x);
      const height = Math.abs(drawCurrent.y - drawStart.y);

      // Minimum size threshold to avoid accidental micro-boxes
      const minSize = imageDimensions ? Math.max(20, imageDimensions.width * 0.02) : 20;
      if (width >= minSize && height >= minSize) {
        if (detectionMode === "manual") {
          addBox({
            left: Math.round(left),
            top: Math.round(top),
            width: Math.round(width),
            height: Math.round(height),
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

      setIsDrawingBox(false);
      setIsRedrawingSelected(false);
      setDrawStart(null);
      setDrawCurrent(null);
    },
    [isDrawingBox, drawStart, drawCurrent, imageDimensions, addBox, detectionMode, autoCorrectionMode, isRedrawingSelected, selectedBoxId, updateBox, triggerResegment]
  );

  // Preview rectangle for drag-to-draw
  const drawPreview = useMemo(() => {
    if (!isDrawingBox || !drawStart || !drawCurrent) return null;
    return {
      x: Math.min(drawStart.x, drawCurrent.x),
      y: Math.min(drawStart.y, drawCurrent.y),
      width: Math.abs(drawCurrent.x - drawStart.x),
      height: Math.abs(drawCurrent.y - drawStart.y),
    };
  }, [isDrawingBox, drawStart, drawCurrent]);

  // Click to add landmark or select box
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;

      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;

      // In auto mode, check if clicking on a box to select it
      if (detectionMode === "auto") {
        const clickedBox = findBoxAtPoint(pos.x, pos.y);

        if (clickedBox) {
          // If clicking on a different box, select it
          if (selectedBoxId !== clickedBox.id) {
            selectBox(clickedBox.id);
            return;
          }
          // In correction mode, keep click for selection only.
          if (!autoCorrectionMode) {
            // If clicking on already selected box, add landmark
            addLandmark(clickedBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
          }
          return;
        }

        // Clicking outside any box in auto mode - do nothing
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
        // Clicking outside boxes in manual mode does nothing (drag to draw a new box)
        return;
      }
    },
    [image, imageDimensions, mode, getPointerPosition, isPointInBounds, selectedBoxId, selectBox, addLandmark, detectionMode, findBoxAtPoint, autoCorrectionMode]
  );

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

  useEffect(() => {
    onBoxesChange(boxes);
  }, [boxes, onBoxesChange]);

  if (imageError) {
    return <div className="text-destructive">Error loading image.</div>;
  }

  const stageW = imageDimensions ? imageDimensions.width * scale : 0;
  const stageH = imageDimensions ? imageDimensions.height * scale : 0;

  // Colors for boxes
  const getBoxColor = (isSelected: boolean) => {
    if (isSelected) return "#3b82f6"; // blue-500
    return "#6b7280"; // gray-500
  };

  return (
    <div
      ref={containerRef}
      className="flex h-full w-full min-h-0 min-w-0 flex-col"
    >
      <div className="flex flex-1 min-h-0 min-w-0 items-center justify-center rounded-xl border bg-muted/30 p-4">
        {image && imageDimensions && (
          <Stage
            width={stageW}
            height={stageH}
            onClick={handleCanvasClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            ref={stageRef}
            scaleX={scale}
            scaleY={scale}
            style={{
              borderRadius: "10px",
              overflow: "hidden",
              border: "1px solid hsl(var(--border))",
              backgroundColor: "hsl(var(--background))",
              cursor: mode ? "default" : isDrawingBox ? "crosshair" : "crosshair",
            }}
          >
            <Layer>
              <KonvaImage
                image={image}
                width={imageDimensions.width}
                height={imageDimensions.height}
              />

              {/* Draw preview rectangle during drag */}
              {drawPreview && (
                <Rect
                  x={drawPreview.x}
                  y={drawPreview.y}
                  width={drawPreview.width}
                  height={drawPreview.height}
                  stroke="#3b82f6"
                  strokeWidth={boxStrokeWidth}
                  dash={[8, 4]}
                  fill="rgba(59, 130, 246, 0.08)"
                />
              )}

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
                    {/* SAM2 mask polygon overlay — hidden after finalization */}
                    {!hideSegmentOutlines && box.maskOutline && box.maskOutline.length > 0 && (
                      <Line
                        points={box.maskOutline.flat()}
                        closed={true}
                        fill={isSelected ? "rgba(59, 130, 246, 0.15)" : "rgba(100, 100, 100, 0.1)"}
                        stroke={isSelected ? "#3b82f6" : "#6b7280"}
                        strokeWidth={boxStrokeWidth * 0.5}
                      />
                    )}
                    {/* OBB polygon outline (rendered when obbCorners available) */}
                    {box.obbCorners && box.obbCorners.length === 4 && (
                      <Line
                        points={box.obbCorners.flat()}
                        closed={true}
                        stroke={boxColor}
                        strokeWidth={boxStrokeWidth}
                        dash={isSelected ? undefined : [8, 4]}
                        fill={isSelected ? "rgba(59, 130, 246, 0.08)" : "transparent"}
                        listening={!isEditableSelected}
                      />
                    )}
                    {/* Interactive rect — center-anchored so rotation uses center pivot.
                        Transparent when OBB polygon (Line above) provides the visual. */}
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
                      fill={isEditableSelected && hasObb ? "rgba(0,0,0,0.001)" : "transparent"}
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
                      fontSize={getTextFontSize * 1.2}
                      fill={boxColor}
                      fontStyle="bold"
                    />
                    {/* Confidence badge if available */}
                    {box.confidence !== undefined && (
                      <Text
                        x={box.left + 5}
                        y={box.top + 5 + getTextFontSize * 1.5}
                        text={`${(box.confidence * 100).toFixed(0)}%`}
                        fontSize={getTextFontSize * 0.9}
                        fill={boxColor}
                      />
                    )}
                    {/* SAM2 re-segmenting indicator */}
                    {resegmentingBoxId === box.id && (
                      <Text
                        x={box.left + 5}
                        y={box.top + box.height - getTextFontSize * 1.5 - 5}
                        text="⟳ Segmenting…"
                        fontSize={getTextFontSize * 0.9}
                        fill="#60a5fa"
                        fontStyle="bold"
                      />
                    )}
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
                      // Skip rendering for skipped landmarks
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
                            x={point.x + 4}
                            y={point.y - 11}
                            text={(lmIndex + 1).toString()}
                            fontSize={getTextFontSize}
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
        )}
      </div>
    </div>
  );
};

export default ImageLabeler;
