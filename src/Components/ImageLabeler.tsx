import React, { useRef, useState, useCallback, useEffect, useContext, useMemo } from "react";
import { Stage, Layer, Image as KonvaImage, Circle, Text, Rect, Line, Transformer } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";
import { BoundingBox } from "../types/Image";
import { DetectionMode } from "./DetectionModeSelector";
import Konva from "konva";

interface ImageLabelerProps {
  imageURL: string;
  onBoxesChange: (boxes: BoundingBox[]) => void;
  color: string;
  opacity: number;
  mode: boolean; // View-only mode
  detectionMode?: DetectionMode;
  autoCorrectionMode?: boolean;
}

const ImageLabeler: React.FC<ImageLabelerProps> = ({
  imageURL,
  onBoxesChange,
  color,
  opacity,
  mode,
  detectionMode = "manual",
  autoCorrectionMode = false,
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

  // In manual mode, hide predicted boxes until user manually draws/edits boxes.
  const visibleBoxes = useMemo<BoundingBox[]>(() => {
    if (detectionMode !== "manual") return boxes;
    return boxes.filter((box) => box.source !== "predicted");
  }, [boxes, detectionMode]);

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
      if (mode) return;
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
    [mode, detectionMode, autoCorrectionMode, selectedBoxId, getPointerPosition, isPointInBounds, findBoxAtPoint]
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
        } else if (
          detectionMode === "auto" &&
          autoCorrectionMode &&
          isRedrawingSelected &&
          selectedBoxId !== null
        ) {
          updateBox(selectedBoxId, {
            left: Math.round(left),
            top: Math.round(top),
            width: Math.round(width),
            height: Math.round(height),
            source: "corrected",
            confidence: undefined,
            maskOutline: undefined,
            detectionMethod: "human_corrected",
          });
        }
      }

      setIsDrawingBox(false);
      setIsRedrawingSelected(false);
      setDrawStart(null);
      setDrawCurrent(null);
    },
    [isDrawingBox, drawStart, drawCurrent, imageDimensions, addBox, detectionMode, autoCorrectionMode, isRedrawingSelected, selectedBoxId, updateBox]
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

    if (detectionMode === "auto" && autoCorrectionMode && node) {
      tr.nodes([node]);
      tr.getLayer()?.batchDraw();
      return;
    }

    tr.nodes([]);
    tr.getLayer()?.batchDraw();
  }, [detectionMode, autoCorrectionMode, selectedBoxId, visibleBoxes]);

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
                const isEditableSelected = detectionMode === "auto" && autoCorrectionMode && isSelected;

                return (
                  <React.Fragment key={`box-${box.id}`}>
                    {/* SAM2 mask polygon overlay */}
                    {box.maskOutline && box.maskOutline.length > 0 && (
                      <Line
                        points={box.maskOutline.flat()}
                        closed={true}
                        fill={isSelected ? "rgba(59, 130, 246, 0.15)" : "rgba(100, 100, 100, 0.1)"}
                        stroke={isSelected ? "#3b82f6" : "#6b7280"}
                        strokeWidth={boxStrokeWidth * 0.5}
                      />
                    )}
                    <Rect
                      ref={isEditableSelected ? selectedRectRef : undefined}
                      x={box.left}
                      y={box.top}
                      width={box.width}
                      height={box.height}
                      stroke={boxColor}
                      strokeWidth={boxStrokeWidth}
                      dash={isSelected ? undefined : [10, 5]}
                      fill={isSelected ? "rgba(59, 130, 246, 0.1)" : "transparent"}
                      draggable={isEditableSelected}
                      onDragEnd={(e) => {
                        if (!isEditableSelected || !imageDimensions) return;
                        const node = e.target;
                        const nextLeft = Math.max(0, Math.min(node.x(), imageDimensions.width - box.width));
                        const nextTop = Math.max(0, Math.min(node.y(), imageDimensions.height - box.height));
                        updateBox(box.id, {
                          left: Math.round(nextLeft),
                          top: Math.round(nextTop),
                          source: "corrected",
                          confidence: undefined,
                          maskOutline: undefined,
                          detectionMethod: "human_corrected",
                        });
                      }}
                      onTransformEnd={() => {
                        if (!isEditableSelected || !imageDimensions || !selectedRectRef.current) return;
                        const node = selectedRectRef.current;
                        const scaleX = node.scaleX();
                        const scaleY = node.scaleY();

                        const rawLeft = node.x();
                        const rawTop = node.y();
                        const rawWidth = Math.max(12, node.width() * scaleX);
                        const rawHeight = Math.max(12, node.height() * scaleY);

                        node.scaleX(1);
                        node.scaleY(1);

                        const left = Math.max(0, Math.min(rawLeft, imageDimensions.width - rawWidth));
                        const top = Math.max(0, Math.min(rawTop, imageDimensions.height - rawHeight));
                        const width = Math.max(12, Math.min(rawWidth, imageDimensions.width - left));
                        const height = Math.max(12, Math.min(rawHeight, imageDimensions.height - top));

                        updateBox(box.id, {
                          left: Math.round(left),
                          top: Math.round(top),
                          width: Math.round(width),
                          height: Math.round(height),
                          source: "corrected",
                          confidence: undefined,
                          maskOutline: undefined,
                          detectionMethod: "human_corrected",
                        });
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
                  </React.Fragment>
                );
              })}

              {detectionMode === "auto" && autoCorrectionMode && (
                <Transformer
                  ref={transformerRef}
                  rotateEnabled={false}
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
                    if (!imageDimensions) return oldBox;
                    if (newBox.width < 12 || newBox.height < 12) return oldBox;
                    if (newBox.x < 0 || newBox.y < 0) return oldBox;
                    if (newBox.x + newBox.width > imageDimensions.width) return oldBox;
                    if (newBox.y + newBox.height > imageDimensions.height) return oldBox;
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
