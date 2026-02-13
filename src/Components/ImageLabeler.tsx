import React, { useRef, useState, useCallback, useEffect, useContext, useMemo } from "react";
import { Stage, Layer, Image as KonvaImage, Circle, Text, Rect } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";
import { BoundingBox } from "../types/Image";
import Konva from "konva";

export type DetectionMode = "single" | "multi";

interface ImageLabelerProps {
  imageURL: string;
  onBoxesChange: (boxes: BoundingBox[]) => void;
  color: string;
  opacity: number;
  mode: boolean; // View-only mode
  detectionMode?: DetectionMode;
}

const ImageLabeler: React.FC<ImageLabelerProps> = ({
  imageURL,
  onBoxesChange,
  color,
  opacity,
  mode,
  detectionMode = "single",
}) => {
  const {
    addLandmark,
    boxes,
    selectedBoxId,
    addBox,
    selectBox,
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

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerSize, setContainerSize] = useState<{ w: number; h: number }>({
    w: 800,
    h: 600,
  });

  const [scale, setScale] = useState(1);

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

  // Check if point is inside a box
  const findBoxAtPoint = useCallback(
    (x: number, y: number): BoundingBox | null => {
      // Check boxes in reverse order (top-most first)
      for (let i = boxes.length - 1; i >= 0; i--) {
        const box = boxes[i];
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
    [boxes]
  );

  // Click to add landmark or select box
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;

      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;

      // In multi-mode, check if clicking on a box to select it
      if (detectionMode === "multi") {
        const clickedBox = findBoxAtPoint(pos.x, pos.y);

        if (clickedBox) {
          // If clicking on a different box, select it
          if (selectedBoxId !== clickedBox.id) {
            selectBox(clickedBox.id);
            return;
          }
          // If clicking on already selected box, add landmark
          addLandmark(clickedBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
          return;
        }

        // Clicking outside any box in multi-mode - do nothing
        return;
      }

      // Single-specimen mode: auto-create a default box if none exists
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

      // Use the first box (we only have one default box now)
      const targetBox = boxes[0];

      // Auto-select the box if not selected
      if (selectedBoxId !== targetBox.id) {
        selectBox(targetBox.id);
      }

      addLandmark(targetBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
    },
    [image, imageDimensions, mode, getPointerPosition, isPointInBounds, boxes, selectedBoxId, addBox, selectBox, addLandmark, detectionMode, findBoxAtPoint]
  );

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
            ref={stageRef}
            scaleX={scale}
            scaleY={scale}
            style={{
              borderRadius: "10px",
              overflow: "hidden",
              border: "1px solid hsl(var(--border))",
              backgroundColor: "hsl(var(--background))",
              cursor: mode ? "default" : "crosshair",
            }}
          >
            <Layer>
              <KonvaImage
                image={image}
                width={imageDimensions.width}
                height={imageDimensions.height}
              />

              {/* Render bounding boxes in multi-mode */}
              {detectionMode === "multi" && boxes.map((box, index) => {
                const isSelected = selectedBoxId === box.id;
                const boxColor = getBoxColor(isSelected);

                return (
                  <React.Fragment key={`box-${box.id}`}>
                    <Rect
                      x={box.left}
                      y={box.top}
                      width={box.width}
                      height={box.height}
                      stroke={boxColor}
                      strokeWidth={boxStrokeWidth}
                      dash={isSelected ? undefined : [10, 5]}
                      fill={isSelected ? "rgba(59, 130, 246, 0.1)" : "transparent"}
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

              {/* Render landmarks */}
              {boxes.map((box) => {
                const isBoxSelected = selectedBoxId === box.id;

                return (
                  <React.Fragment key={`landmarks-${box.id}`}>
                    {box.landmarks.map((point, lmIndex) => {
                      // Skip rendering for skipped landmarks
                      if (point.isSkipped) return null;

                      // In multi-mode, dim landmarks of non-selected boxes
                      const landmarkOpacity = detectionMode === "multi" && !isBoxSelected
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
