import React, { useRef, useState, useCallback, useEffect, useContext, useMemo } from "react";
import { Stage, Layer, Image as KonvaImage, Circle, Text, Rect, Transformer } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";
import { ToolMode, BoundingBox } from "../types/Image";
import Konva from "konva";
import { toast } from "sonner";

interface ImageLabelerProps {
  imageURL: string;
  onBoxesChange: (boxes: BoundingBox[]) => void;
  color: string;
  opacity: number;
  mode: boolean;
  toolMode: ToolMode;
}

const MIN_BOX_SIZE = 10;

const ImageLabeler: React.FC<ImageLabelerProps> = ({
  imageURL,
  onBoxesChange,
  color,
  opacity,
  mode,
  toolMode,
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

  const [image, imageDimensions, imageError] = useImageLoader(imageURL);
  const stageRef = useRef<Konva.Stage>(null);
  const transformerRef = useRef<Konva.Transformer>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerSize, setContainerSize] = useState<{ w: number; h: number }>({
    w: 800,
    h: 600,
  });

  const [scale, setScale] = useState(1);

  // Box drawing state
  const [isDrawingBox, setIsDrawingBox] = useState(false);
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(null);
  const [boxPreview, setBoxPreview] = useState<{ left: number; top: number; width: number; height: number } | null>(null);

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
    const imageDiagonal = Math.sqrt(
      imageDimensions.width ** 2 + imageDimensions.height ** 2
    );
    return Math.max(baseRadius, imageDiagonal * 0.003);
  }, [imageDimensions]);

  const getTextFontSize = useMemo(() => {
    if (!imageDimensions) return 7;
    const imageDiagonal = Math.sqrt(
      imageDimensions.width ** 2 + imageDimensions.height ** 2
    );
    return Math.max(7, imageDiagonal * 0.01);
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
      if (e.ctrlKey && e.key === "z") {
        e.preventDefault();
        undo();
      } else if (e.ctrlKey && e.key === "y") {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [undo, redo]);

  // Attach transformer to selected box
  useEffect(() => {
    if (!transformerRef.current || toolMode !== "select") return;

    const stage = stageRef.current;
    if (!stage) return;

    if (selectedBoxId !== null) {
      const selectedNode = stage.findOne(`#box-${selectedBoxId}`);
      if (selectedNode) {
        transformerRef.current.nodes([selectedNode]);
        transformerRef.current.getLayer()?.batchDraw();
      } else {
        transformerRef.current.nodes([]);
      }
    } else {
      transformerRef.current.nodes([]);
    }
  }, [selectedBoxId, boxes, toolMode]);

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

  const findBoxContainingPoint = useCallback(
    (x: number, y: number): BoundingBox | undefined => {
      return boxes.find(
        (box) =>
          x >= box.left &&
          x <= box.left + box.width &&
          y >= box.top &&
          y <= box.top + box.height
      );
    },
    [boxes]
  );

  const handleMouseDown = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;

      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;

      if (toolMode === "box") {
        // Start drawing a box
        setIsDrawingBox(true);
        setBoxStart(pos);
        setBoxPreview({ left: pos.x, top: pos.y, width: 0, height: 0 });
      } else if (toolMode === "select") {
        // Check if we clicked on a box
        const clickedBox = findBoxContainingPoint(pos.x, pos.y);
        if (clickedBox) {
          selectBox(clickedBox.id);
        } else {
          selectBox(null);
        }
      }
    },
    [image, imageDimensions, mode, toolMode, getPointerPosition, isPointInBounds, findBoxContainingPoint, selectBox]
  );

  const handleMouseMove = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!isDrawingBox || !boxStart || toolMode !== "box") return;

      const pos = getPointerPosition(e);
      if (!pos) return;

      // Calculate box dimensions (handle drag in any direction)
      const left = Math.min(boxStart.x, pos.x);
      const top = Math.min(boxStart.y, pos.y);
      const width = Math.abs(pos.x - boxStart.x);
      const height = Math.abs(pos.y - boxStart.y);

      setBoxPreview({ left, top, width, height });
    },
    [isDrawingBox, boxStart, toolMode, getPointerPosition]
  );

  const handleMouseUp = useCallback(
    () => {
      if (toolMode === "box" && isDrawingBox && boxPreview) {
        // Finalize the box if it meets minimum size
        if (boxPreview.width >= MIN_BOX_SIZE && boxPreview.height >= MIN_BOX_SIZE) {
          addBox({
            left: Math.round(boxPreview.left),
            top: Math.round(boxPreview.top),
            width: Math.round(boxPreview.width),
            height: Math.round(boxPreview.height),
          });
        }
        setIsDrawingBox(false);
        setBoxStart(null);
        setBoxPreview(null);
      }
    },
    [toolMode, isDrawingBox, boxPreview, addBox]
  );

  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;

      // Only handle landmark mode clicks
      if (toolMode !== "landmark") return;

      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;

      // Find which box contains this click
      const clickedBox = findBoxContainingPoint(pos.x, pos.y);

      if (!clickedBox) {
        toast.info("Click inside a bounding box to add a landmark");
        return;
      }

      // Auto-select the box and add landmark
      addLandmark(clickedBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
    },
    [image, imageDimensions, mode, toolMode, getPointerPosition, isPointInBounds, findBoxContainingPoint, addLandmark]
  );

  const handleTransformEnd = useCallback(
    (e: Konva.KonvaEventObject<Event>) => {
      const node = e.target as Konva.Rect;
      const boxId = parseInt(node.id().replace("box-", ""), 10);

      // Get new dimensions
      const newLeft = node.x();
      const newTop = node.y();
      const newWidth = node.width() * node.scaleX();
      const newHeight = node.height() * node.scaleY();

      // Reset scale
      node.scaleX(1);
      node.scaleY(1);

      // Update box in state
      updateBox(boxId, {
        left: Math.round(newLeft),
        top: Math.round(newTop),
        width: Math.round(newWidth),
        height: Math.round(newHeight),
      });
    },
    [updateBox]
  );

  useEffect(() => {
    onBoxesChange(boxes);
  }, [boxes, onBoxesChange]);

  if (imageError) {
    return <div className="text-destructive">Error loading image.</div>;
  }

  const stageW = imageDimensions ? imageDimensions.width * scale : 0;
  const stageH = imageDimensions ? imageDimensions.height * scale : 0;

  // Determine cursor based on tool mode
  const getCursor = () => {
    if (mode) return "default";
    switch (toolMode) {
      case "box":
        return "crosshair";
      case "landmark":
        return "crosshair";
      case "select":
        return "pointer";
      default:
        return "default";
    }
  };

  // Track global landmark index for numbering
  let globalLandmarkIndex = 0;

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
              cursor: getCursor(),
            }}
          >
            <Layer>
              <KonvaImage
                image={image}
                width={imageDimensions.width}
                height={imageDimensions.height}
              />

              {/* Render existing boxes */}
              {boxes.map((box) => {
                const isSelected = box.id === selectedBoxId;
                return (
                  <React.Fragment key={box.id}>
                    <Rect
                      id={`box-${box.id}`}
                      x={box.left}
                      y={box.top}
                      width={box.width}
                      height={box.height}
                      stroke={isSelected ? "#00ff00" : "#ffffff"}
                      strokeWidth={isSelected ? 2 : 1}
                      fill={isSelected ? "rgba(0, 255, 0, 0.1)" : "rgba(255, 255, 255, 0.05)"}
                      draggable={toolMode === "select" && isSelected}
                      onDragEnd={(e) => {
                        updateBox(box.id, {
                          left: Math.round(e.target.x()),
                          top: Math.round(e.target.y()),
                        });
                      }}
                      onTransformEnd={handleTransformEnd}
                      onClick={(e) => {
                        if (toolMode === "select") {
                          e.cancelBubble = true;
                          selectBox(box.id);
                        }
                      }}
                    />
                    {/* Render landmarks inside this box */}
                    {box.landmarks.map((point) => {
                      globalLandmarkIndex++;
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
                            x={point.x + 4}
                            y={point.y - 11}
                            text={globalLandmarkIndex.toString()}
                            fontSize={getTextFontSize}
                            fill={color}
                            opacity={opacity / 100}
                          />
                        </React.Fragment>
                      );
                    })}
                  </React.Fragment>
                );
              })}

              {/* Render box preview while drawing */}
              {boxPreview && isDrawingBox && (
                <Rect
                  x={boxPreview.left}
                  y={boxPreview.top}
                  width={boxPreview.width}
                  height={boxPreview.height}
                  stroke="#00ff00"
                  strokeWidth={2}
                  dash={[5, 5]}
                  fill="rgba(0, 255, 0, 0.1)"
                />
              )}

              {/* Transformer for resizing selected box */}
              {toolMode === "select" && (
                <Transformer
                  ref={transformerRef}
                  boundBoxFunc={(oldBox, newBox) => {
                    // Enforce minimum size
                    if (newBox.width < MIN_BOX_SIZE || newBox.height < MIN_BOX_SIZE) {
                      return oldBox;
                    }
                    return newBox;
                  }}
                  rotateEnabled={false}
                  keepRatio={false}
                />
              )}
            </Layer>
          </Stage>
        )}
      </div>
    </div>
  );
};

export default ImageLabeler;
