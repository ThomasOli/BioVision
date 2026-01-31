import React, { useRef, useCallback, useEffect, useContext, useState } from "react";
import { motion } from "framer-motion";
import { X } from "lucide-react";
import { Stage, Layer, Image as KonvaImage, Circle, Text, Rect, Transformer } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";
import { Button } from "@/Components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogClose,
} from "@/Components/ui/dialog";
import { modalContent } from "@/lib/animations";
import { ToolMode, BoundingBox } from "../types/Image";
import Konva from "konva";
import { toast } from "sonner";

interface MagnifiedImageLabelerProps {
  imageURL: string;
  onBoxesChange: (boxes: BoundingBox[]) => void;
  color: string;
  opacity: number;
  open: boolean;
  onClose: () => void;
  mode: boolean;
  toolMode: ToolMode;
}

const MIN_BOX_SIZE = 10;

const MagnifiedImageLabeler: React.FC<MagnifiedImageLabelerProps> = ({
  imageURL,
  color,
  opacity,
  open,
  onClose,
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

  // Box drawing state
  const [isDrawingBox, setIsDrawingBox] = useState(false);
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(null);
  const [boxPreview, setBoxPreview] = useState<{ left: number; top: number; width: number; height: number } | null>(null);

  const MAX_WIDTH = window.innerWidth * 0.9;
  const MAX_HEIGHT = window.innerHeight * 0.9;

  const calculateScale = () => {
    if (!imageDimensions) return 1;
    const widthScale = MAX_WIDTH / imageDimensions.width;
    const heightScale = MAX_HEIGHT / imageDimensions.height;
    return Math.min(widthScale, heightScale);
  };

  const scale = calculateScale();
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

  // Attach transformer to selected box
  useEffect(() => {
    if (!transformerRef.current || toolMode !== "select" || !open) return;

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
  }, [selectedBoxId, boxes, toolMode, open]);

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
        setIsDrawingBox(true);
        setBoxStart(pos);
        setBoxPreview({ left: pos.x, top: pos.y, width: 0, height: 0 });
      } else if (toolMode === "select") {
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

      const left = Math.min(boxStart.x, pos.x);
      const top = Math.min(boxStart.y, pos.y);
      const width = Math.abs(pos.x - boxStart.x);
      const height = Math.abs(pos.y - boxStart.y);

      setBoxPreview({ left, top, width, height });
    },
    [isDrawingBox, boxStart, toolMode, getPointerPosition]
  );

  const handleMouseUp = useCallback(() => {
    if (toolMode === "box" && isDrawingBox && boxPreview) {
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
  }, [toolMode, isDrawingBox, boxPreview, addBox]);

  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;
      if (toolMode !== "landmark") return;

      const pos = getPointerPosition(e);
      if (!pos || !isPointInBounds(pos.x, pos.y)) return;

      const clickedBox = findBoxContainingPoint(pos.x, pos.y);

      if (!clickedBox) {
        toast.info("Click inside a bounding box to add a landmark");
        return;
      }

      addLandmark(clickedBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
    },
    [image, imageDimensions, mode, toolMode, getPointerPosition, isPointInBounds, findBoxContainingPoint, addLandmark]
  );

  const handleTransformEnd = useCallback(
    (e: Konva.KonvaEventObject<Event>) => {
      const node = e.target as Konva.Rect;
      const boxId = parseInt(node.id().replace("box-", ""), 10);

      const newLeft = node.x();
      const newTop = node.y();
      const newWidth = node.width() * node.scaleX();
      const newHeight = node.height() * node.scaleY();

      node.scaleX(1);
      node.scaleY(1);

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

  if (imageError) {
    return <div className="text-destructive">Error loading image.</div>;
  }

  // Track global landmark index for numbering
  let globalLandmarkIndex = 0;

  return (
    <Dialog open={open} onOpenChange={(value) => !value && onClose()}>
      <DialogContent
        className="max-w-[90vw] max-h-[90vh] p-4 overflow-auto"
        style={{
          width: imageDimensions ? imageDimensions.width * scale + 32 : "auto",
        }}
      >
        <motion.div
          variants={modalContent}
          initial="hidden"
          animate="visible"
          exit="exit"
        >
          <div className="flex justify-end mb-2">
            <DialogClose asChild>
              <Button variant="ghost" size="icon" aria-label="Close Magnified View">
                <X className="h-4 w-4" />
              </Button>
            </DialogClose>
          </div>

          {image && imageDimensions && (
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
                cursor: getCursor(),
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
        </motion.div>
      </DialogContent>
    </Dialog>
  );
};

export default MagnifiedImageLabeler;
