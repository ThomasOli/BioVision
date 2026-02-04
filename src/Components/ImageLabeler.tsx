import React, { useRef, useState, useCallback, useEffect, useContext, useMemo } from "react";
import { Stage, Layer, Image as KonvaImage, Circle, Text } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";
import { BoundingBox } from "../types/Image";
import Konva from "konva";

interface ImageLabelerProps {
  imageURL: string;
  onBoxesChange: (boxes: BoundingBox[]) => void;
  color: string;
  opacity: number;
  mode: boolean; // View-only mode
}

const ImageLabeler: React.FC<ImageLabelerProps> = ({
  imageURL,
  onBoxesChange,
  color,
  opacity,
  mode,
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

  // Click to add landmark - auto-creates box if needed
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;

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

      // Use the first box (we only have one default box now)
      const targetBox = boxes[0];

      // Auto-select the box if not selected
      if (selectedBoxId !== targetBox.id) {
        selectBox(targetBox.id);
      }

      addLandmark(targetBox.id, { x: Math.round(pos.x), y: Math.round(pos.y) });
    },
    [image, imageDimensions, mode, getPointerPosition, isPointInBounds, boxes, selectedBoxId, addBox, selectBox, addLandmark]
  );

  useEffect(() => {
    onBoxesChange(boxes);
  }, [boxes, onBoxesChange]);

  if (imageError) {
    return <div className="text-destructive">Error loading image.</div>;
  }

  const stageW = imageDimensions ? imageDimensions.width * scale : 0;
  const stageH = imageDimensions ? imageDimensions.height * scale : 0;

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
              ))}
            </Layer>
          </Stage>
        )}
      </div>
    </div>
  );
};

export default ImageLabeler;
