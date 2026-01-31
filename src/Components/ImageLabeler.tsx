import React, { useRef, useState, useCallback, useEffect, useContext, useMemo } from "react";
import { Stage, Layer, Image as KonvaImage, Circle, Text } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";

interface Point {
  x: number;
  y: number;
  id: number;
}

interface ImageLabelerProps {
  imageURL: string;
  onPointsChange: (points: Point[]) => void;
  color: string;
  opacity: number;
  mode: boolean;
}

const ImageLabeler: React.FC<ImageLabelerProps> = ({
  imageURL,
  onPointsChange,
  color,
  opacity,
  mode,
}) => {
  const { addPoint, points, undo, redo } = useContext(UndoRedoClearContext);

  const [image, imageDimensions, imageError] = useImageLoader(imageURL);
  const stageRef = useRef<any>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerSize, setContainerSize] = useState<{ w: number; h: number }>({
    w: 800,
    h: 600,
  });

  const [scale, setScale] = useState(1);

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

  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;

      const stage = e.target.getStage();
      const pointerPosition = stage?.getPointerPosition();
      if (!pointerPosition) return;

      const x = (pointerPosition.x - stage!.x()) / stage!.scaleX();
      const y = (pointerPosition.y - stage!.y()) / stage!.scaleY();

      if (
        x < 0 ||
        y < 0 ||
        x > imageDimensions.width ||
        y > imageDimensions.height
      )
        return;

      addPoint({ x: Math.round(x), y: Math.round(y), id: Date.now() });
    },
    [image, imageDimensions, mode, addPoint]
  );

  useEffect(() => {
    onPointsChange(points);
  }, [points, onPointsChange]);

  if (imageError) {
    return <div className="text-destructive">Error loading image.</div>;
  }

  const stageW = imageDimensions ? imageDimensions.width * scale : 0;
  const stageH = imageDimensions ? imageDimensions.height * scale : 0;

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
              {points.map((point, index) => (
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
                    text={(index + 1).toString()}
                    fontSize={getTextFontSize}
                    fill={color}
                    opacity={opacity / 100}
                  />
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
