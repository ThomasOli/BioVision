import React, { useRef, useCallback, useEffect, useContext } from "react";
import { motion } from "framer-motion";
import { X } from "lucide-react";
import { Stage, Layer, Image as KonvaImage, Circle, Text } from "react-konva";
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

interface Point {
  x: number;
  y: number;
  id: number;
}

interface MagnifiedImageLabelerProps {
  imageURL: string;
  onPointsChange: (points: Point[]) => void;
  color: string;
  opacity: number;
  open: boolean;
  onClose: () => void;
  mode: boolean;
}

const MagnifiedImageLabeler: React.FC<MagnifiedImageLabelerProps> = ({
  imageURL,
  onPointsChange,
  color,
  opacity,
  open,
  onClose,
  mode,
}) => {
  const { addPoint, points, undo, redo } = useContext(UndoRedoClearContext);
  const [image, imageDimensions, imageError] = useImageLoader(imageURL);
  const stageRef = useRef<any>(null);

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

  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;

      const stage = e.target.getStage();
      const pointerPosition = stage?.getPointerPosition();
      if (pointerPosition) {
        const x = (pointerPosition.x - stage!.x()) / scale;
        const y = (pointerPosition.y - stage!.y()) / scale;

        if (
          x < 0 ||
          y < 0 ||
          x > imageDimensions.width ||
          y > imageDimensions.height
        ) {
          return;
        }

        const newPoint: Point = {
          x: x,
          y: y,
          id: Date.now(),
        };
        addPoint(newPoint);
      }
    },
    [image, imageDimensions, points, onPointsChange, scale, mode, addPoint]
  );

  const getTextConfig = useCallback(() => {
    if (!imageDimensions) return { fontSize: 7, offsetX: 5, offsetY: 5 };
    const imageDiagonal = Math.sqrt(
      Math.pow(imageDimensions.width, 2) + Math.pow(imageDimensions.height, 2)
    );
    const fontSize = Math.max(6, imageDiagonal * 0.007);
    return { fontSize };
  }, [imageDimensions]);

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

  if (imageError) {
    return <div className="text-destructive">Error loading image.</div>;
  }

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
              ref={stageRef}
              style={{
                border: "1px solid hsl(var(--border))",
                backgroundColor: "hsl(var(--muted))",
                cursor: "crosshair",
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
                      x={point.x + 2.5}
                      y={point.y - 8}
                      text={(index + 1).toString()}
                      fontSize={getTextConfig().fontSize}
                      fill={color}
                      opacity={opacity / 100}
                    />
                  </React.Fragment>
                ))}
              </Layer>
            </Stage>
          )}
        </motion.div>
      </DialogContent>
    </Dialog>
  );
};

export default MagnifiedImageLabeler;
