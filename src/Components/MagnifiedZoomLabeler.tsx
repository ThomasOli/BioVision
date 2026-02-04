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
    undo,
    redo,
  } = useContext(UndoRedoClearContext);

  const [image, imageDimensions, imageError] = useImageLoader(imageURL);
  const stageRef = useRef<Konva.Stage>(null);

  // Track if we just created a box to avoid double-adding landmarks
  const pendingBoxRef = useRef<{ x: number; y: number } | null>(null);

  // Show/hide landmark guide
  const [showGuide, setShowGuide] = useState(true);

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

      // Use the first box
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
              <div className="flex-1 min-w-0">
                <Stage
                  width={imageDimensions.width * scale}
                  height={imageDimensions.height * scale}
                  onClick={handleCanvasClick}
                  ref={stageRef}
                  style={{
                    border: "1px solid hsl(var(--border))",
                    backgroundColor: "hsl(var(--muted))",
                    cursor: mode ? "default" : "crosshair",
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
