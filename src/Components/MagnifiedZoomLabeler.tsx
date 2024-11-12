// src/Components/MagnifiedImageLabeler.tsx
import React, {
  useRef,
  useState,
  useCallback,
  useEffect,
  useContext,
} from "react";
import { Modal, Box, IconButton } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import { Stage, Layer, Image as KonvaImage, Circle, Text } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { UndoRedoClearContext } from "./UndoRedoClearContext";

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

  // Define maximum dimensions for the magnified view
  const MAX_WIDTH = window.innerWidth * 0.9;
  const MAX_HEIGHT = window.innerHeight * 0.9;

  // Calculate scaling factor to fit the image within the modal
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
    return Math.max(baseRadius, imageDiagonal * 0.002); // 0.3% of diagonal length
  }, [imageDimensions]);
  // Handle canvas click to add a point
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;

      const stage = e.target.getStage();
      const pointerPosition = stage?.getPointerPosition();
      if (pointerPosition) {
        // Convert pointer position to image coordinates
        const x = (pointerPosition.x - stage!.x()) / scale;
        const y = (pointerPosition.y - stage!.y()) / scale;

        // Check if click is within image boundaries
        if (
          x < 0 ||
          y < 0 ||
          x > imageDimensions.width ||
          y > imageDimensions.height
        ) {
          return; // Click outside image area
        }

        const newPoint: Point = {
          x: x,
          y: y,
          id: Date.now(),
        };
        addPoint(newPoint);
      }
    },
    [image, imageDimensions, points, onPointsChange, scale]
  );

  const getTextConfig = useCallback(() => {
    if (!imageDimensions) return { fontSize: 7, offsetX: 5, offsetY: 5 };
    const imageDiagonal = Math.sqrt(
      Math.pow(imageDimensions.width, 2) + Math.pow(imageDimensions.height, 2)
    );
    const fontSize = Math.max(6, imageDiagonal * 0.007); // 0.8% of diagonal length
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

  // // Remove a point
  // const handlePointRightClick = useCallback(
  //   (e: KonvaEventObject<PointerEvent>, id: number) => {
  //     e.evt.preventDefault();
  //     const updatedPoints = points.filter((point) => point.id !== id);
  //     setPoints(updatedPoints);
  //     onPointsChange(updatedPoints);
  //   },
  //   [points, onPointsChange]
  // );

  // // Handle point drag end
  // const handlePointDragEnd = useCallback(
  //   (e: KonvaEventObject<DragEvent>, id: number) => {
  //     const { x, y } = e.target.position();
  //     const updatedPoints = points.map((p) => (p.id === id ? { ...p, x: x / scale, y: y / scale } : p));
  //     setPoints(updatedPoints);
  //     onPointsChange(updatedPoints);
  //   },
  //   [points, onPointsChange, scale]
  // );

  if (imageError) {
    return <div style={{ color: "red" }}>Error loading image.</div>;
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      aria-labelledby="magnified-image-labeler"
      closeAfterTransition
    >
      <Box
        sx={{
          position: "absolute" as "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          maxWidth: MAX_WIDTH,
          maxHeight: MAX_HEIGHT,
          bgcolor: "background.paper",
          boxShadow: 24,
          p: 2,
          outline: "none",
          overflow: "auto", // To handle images larger than the viewport
        }}
      >
        <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
          <IconButton onClick={onClose} aria-label="Close Magnified View">
            <CloseIcon />
          </IconButton>
        </Box>
        {image && imageDimensions && (
          <Stage
            width={imageDimensions.width * scale}
            height={imageDimensions.height * scale}
            onClick={handleCanvasClick}
            ref={stageRef}
            style={{
              border: "1px solid gray",
              backgroundColor: "#f0f0f0",
              cursor: "crosshair",
              display: "block",
              margin: "0 auto",
            }}
          >
            <Layer scaleX={scale} scaleY={scale}>
              {/* Render the uploaded image */}
              <KonvaImage
                image={image}
                width={imageDimensions.width}
                height={imageDimensions.height}
              />
              {/* Render the points */}
              {points.map((point, index) => (
                <React.Fragment key={point.id}>
                  <Circle
                    x={point.x}
                    y={point.y}
                    radius={getScaledRadius()} // Fixed radius
                    fill={color}
                    opacity={opacity / 100} // Convert percentage to decimal
                    // draggable
                    // onDragEnd={(e) => handlePointDragEnd(e, point.id)}
                    // onContextMenu={(e) => handlePointRightClick(e, point.id)}
                  />
                  <Text
                    x={point.x + 2.5}
                    y={point.y - 8}
                    text={(index + 1).toString()}
                    fontSize={getTextConfig().fontSize} // Fixed font size
                    fill={color}
                    opacity={opacity / 100}
                  />
                </React.Fragment>
              ))}
            </Layer>
          </Stage>
        )}
      </Box>
    </Modal>
  );
};

export default MagnifiedImageLabeler;
