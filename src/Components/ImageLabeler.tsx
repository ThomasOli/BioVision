// src/Components/ImageLabeler.tsx
import React, {
  useRef,
  useState,
  useCallback,
  useEffect,
  useContext,
} from "react";
import { Stage, Layer, Image as KonvaImage, Circle, Text } from "react-konva";
import useImageLoader from "../hooks/useImageLoader";
import { KonvaEventObject } from "konva/lib/Node";
import { Button, Box } from "@mui/material";
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
  const [scale, setScale] = useState(1);

  // Add this calculation for dynamic radius
  const baseRadius = 3;
  const getScaledRadius = useCallback(() => {
    if (!imageDimensions) return baseRadius;
    const imageDiagonal = Math.sqrt(
      Math.pow(imageDimensions.width, 2) + Math.pow(imageDimensions.height, 2)
    );
    return Math.max(baseRadius, imageDiagonal * 0.003); // 0.3% of diagonal length
  }, [imageDimensions]);

  // Adjust the scale based on available screen space
  useEffect(() => {
    const updateScale = () => {
      if (imageDimensions) {
        const availableWidth = window.innerWidth * 0.6; // Reserve 60% of screen width
        const availableHeight = window.innerHeight * 0.6; // Reserve 60% of screen height

        const widthScale = availableWidth / imageDimensions.width;
        const heightScale = availableHeight / imageDimensions.height;
        setScale(Math.min(widthScale, heightScale));
      }
    };

    updateScale(); // Initial scale calculation
    window.addEventListener("resize", updateScale); // Add resize event listener
    return () => window.removeEventListener("resize", updateScale); // Cleanup
  }, [imageDimensions]);

  // Add this useEffect for keyboard shortcuts
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

  // Handle canvas click to add a point
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions || mode) return;
      const stage = e.target.getStage();
      const pointerPosition = stage?.getPointerPosition();
      if (pointerPosition) {
        // Convert pointer position to image coordinates considering the scale
        const x = (pointerPosition.x - stage!.x()) / stage!.scaleX();
        const y = (pointerPosition.y - stage!.y()) / stage!.scaleY();

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

        // pushToHistory(newPoint);
        addPoint(newPoint);
      }
    },
    [image, imageDimensions, onPointsChange]
  );

  // Export points to JSON
  const handleExport = useCallback(() => {
    if (!imageDimensions) return;

    const data = {
      imageURL,
      imageDimensions,
      points: points.map(({ x, y, id }) => ({
        x: Math.round(x),
        y: Math.round(y),
        id,
      })),
    };
    const jsonData = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonData], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    // Create a link to trigger download
    const a = document.createElement("a");
    a.href = url;
    a.download = `labeled_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [imageDimensions, imageURL]);

  const getTextConfig = useCallback(() => {
    if (!imageDimensions) return { fontSize: 7, offsetX: 5, offsetY: 5 };
    const imageDiagonal = Math.sqrt(
      Math.pow(imageDimensions.width, 2) + Math.pow(imageDimensions.height, 2)
    );
    const fontSize = Math.max(7, imageDiagonal * 0.01); // 0.8% of diagonal length
    return { fontSize };
  }, [imageDimensions]);

  if (imageError) {
    return <div style={{ color: "red" }}>Error loading image.</div>;
  }

  return (
    <Box
      sx={{ display: "flex", flexDirection: "column", alignItems: "center" }}
    >
      {image && imageDimensions && (
        <>
          <Stage
            width={imageDimensions.width * scale}
            height={imageDimensions.height * scale}
            onClick={handleCanvasClick}
            ref={stageRef}
            style={{
              border: "1px solid gray",
              backgroundColor: "#f0f0f0",
              cursor: "crosshair",
              marginRight: "auto",
            }}
            scaleX={scale}
            scaleY={scale}
          >
            <Layer>
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
                    radius={getScaledRadius()}
                    fill={color}
                    opacity={opacity / 100}
                  />
                  <Text
                    x={point.x + 4}
                    y={point.y - 11}
                    text={(index + 1).toString()}
                    fontSize={getTextConfig().fontSize}
                    fill={color}
                    align="left"
                    verticalAlign="middle"
                    opacity={opacity / 100}
                  />
                </React.Fragment>
              ))}
            </Layer>
          </Stage>

          {points.length > 0 && (
            <Button
              variant="contained"
              color="primary"
              onClick={handleExport}
              sx={{ alignSelf: "flex-start", marginTop: "10px" }} // Align left below the canvas
            >
              Export Data as JSON
            </Button>
          )}
        </>
      )}
    </Box>
  );
};

export default ImageLabeler;
