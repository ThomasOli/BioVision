// src/Components/ImageLabeler.tsx
import React, {
  useRef,
  useState,
  useCallback,
  useEffect,
  useContext,
} from "react";
import { Stage, Layer, Image as KonvaImage, Circle } from "react-konva";
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
  initialPoints: Point[];
  onPointsChange: (points: Point[]) => void;
  color: string;
  opacity: number;
}

const ImageLabeler: React.FC<ImageLabelerProps> = ({
  imageURL,
  initialPoints,
  onPointsChange,
  color,
  opacity,
}) => {
  const [points, setPoints] = useState<Point[]>(initialPoints || []);

  // ----------------------------------------------------

  const { setPoints2 } = useContext(UndoRedoClearContext);

  const [history, setHistory] = useState<Point[]>([]); // Store past states (e.g., canvas data URL)
  const [future, setFuture] = useState<Point[]>([]); // Store future states for redo
  const [usedClear, setUsedClear] = useState(false);

  const pushToHistory = useCallback((newState: Point) => {
    setHistory((prev) => [...prev, newState]);
    setFuture([]); // Clear future when a new action is made
  }, []);

  const undo = useCallback(() => {
    if (usedClear) {
      console.log("undo cler");
      setFuture([]);

      setPoints(history);

      setUsedClear(false);
    } else {
      const newRedoPoint = history[history.length - 1];
      // console.log("  newRedoPoint is ")
      // console.log(newRedoPoint)
      const newHistory = [...history];
      newHistory.splice(-1, 1);
      setHistory(newHistory);

      const newFuture = [...future, newRedoPoint];
      setFuture(newFuture);

      const newPoints = [...points]; // Create a copy of the array
      newPoints.splice(-1, 1); // Remove the last element using splice
      setPoints(newPoints);

      // console.log("the new points are", newPoints);
      // console.log("the new history is", newHistory);
      // console.log("the new future is", newFuture);
    }
  }, [points, future, history]);

  const redo = useCallback(() => {
    if (future.length === 0) return; // Ensure there is something to redo

    const newUndoPoint = future[future.length - 1]; // Get the last element from future
    // console.log("newRedoPoint is", newRedoPoint);

    const newFuture = [...future]; // Copy the future array
    newFuture.splice(-1, 1); // Remove the last element from future
    setFuture(newFuture); // Update future state

    const newHistory = [...history, newUndoPoint]; // Add the redo point back to history
    setHistory(newHistory); // Update history state

    const newPoints = [...points, newUndoPoint]; // Add the redo point back to points
    setPoints(newPoints); // Update points state

    // console.log("the new points are", newPoints);
    // console.log("the new history is", newHistory);
    // console.log("the new future is", newFuture);
  }, [points, future, history]);

  const clear = useCallback(() => {
    // Save the current points to history so that undo can bring them back
    const newHistory = [...history]; // Add the current points to history
    newHistory.concat(points);
    setHistory(newHistory);
    // console.log("history is ", history)

    // Clear the points by setting it to an empty array
    setPoints([]);

    // Optionally reset future, since clearing might represent a new action that prevents redo
    setFuture([]);

    // console.log("Points cleared.");
    // console.log("History updated:", newHistory);
    setUsedClear(true);
  }, [points, future, history]);

  // ----------------------------------------------------

  const [image, imageDimensions, imageError] = useImageLoader(imageURL);
  const stageRef = useRef<any>(null);
  const [scale, setScale] = useState(1); // State for scaling

  // Update points when initialPoints or imageURL changes
  useEffect(() => {
    setPoints(initialPoints || []);
  }, [initialPoints, imageURL]);

  // Adjust the scale based on available screen space
  useEffect(() => {
    if (imageDimensions) {
      const availableWidth = window.innerWidth * 0.6; // Reserve 60% of screen width
      const availableHeight = window.innerHeight * 0.6; // Reserve 60% of screen height

      const widthScale = availableWidth / imageDimensions.width;
      const heightScale = availableHeight / imageDimensions.height;
      setScale(Math.min(widthScale, heightScale));
    }
  }, [imageDimensions]);

  // Handle canvas click to add a point
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions) return;

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
        const updatedPoints = [...points, newPoint];
        setPoints(updatedPoints);
        pushToHistory(newPoint);
        onPointsChange(updatedPoints);
        // console.log("history is: ", history)
        setPoints2(0, newPoint);
      }
    },
    [image, imageDimensions, points, onPointsChange]
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
  }, [points, imageDimensions, imageURL]);

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
              {points.map((point) => (
                <React.Fragment key={point.id}>
                  <Circle
                    x={point.x}
                    y={point.y}
                    radius={3}
                    fill={color}
                    opacity={opacity / 100} // Convert percentage to decimal
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
