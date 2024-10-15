// src/Components/ImageLabeler.tsx
import React, { useRef, useState, useCallback, useEffect, useContext } from 'react';
import { Stage, Layer, Image as KonvaImage, Circle } from 'react-konva';
import useImageLoader from '../hooks/useImageLoader';
import { KonvaEventObject } from 'konva/lib/Node';
import { Button, Box } from '@mui/material';
import { MyContext } from './MyContext';
import { HistoryRounded } from '@mui/icons-material';

interface Point {
  x: number;
  y: number;
  id: number;
}

interface ImageLabelerProps {
  imageURL: string;
  initialPoints: Point[];
  initialHistory: Point[];
  onPointsChange: (points: Point[]) => void;
  color: string;
  opacity: number;
}

const ImageLabeler: React.FC<ImageLabelerProps> = ({
  imageURL,
  initialPoints,
  initialHistory,
  onPointsChange,
  color,
  opacity,
}) => {
  const [points, setPoints] = useState<Point[]>(initialPoints || []);

  // ----------------------------------------------------

  const [history, setHistory] = useState<Point[]>(initialHistory || []); // Store past states (e.g., canvas data URL)
  const [future, setFuture] = useState<Point[]>([]); // Store future states for redo

  const pushToHistory = useCallback((newState: Point) => {
    setHistory((prev) => [...prev, newState]);
    setFuture([]); // Clear future when a new action is made
  }, []);

 const undo = useCallback(() => {
    if (history.length === 0)
      return

    setHistory((prev) => {
      const newFuture = prev[prev.length - 1];
      setFuture((future) => [...future, newFuture]);
      return prev.slice(-1, 1); // Remove the last state from history
    });

    const newPoints = [...points];  // Create a copy of the array
    // console.log("there are " + newPoints.length + " points before undo ") 
    newPoints.splice(-1, 1);       // Remove the last element using splice
    // console.log("there are " + newPoints.length + " points after undo ") 
    setPoints(newPoints);
    // console.log(points) 
    console.log("there are this many points in history: " + history.length)
  }, [points]);

  const redo = useCallback(() => {
    setFuture((prev) => {
      if (prev.length === 0) return prev;

      const newHistory = prev[0];
      setHistory((history) => [...history, newHistory]);
      return prev.slice(1); // Remove the first state from future
    });

    const newPoints = [...points];  // Create a copy of the array
    console.log("there are " + newPoints.length + " points before redo ") 
    newPoints.splice(-1, 1);       // Remove the last element using splice
    console.log("there are " + newPoints.length + " points after redo ") 
    setPoints(newPoints);   
    // console.log(points) 
  }, [points]);

  // ----------------------------------------------------


  const [image, imageDimensions, imageError] = useImageLoader(imageURL);
  const stageRef = useRef<any>(null);
  

  // Zoom state (for standard view zooming, if needed)
  // For this implementation, zooming is handled via magnified view
  // Hence, we can remove or disable the existing zoom controls here

  // Update points when initialPoints or imageURL changes
  useEffect(() => {
    console.log('ImageLabeler: Updating points based on new props.');
    setPoints(initialPoints || []);
  }, [initialPoints, imageURL]);

  // Handle canvas click to add a point
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image || !imageDimensions) return;

      const stage = e.target.getStage();
      const pointerPosition = stage?.getPointerPosition();
      if (pointerPosition) {
        // Convert pointer position to image coordinates
        const x = (pointerPosition.x - stage!.x()) / stage!.scaleX();
        const y = (pointerPosition.y - stage!.y()) / stage!.scaleY();

        // Check if click is within image boundaries
        if (x < 0 || y < 0 || x > imageDimensions.width || y > imageDimensions.height) {
          return; // Click outside image area
        }

        const newPoint: Point = {
          x: x,
          y: y,
          id: Date.now(),
        };
        const updatedPoints = [...points, newPoint];
        setPoints(updatedPoints);
        console.log("there are " + updatedPoints.length + " after click")
        pushToHistory(newPoint)
        onPointsChange(updatedPoints);
        console.log(history)
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

  // Remove a point
  const handlePointRightClick = useCallback(
    (e: KonvaEventObject<PointerEvent>, id: number) => {
      e.evt.preventDefault();
      const updatedPoints = points.filter((point) => point.id !== id);
      setPoints(updatedPoints);
      onPointsChange(updatedPoints);
    },
    [points, onPointsChange]
  );

  // Handle point drag end
  const handlePointDragEnd = useCallback(
    (e: KonvaEventObject<DragEvent>, id: number) => {
      const { x, y } = e.target.position();
      const updatedPoints = points.map((p) =>
        p.id === id ? { ...p, x, y } : p
      );
      setPoints(updatedPoints);
      onPointsChange(updatedPoints);
    },
    [points, onPointsChange]
  );

  if (imageError) {
    return <div style={{ color: 'red' }}>Error loading image.</div>;
  }

  return (
    <Box sx={{ display: 'inline-block', position: 'relative' }}>
      {image && imageDimensions && (
        <Stage
          width={imageDimensions.width}
          height={imageDimensions.height}
          onClick={handleCanvasClick}
          ref={stageRef}
          style={{
            border: '1px solid gray',
            backgroundColor: '#f0f0f0',
            cursor: 'crosshair',
          }}
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
                  draggable
                  onDragEnd={(e) => handlePointDragEnd(e, point.id)}
                  onContextMenu={(e) => handlePointRightClick(e, point.id)}
                />
              </React.Fragment>
            ))}
          </Layer>
        </Stage>
      )}
      {points.length > 0 && (
        <Button
        variant="contained"
        color="primary"
        onClick={handleExport}
        sx={{ marginTop: '10px' }}
      >
        Export Data as JSON
      </Button>
      )}
      {history.length > 0 && (
        <Button
          variant="contained"
          color="primary"
          onClick={undo}
          sx={{ marginTop: '10px' }}
        >
          undo
        </Button>
      )}
        
    </Box>
  );
};

export default ImageLabeler;
