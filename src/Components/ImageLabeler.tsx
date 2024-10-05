// src/Components/ImageLabeler.tsx
import React, { useRef, useState, useCallback } from 'react';
import { Stage, Layer, Image as KonvaImage, Circle, Text } from 'react-konva';
import useImageLoader from '../hooks/useImageLoader';
import { KonvaEventObject } from 'konva/lib/Node';

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

const ImageLabeler: React.FC<ImageLabelerProps> = ({ imageURL, initialPoints, onPointsChange, color, opacity }) => {
  const [points, setPoints] = useState<Point[]>(initialPoints || []);
  const [image, imageDimensions] = useImageLoader(imageURL);
  const stageRef = useRef<any>(null);

  // Handle canvas click to add a point
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image) return;

      const stage = e.target.getStage();
      const pointerPosition = stage?.getPointerPosition();
      if (pointerPosition) {
        const newPoint: Point = {
          x: pointerPosition.x,
          y: pointerPosition.y,
          id: Date.now(),
        };
        const updatedPoints = [...points, newPoint];
        setPoints(updatedPoints);
        onPointsChange(updatedPoints);
      }
    },
    [image, points, onPointsChange]
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
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    // Create a link to trigger download
    const a = document.createElement('a');
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
      const updatedPoints = points.map((p) => (p.id === id ? { ...p, x, y } : p));
      setPoints(updatedPoints);
      onPointsChange(updatedPoints);
    },
    [points, onPointsChange]
  );

  return (
    <div>
      {image && imageDimensions && (
        <Stage
          width={800} // Adjust based on your design
          height={600} // Adjust based on your design
          onClick={handleCanvasClick}
          ref={stageRef}
          style={{ border: '1px solid gray', marginTop: '10px', backgroundColor: '#f0f0f0' }}
        >
          <Layer>
            {/* Render the uploaded image */}
            <KonvaImage image={image} width={imageDimensions.width} height={imageDimensions.height} />
            {/* Render the points */}
            {points.map((point) => (
              <React.Fragment key={point.id}>
                <Circle
                  x={point.x}
                  y={point.y}
                  radius={5}
                  fill={color}
                  opacity={opacity / 100}
                  draggable
                  onDragEnd={(e) => handlePointDragEnd(e, point.id)}
                  onContextMenu={(e) => handlePointRightClick(e, point.id)}
                />
                <Text
                  x={point.x + 10}
                  y={point.y - 10}
                  text={`(${Math.round(point.x)}, ${Math.round(point.y)})`}
                  fontSize={12}
                  fill="black"
                />
              </React.Fragment>
            ))}
          </Layer>
        </Stage>
      )}
      {points.length > 0 && (
        <button onClick={handleExport} style={{ marginTop: '10px', padding: '10px 20px' }}>
          Export Data as JSON
        </button>
      )}
    </div>
  );
};

export default ImageLabeler;
