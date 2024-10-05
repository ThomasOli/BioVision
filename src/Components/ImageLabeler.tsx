// components/ImageLabeler.tsx
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
}

const ImageLabeler: React.FC<ImageLabelerProps> = ({ imageURL, initialPoints, onPointsChange }) => {
  const [points, setPoints] = useState<Point[]>(initialPoints || []);
  const [image, imageDimensions] = useImageLoader(imageURL);
  const stageRef = useRef<any>(null);

  // Zoom and Pan State
  const [scale, setScale] = useState<number>(1);
  const [position, setPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Handle canvas click to add a point
  const handleCanvasClick = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!image) return;

      const stage = e.target.getStage();
      const pointerPosition = stage?.getPointerPosition();
      if (pointerPosition) {
        const newPoint: Point = {
          x: (pointerPosition.x - position.x) / scale,
          y: (pointerPosition.y - position.y) / scale,
          id: Date.now(),
        };
        const updatedPoints = [...points, newPoint];
        setPoints(updatedPoints);
        onPointsChange(updatedPoints);
      }
    },
    [image, points, onPointsChange, scale, position]
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
      const updatedPoints = points.map((p) => (p.id === id ? { ...p, x: x / scale, y: y / scale } : p));
      setPoints(updatedPoints);
      onPointsChange(updatedPoints);
    },
    [points, onPointsChange, scale]
  );

  // Handle Zoom
  const handleWheel = useCallback(
    (e: KonvaEventObject<WheelEvent>) => {
      e.evt.preventDefault();
      const stage = e.target.getStage();
      if (!stage) return;

      const oldScale = stage.scaleX();
      const pointer = stage.getPointerPosition();
      if (!pointer) return;

      const scaleBy = 1.05;
      const direction = e.evt.deltaY > 0 ? -1 : 1;
      const newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;

      setScale(newScale);

      const mousePointTo = {
        x: (pointer.x - position.x) / oldScale,
        y: (pointer.y - position.y) / oldScale,
      };

      const newPos = {
        x: pointer.x - mousePointTo.x * newScale,
        y: pointer.y - mousePointTo.y * newScale,
      };
      setPosition(newPos);
    },
    [position]
  );

  return (
    <div>
      {image && imageDimensions && (
        <Stage
          width={800}
          height={600}
          onClick={handleCanvasClick}
          onWheel={handleWheel}
          scaleX={scale}
          scaleY={scale}
          x={position.x}
          y={position.y}
          draggable
          ref={stageRef}
          style={{ border: '1px solid gray', marginTop: '10px', backgroundColor: '#f0f0f0' }}
        >
          <Layer>
            <KonvaImage image={image} width={imageDimensions.width} height={imageDimensions.height} />
            {points.map((point) => (
              <React.Fragment key={point.id}>
                <Circle
                  x={point.x * scale + position.x}
                  y={point.y * scale + position.y}
                  radius={5 * scale}
                  fill="red"
                  draggable
                  onDragEnd={(e) => handlePointDragEnd(e, point.id)}
                  onContextMenu={(e) => handlePointRightClick(e, point.id)}
                />
                <Text
                  x={point.x * scale + position.x + 10}
                  y={point.y * scale + position.y - 10}
                  text={`(${Math.round(point.x)}, ${Math.round(point.y)})`}
                  fontSize={12 * scale}
                  fill="black"
                />
              </React.Fragment>
            ))}
          </Layer>
        </Stage>
      )}
      {points.length > 0 && (
        <button onClick={handleExport} style={{ marginTop: '10px' }}>
          Export Data as JSON
        </button>
      )}
    </div>
  );
};

export default ImageLabeler;
