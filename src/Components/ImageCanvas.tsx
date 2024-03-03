import React, { useRef, useEffect } from 'react';
import { fabric } from 'fabric';

interface ImageCanvasProps {
  imageUrl: string;
  color: string;
  opacity: number;
}

const ImageCanvas: React.FC<ImageCanvasProps> = ({ imageUrl, color, opacity }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = new fabric.Canvas(canvasRef.current, {
      selection: false // Disable object selection
    });

    // Load image onto the canvas
    fabric.Image.fromURL(imageUrl, function (img) {
      if (img) {
        canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
          scaleX: canvas.getWidth() / img.width!,
          scaleY: canvas.getHeight() / img.height!
        });

        // Add dots on top of the image
        canvas.on('mouse:down', function (options) {
          const pointer = canvas.getPointer(options.e);
          const x = pointer.x;
          const y = pointer.y;

          // Create a dot
          const dot = new fabric.Circle({
            radius: 5,
            fill: color,
            left: x,
            top: y,
            selectable: false, // Disable selection of the dot
            opacity: opacity,
          });

          canvas.add(dot);
        });
      }
    });

    return () => {
      canvas.dispose(); // Cleanup Fabric canvas
    };
  }, [imageUrl]);

  return <canvas ref={canvasRef} />;
};

export default ImageCanvas;
