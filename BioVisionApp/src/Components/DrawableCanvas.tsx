import React, { useEffect, useState } from "react";
import * as fabric from "fabric";
import { isEqual } from "lodash";

import CanvasToolbar from "../Canvas/CanvasToolbar";
import { useCanvasState } from "./DrawableCanvasState";
import { tools, FabricTool } from "../lib";

/**
 * Arguments previously received from the Python side are now props
 */
interface DrawableCanvasProps {
  fillColor: string;
  strokeWidth: number;
  strokeColor: string;
  backgroundColor: string;
  backgroundImageURL: string;
  realtimeUpdateStreamlit: boolean;
  canvasWidth: number;
  canvasHeight: number;
  drawingMode: string;
  initialDrawing: any;
  displayToolbar: boolean;
  displayRadius: number;
}

const DrawableCanvas: React.FC<DrawableCanvasProps> = ({
  fillColor,
  strokeWidth,
  strokeColor,
  backgroundColor,
  backgroundImageURL,
  canvasWidth,
  canvasHeight,
  drawingMode,
  initialDrawing,
  displayToolbar,
  displayRadius,
}) => {
  const [canvas, setCanvas] = useState<fabric.Canvas | null>(null);
  const [backgroundCanvas, setBackgroundCanvas] = useState<fabric.StaticCanvas | null>(null);

  const {
    canvasState: { currentState, initialState },
    saveState,
    resetState,
  } = useCanvasState();

  // Initialize canvases
  useEffect(() => {
    const c = new fabric.Canvas("canvas", {
      enableRetinaScaling: false,
      backgroundColor,
      width: canvasWidth,
      height: canvasHeight,
    });
    const bgC = new fabric.StaticCanvas("backgroundimage-canvas", {
      enableRetinaScaling: false,
    });
    setCanvas(c);
    setBackgroundCanvas(bgC);
  }, [backgroundColor, canvasWidth, canvasHeight]);

  // Load initial drawing
  useEffect(() => {
    if (canvas && !isEqual(initialState, initialDrawing)) {
      canvas.loadFromJSON(initialDrawing, () => {
        canvas.renderAll();
        resetState(initialDrawing);
      });
    }
  }, [canvas, initialDrawing, initialState, resetState]);

  // Update background image
  useEffect(() => {
    if (backgroundCanvas && backgroundImageURL) {
      fabric.Image.fromURL(backgroundImageURL, (img) => {
        if (img.width && img.height) { // Ensure img properties are available
          const scaleX = backgroundCanvas.width ? backgroundCanvas.width / img.width : 1;
          const scaleY = backgroundCanvas.height ? backgroundCanvas.height / img.height : 1;
  
          backgroundCanvas.setBackgroundImage(img, backgroundCanvas.renderAll.bind(backgroundCanvas), {
            scaleX,
            scaleY,
          });
        }
      });
    }
  }, [backgroundCanvas, backgroundImageURL]);

  // Update tool based on drawingMode
  useEffect(() => {
    if (canvas) {
      const selectedTool = new tools[drawingMode](canvas) as FabricTool;
      const cleanupToolEvents = selectedTool.configureCanvas({
        fillColor,
        strokeWidth,
        strokeColor,
        displayRadius,
      });

      canvas.on("mouse:up", () => {
        saveState(canvas.toJSON());
      });

      return () => cleanupToolEvents();
    }
  }, [canvas, drawingMode, fillColor, strokeWidth, strokeColor, displayRadius, saveState]);

  return (
    <div style={{ position: "relative" }}>
      <canvas
        id="backgroundimage-canvas"
        width={canvasWidth}
        height={canvasHeight}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          zIndex: 0,
        }}
      />
      <canvas
        id="canvas"
        width={canvasWidth}
        height={canvasHeight}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          zIndex: 10,
          border: "lightgrey 1px solid",
        }}
      />
      {/* {displayToolbar && (
        // Toolbar component might need to be adjusted to work without Streamlit
        <CanvasToolbar />
      )} */}
    </div>
  );
};

export default DrawableCanvas;
