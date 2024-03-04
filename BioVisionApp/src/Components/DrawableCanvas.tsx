import React, { FunctionComponent, useEffect, useRef, useState } from "react"

import { fabric } from 'fabric';

import CanvasToolbar from "../Canvas/CanvasToolbar"

import { useCanvasState } from "./DrawableCanvasState"
import { tools, FabricTool } from "../lib"
import PointTool from "../lib/point";

/**
 * Arguments Streamlit receives from the Python side
 */
interface CanvasProps {
  fillColor: string;
  strokeWidth: number;
  strokeColor: string;
  backgroundColor: string;
  backgroundImageURL: string;
  drawingMode: string;
  initialDrawing: any; // Adjust the type according to your data structure
  displayToolbar: boolean;
}

/**
 * Define logic for the canvas area
 */
const DrawableCanvas = (props: CanvasProps) => {
  const {
    backgroundImageURL,
    drawingMode,
    fillColor,
    strokeWidth,
    strokeColor,
    initialDrawing,
    displayToolbar,
    
  } = props;


  /**
   * State initialization
   */
//  const {

//     saveState,
//     undo,
//     redo,
//     canUndo,
//     canRedo,
//     resetState,
//   } = useCanvasState();

  /**
   * Initialize canvases on component mount
   * NB: Remount component by changing its key instead of defining deps
   */
  useEffect(() => {
    
    const c = new fabric.Canvas("canvas", {
      enableRetinaScaling: false,
      selection: false,
    })

    fabric.Image.fromURL(backgroundImageURL, function (img) {
      if (img) {
        c.setDimensions({ width: img.getOriginalSize().width, height: img.getOriginalSize().height });
        c.setBackgroundImage(img, c.renderAll.bind(c), {
          
        });

        // Add dots on top of the image
        c.on('mouse:down', function (options) {
          const pointer = c.getPointer(options.e);
          const x = pointer.x;
          const y = pointer.y;

          // Create a dot
          const dot = new fabric.Circle({
            radius: 5,
            fill: fillColor,
            left: x,
            top: y,
            selectable: false, // Disable selection of the dot
          });

          c.add(dot);
        });
      }
    });

    return () => {
      c.dispose(); // Cleanup Fabric canvas
    };
        
    // return () => {
    //   if (canvas && backgroundCanvas) {
    //    canvas.dispose();
    //    backgroundCanvas.dispose();
    //   }
    // };
  }, [backgroundImageURL])


  // useEffect(() => {
    
  //   if (canvas && width!=0) {
  //     // Initialize and configure the selected tool
      
  //     // Add event listener for mouse up event
  //     canvas.on("mouse:up", () => {
  //       saveState(canvas.toJSON());
  //     });
     
  //     // Cleanup function to remove event listeners and any other cleanup needed
  //     return () => {
  //       canvas.off("mouse:up", () => {})
  //       canvas.off("mouse:dblclick", () => {})
  //     };
  //   }
  // }, [strokeWidth, strokeColor, fillColor, drawingMode, saveState, canvas]); // Update dependencies as needed
  
  
  
  /**
   * Render canvas w/ toolbar
   */
 return (
  <>
        <canvas
          id="canvas"
          style={{ border: "lightgrey 1px solid"}}
        />

    </>
  );
};

export default DrawableCanvas; 