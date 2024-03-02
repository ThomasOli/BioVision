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
  width: number;
  height:number;
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
    height,
    width,
    displayToolbar,
    
  } = props;


  /**
   * State initialization
   */
  
  const [canvas, setCanvas] = useState(new fabric.Canvas(""))
 
  const [backgroundCanvas, setBackgroundCanvas] = useState(
    new fabric.StaticCanvas("")
  )

 const {

    saveState,
    undo,
    redo,
    canUndo,
    canRedo,
    resetState,
  } = useCanvasState();

  /**
   * Initialize canvases on component mount
   * NB: Remount component by changing its key instead of defining deps
   */
  useEffect(() => {
    
    const c = new fabric.Canvas("canvas", {
      enableRetinaScaling: false,
    })
    

    const imgC = new fabric.StaticCanvas("backgroundimage-canvas", {
      enableRetinaScaling: false,
     
    })
    setCanvas(c)
    setBackgroundCanvas(imgC)

    
    // return () => {
    //   if (canvas && backgroundCanvas) {
    //    canvas.dispose();
    //    backgroundCanvas.dispose();
    //   }
    // };
  }, [backgroundImageURL])


  // useEffect(() => {
  //   if (!isEqual(initialState, initialDrawing)) {
  //     canvas.loadFromJSON(initialDrawing, () => {
  //       canvas.renderAll()
  //       resetState(initialDrawing)
  //     })
  //   }
  // }, [canvas, initialDrawing, initialState, resetState])

  /**
   * Update background image
   */
  // useEffect(() => {
  //   if (backgroundCanvas && backgroundImageURL) {
  //       //@ts-ignore 
  //       fabric.FabricImage.fromURL(backgroundImageURL, (img) => {
  //         // Optionally scale image to fit the canvas
          
  //         img.set({
  //           originX: "left",
  //           originY: "top",
  //           selectable: false, // make the background image unselectable
  //         });
  
  //         backgroundCanvas.add(img);
  //         backgroundCanvas.renderAll();
  //       });
  //     }
  //   }, [backgroundCanvas, backgroundImageURL]);
  
  useEffect(() => {
    if (backgroundImageURL && backgroundCanvas) {
        fabric.Image.fromURL(backgroundImageURL, function(img){
        backgroundCanvas.setBackgroundImage(img, backgroundCanvas.renderAll.bind(backgroundCanvas),{})
        
        })
    }
}, [backgroundImageURL, backgroundCanvas]);




  // useEffect(() => {
  //   if (backgroundImageURL && backgroundCanvas ) {
  //     var bgImage = new Image();
  //     bgImage.onload = function () {
  //         backgroundCanvas.getContext().drawImage(bgImage, 0, 0, 100, 100);
  //         backgroundCanvas.renderAll();
        
  //     };
  //     bgImage.src = backgroundImageURL;
  //   }
  // }, [backgroundCanvas, backgroundImageURL]);

    //   useEffect(() => {
  //     if (shouldReloadCanvas) {
  //       canvas.loadFromJSON(currentState, () => {})
  //   }
  // }, [canvas, shouldReloadCanvas, currentState])


  useEffect(() => {
    
    if (canvas && width!=0) {
      // Initialize and configure the selected tool
      
      const selectedTool = new PointTool(canvas) as FabricTool;
      const cleanupToolEvents = selectedTool.configureCanvas({
        fillColor: fillColor,
        strokeWidth: strokeWidth,
        strokeColor: strokeColor,
      });
     
      // Add event listener for mouse up event
      canvas.on("mouse:up", () => {
        saveState(canvas.toJSON());
      });
     
      // Cleanup function to remove event listeners and any other cleanup needed
      return () => {
        cleanupToolEvents();
        canvas.off("mouse:up", () => {})
        canvas.off("mouse:dblclick", () => {})
      };
    }
  }, [strokeWidth, strokeColor, fillColor, drawingMode, saveState, canvas]); // Update dependencies as needed
  
  
  
  /**
   * Render canvas w/ toolbar
   */
 return (
  <>

    <div style={{ position: "relative", height: "100%", width: "100%" }}>
      <div
        style={{
          position: "absolute",
          zIndex: 0,
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
         
        }}
      >
        <canvas
        id = "backgroundimage-canvas"
        width={width}
        height={height}
        style={{ border: "lightgrey 1px solid",  objectFit: "contain", maxWidth: "100%", maxHeight: "100%" }}
        >
        </canvas>
        {/* <img
        style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
        src = {backgroundImageURL}>
        
        </img> */}
      </div>
      <div
        style={{
          position: "absolute",
          zIndex: 10,
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
        }}
      >
        <canvas
          id="canvas"
          width={width}
          height={height}
          style={{ border: "lightgrey 1px solid",  objectFit: "contain", maxWidth: "100%", maxHeight: "100%" }}
        />

       
      </div>
      {/* {displayToolbar && (
        <CanvasToolbar
          topPosition={height}
          leftPosition={width}
          canUndo={canUndo}
          canRedo={canRedo}
          undoCallback={undo}
          redoCallback={redo}
          resetCallback={() => {
          }}
        />
      )} */}
    </div>

    </>
  );
};

export default DrawableCanvas; 