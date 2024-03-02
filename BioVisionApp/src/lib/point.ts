import { fabric } from "fabric";
import FabricTool, { ConfigureCanvasProps } from "./fabrictool"

class PointTool extends FabricTool {
  isMouseDown: boolean = false
  fillColor: string = "#000000"
  strokeWidth: number = 10
  strokeColor: string = "#000000"
  currentCircle: fabric.Circle = new fabric.Circle()
  currentStartX: number = 0
  currentStartY: number = 0
  displayRadius: number = 1

  constructor(canvas: fabric.Canvas) {
    super(canvas);
    this.onMouseDown = this.onMouseDown.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);
    this.onMouseOut = this.onMouseOut.bind(this);
  }

  configureCanvas({
    strokeWidth,
    strokeColor,
    fillColor,
    
  }: ConfigureCanvasProps): () => void {
    console.log(this._canvas)

    this._canvas.isDrawingMode = false
    this._canvas.selection = false
    this._canvas.forEachObject((o) => (o.selectable = o.evented = false))

    this.strokeWidth = strokeWidth
    this.strokeColor = strokeColor
    this.fillColor = "#000000"

    this._canvas.on("mouse:down", (e: any) => this.onMouseDown(e))
    this._canvas.on("mouse:move", (e: any) => this.onMouseMove(e))
    this._canvas.on("mouse:up", (e: any) => this.onMouseUp(e))
    this._canvas.on("mouse:out", (e: any) => this.onMouseOut(e))
    return () => {
      //@ts-ignore
      this._canvas.off("mouse:down")
      //@ts-ignore
      this._canvas.off("mouse:move")
      //@ts-ignore
      this._canvas.off("mouse:up")
      //@ts-ignore
      this._canvas.off("mouse:out")
    }
  
  }
  
  onMouseDown(o: any) {
    console.log("clicked")
    let canvas = this._canvas
    let _clicked = o.e["button"]
    this.isMouseDown = true
    let pointer = canvas.getPointer(o.e)
    this.currentStartX = pointer.x - (this.strokeWidth / 2.)
    this.currentStartY = pointer.y //- (this._minRadius + this.strokeWidth)
    this.currentCircle = new fabric.Circle({
      left: this.currentStartX,
      top: this.currentStartY,
      originX: "left",
      originY: "center",
      strokeWidth: this.strokeWidth,
      stroke: this.strokeColor,
      fill: this.fillColor,
      selectable: false,
      evented: false,
      radius: 2,
    })
    if (_clicked === 0) {
      

      canvas.add(this.currentCircle)
      this._canvas.renderAll(); 
    }
  }

  onMouseMove(o: any) {
        console.log("clicked")

    if (!this.isMouseDown) return
    let canvas = this._canvas
    this.currentCircle.setCoords()
    canvas.renderAll()
  }

  onMouseUp(o: any) {
    console.log("clicked")

    this.isMouseDown = false
  }

  onMouseOut(o: any) {
    console.log("clicked")

    this.isMouseDown = false
  }

}

export default PointTool
