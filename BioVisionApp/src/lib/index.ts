import CircleTool from "./circle"
import FabricTool from "./fabrictool"
import FreedrawTool from "./freedraw"

import TransformTool from "./transform"
import PointTool from "./point"

// TODO: Should make TS happy on the Map of selectedTool --> FabricTool
const tools: any = {
  circle: CircleTool,
  freedraw: FreedrawTool,
  transform: TransformTool,
  point: PointTool
}

export { tools, FabricTool }
