import { BoundingBox } from "../types/Image";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Trash2, MousePointer2 } from "lucide-react";

interface BoxManagerProps {
  boxes: BoundingBox[];
  selectedBoxId: number | null;
  onSelectBox: (boxId: number) => void;
  onDeleteBox: (boxId: number) => void;
  disabled?: boolean;
}

export function BoxManager({
  boxes,
  selectedBoxId,
  onSelectBox,
  onDeleteBox,
  disabled = false,
}: BoxManagerProps) {
  if (boxes.length === 0) {
    return (
      <div className="p-3 bg-zinc-800 rounded-lg">
        <p className="text-sm text-zinc-400 text-center">
          No specimens detected. Click "Detect" to find specimens in the image.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 p-3 bg-zinc-800 rounded-lg">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-zinc-200">
          Specimens ({boxes.length})
        </span>
      </div>

      <ScrollArea className="max-h-48">
        <div className="flex flex-col gap-1">
          {boxes.map((box, index) => (
            <div
              key={box.id}
              className={`flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                selectedBoxId === box.id
                  ? "bg-blue-600/30 border border-blue-500"
                  : "bg-zinc-700/50 hover:bg-zinc-700 border border-transparent"
              }`}
              onClick={() => !disabled && onSelectBox(box.id)}
            >
              <div className="flex items-center gap-2">
                <div
                  className={`w-6 h-6 rounded flex items-center justify-center text-xs font-bold ${
                    selectedBoxId === box.id
                      ? "bg-blue-500 text-white"
                      : "bg-zinc-600 text-zinc-300"
                  }`}
                >
                  {index + 1}
                </div>
                <div className="flex flex-col">
                  <span className="text-sm text-zinc-200">
                    Specimen {index + 1}
                  </span>
                  <span className="text-xs text-zinc-400">
                    {box.landmarks.length} landmarks
                    {box.confidence !== undefined && (
                      <> · {(box.confidence * 100).toFixed(0)}% conf</>
                    )}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-1">
                {selectedBoxId === box.id && (
                  <MousePointer2 className="w-4 h-4 text-blue-400" />
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-zinc-400 hover:text-red-400 hover:bg-red-400/10"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteBox(box.id);
                  }}
                  disabled={disabled}
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      <p className="text-xs text-zinc-500 mt-1">
        Click a specimen to select it for annotation
      </p>
    </div>
  );
}

export default BoxManager;
