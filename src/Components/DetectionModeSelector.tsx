import { useEffect, useState } from "react";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import { Slider } from "./ui/slider";

export type DetectionMode = "single" | "multi";

interface DetectionModeSelectorProps {
  mode: DetectionMode;
  onModeChange: (mode: DetectionMode) => void;
  confThreshold: number;
  onConfThresholdChange: (value: number) => void;
  disabled?: boolean;
}

export function DetectionModeSelector({
  mode,
  onModeChange,
  confThreshold,
  onConfThresholdChange,
  disabled = false,
}: DetectionModeSelectorProps) {
  const [detectionInfo, setDetectionInfo] = useState<{
    available: boolean;
    primary_method?: string;
  } | null>(null);

  useEffect(() => {
    // Check detection availability (OpenCV is always available)
    window.api.checkYolo().then((result) => {
      setDetectionInfo({
        available: result.available,
        primary_method: result.primary_method || "opencv",
      });
    }).catch(() => {
      // OpenCV detection is always available
      setDetectionInfo({ available: true, primary_method: "opencv" });
    });
  }, []);

  const isMultiMode = mode === "multi";
  const isAvailable = detectionInfo?.available ?? true; // OpenCV always available

  return (
    <div className="flex flex-col gap-3 p-3 bg-zinc-800 rounded-lg">
      <div className="flex items-center justify-between">
        <Label htmlFor="detection-mode" className="text-sm font-medium text-zinc-200">
          Multi-Specimen Mode
        </Label>
        <Switch
          id="detection-mode"
          checked={isMultiMode}
          onCheckedChange={(checked) => onModeChange(checked ? "multi" : "single")}
          disabled={disabled || !isAvailable}
        />
      </div>

      {isMultiMode && (
        <div className="flex flex-col gap-2 pt-2 border-t border-zinc-700">
          <p className="text-xs text-zinc-400">
            Uses contour detection to find multiple specimens
          </p>
          <div className="flex items-center justify-between">
            <Label className="text-xs text-zinc-400">
              Min Area: {(confThreshold * 100).toFixed(0)}%
            </Label>
          </div>
          <Slider
            value={[confThreshold]}
            onValueChange={([value]) => onConfThresholdChange(value)}
            min={0.01}
            max={0.2}
            step={0.01}
            disabled={disabled}
            className="w-full"
          />
          <p className="text-xs text-zinc-500">
            Minimum specimen size as % of image area
          </p>
        </div>
      )}
    </div>
  );
}

export default DetectionModeSelector;
