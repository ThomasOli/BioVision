import { useCallback, useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Switch } from "./ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";
import { selectSam2Enabled } from "../state/hardwareSlice";

export type DetectionMode = "manual" | "auto";
export type DetectionPreset = "balanced" | "precision" | "recall" | "single_object";

interface DetectionModeSelectorProps {
  mode: DetectionMode;
  onModeChange: (mode: DetectionMode) => void;
  detectionPreset?: DetectionPreset;
  onDetectionPresetChange?: (preset: DetectionPreset) => void;
  disabled?: boolean;
  className?: string;
  onClassNameChange?: (name: string) => void;
  samEnabled?: boolean;
  onSamEnabledChange?: (enabled: boolean) => void;
}

export function DetectionModeSelector({
  mode,
  onModeChange,
  detectionPreset = "balanced",
  onDetectionPresetChange,
  disabled = false,
  className = "",
  onClassNameChange,
  samEnabled = false,
  onSamEnabledChange,
}: DetectionModeSelectorProps) {
  // Redux-backed hardware capability (computed from system probe at startup)
  const reduxSam2Enabled = useSelector(selectSam2Enabled);

  const [capabilityInfo, setCapabilityInfo] = useState<{
    mode: string;
    gpu: boolean;
    yolo_ready: boolean;
    sam2_ready: boolean;
    yolo_failed: boolean;
    sam2_failed: boolean;
    yolo_error?: string | null;
  } | null>(null);

  const setCapabilityInfoIfChanged = useCallback((next: {
    mode: string;
    gpu: boolean;
    yolo_ready: boolean;
    sam2_ready: boolean;
    yolo_failed: boolean;
    sam2_failed: boolean;
    yolo_error?: string | null;
  }) => {
    setCapabilityInfo((prev) => {
      if (
        prev &&
        prev.mode === next.mode &&
        prev.gpu === next.gpu &&
        prev.yolo_ready === next.yolo_ready &&
        prev.sam2_ready === next.sam2_ready &&
        prev.yolo_failed === next.yolo_failed &&
        prev.sam2_failed === next.sam2_failed &&
        (prev.yolo_error ?? null) === (next.yolo_error ?? null)
      ) {
        return prev;
      }
      return next;
    });
  }, []);

  const refreshCapabilities = useCallback(() => {
    window.api.checkSuperAnnotator().then((result) => {
      setCapabilityInfoIfChanged({
        mode: result.mode,
        gpu: result.gpu,
        yolo_ready: result.yolo_ready,
        sam2_ready: result.sam2_ready,
        yolo_failed: !!result.yolo_failed,
        sam2_failed: !!result.sam2_failed,
        yolo_error: result.yolo_error,
      });
    }).catch(() => {
      setCapabilityInfoIfChanged({
        mode: "classic_fallback",
        gpu: false,
        yolo_ready: false,
        sam2_ready: false,
        yolo_failed: false,
        sam2_failed: false,
      });
    });
  }, [setCapabilityInfoIfChanged]);

  useEffect(() => {
    refreshCapabilities();
  }, [refreshCapabilities]);

  // Poll capabilities while YOLO is pending so the badge updates after auto-init
  useEffect(() => {
    if (
      !capabilityInfo ||
      capabilityInfo.yolo_ready ||
      capabilityInfo.yolo_failed ||
      capabilityInfo.mode === "classic_fallback"
    ) return;
    const id = setInterval(refreshCapabilities, 4000);
    return () => clearInterval(id);
  }, [capabilityInfo, refreshCapabilities]);

  const modeOptions: { value: DetectionMode; label: string; desc: string }[] = [
    { value: "manual", label: "Manual", desc: "Draw bounding boxes manually" },
    { value: "auto", label: "Auto (AI)", desc: "AI detection with YOLO-World zero-shot fallback or the session OBB detector" },
  ];

  const sam2Available = capabilityInfo?.mode === "auto_high_performance";
  const isClassicFallback = capabilityInfo?.mode === "classic_fallback";
  const classicFallbackReason = capabilityInfo
    ? capabilityInfo.yolo_failed
      ? `Python error: ${capabilityInfo.yolo_error ?? "initialization failed"}`
      : "Requires >1 GB free RAM or GPU"
    : null;

  // SAM2 is interactable only when hardware AND runtime capability both confirm it
  const sam2Disabled = disabled || !reduxSam2Enabled || (capabilityInfo !== null && !sam2Available);

  return (
    <div className="flex flex-col gap-3 p-3 bg-zinc-800 rounded-lg">
      {/* 2-option segmented control */}
      <div className="flex rounded-lg bg-zinc-900 p-0.5">
        {modeOptions.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onModeChange(opt.value)}
            disabled={disabled}
            className={`flex-1 px-2 py-1.5 text-xs font-medium rounded-md transition-all ${
              mode === opt.value
                ? "bg-zinc-700 text-zinc-100 shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <p className="text-xs text-zinc-500">
        {modeOptions.find((o) => o.value === mode)?.desc}
      </p>

      {/* Manual mode controls */}
      {mode === "manual" && (
        <div className="flex flex-col gap-2 pt-2 border-t border-zinc-700">
          <p className="text-xs text-zinc-500">
            Drag to draw a box. Select a box to drag, resize, or rotate it — use the rotation handle above the box to create oriented (OBB) boxes.
          </p>
        </div>
      )}

      {/* Auto (AI) mode controls */}
      {mode === "auto" && (
        <div className="flex flex-col gap-3 pt-2 border-t border-zinc-700">
          <div className="flex flex-col gap-1.5">
            <Label className="text-xs text-zinc-400">
              Detection goal
            </Label>
            <select
              value={detectionPreset}
              onChange={(e) => onDetectionPresetChange?.(e.target.value as DetectionPreset)}
              disabled={disabled}
              className="h-8 rounded-md border border-zinc-700 bg-zinc-900 px-2 text-xs text-white"
            >
              <option value="balanced">Balanced</option>
              <option value="precision">Precision (fewer false positives)</option>
              <option value="recall">Recall (find more objects)</option>
              <option value="single_object">Single object focus</option>
            </select>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label className="text-xs text-zinc-400">
              Object class / prompt
            </Label>
            <Input
              value={className}
              onChange={(e) => onClassNameChange?.(e.target.value)}
              placeholder="e.g. Fish, Butterfly, Leaf..."
              className="h-8 text-xs text-white placeholder:text-zinc-400 bg-zinc-900 border-zinc-700"
              disabled={disabled}
            />
          </div>

          {/* SAM2 refinement toggle — gated by both hardware probe and runtime capability */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex flex-col gap-0.5">
              <Label className="text-xs text-zinc-400">
                SAM2 mask refinement
              </Label>
              <p className="text-[10px] text-zinc-500">
                {sam2Available
                  ? "Precise pixel masks for crowded scenes"
                  : reduxSam2Enabled
                    ? "Not available in current detection mode"
                    : "Requires GPU + 8 GB RAM"}
              </p>
            </div>
            <Tooltip>
              <TooltipTrigger asChild>
                <span>
                  <Switch
                    checked={samEnabled}
                    onCheckedChange={(checked) => onSamEnabledChange?.(checked)}
                    disabled={sam2Disabled}
                  />
                </span>
              </TooltipTrigger>
              {sam2Disabled && (
                <TooltipContent side="left">
                  {!reduxSam2Enabled
                    ? "SAM2 requires a GPU (CUDA or Apple MPS) and at least 8 GB of system RAM."
                    : "SAM2 is only available in high-performance mode (GPU detected but running in lite mode)."}
                </TooltipContent>
              )}
            </Tooltip>
          </div>

          {/* Capability badges */}
          {capabilityInfo && (
            <div className="flex flex-wrap items-center gap-1.5 pt-1">
              <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium ${
                capabilityInfo.yolo_ready
                  ? "bg-green-500/20 text-green-400"
                  : capabilityInfo.yolo_failed
                    ? "bg-red-500/20 text-red-400"
                  : capabilityInfo.mode !== "classic_fallback"
                    ? "bg-yellow-500/20 text-yellow-400"
                    : "bg-zinc-500/20 text-zinc-400"
              }`}>
                YOLO {capabilityInfo.yolo_ready ? "ready" : capabilityInfo.yolo_failed ? "unavailable" : isClassicFallback ? "N/A" : "pending"}
              </span>
              {(sam2Available || capabilityInfo.sam2_ready) && (
                <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium ${
                  capabilityInfo.sam2_ready
                    ? "bg-green-500/20 text-green-400"
                    : "bg-yellow-500/20 text-yellow-400"
                }`}>
                  SAM2 {capabilityInfo.sam2_ready ? "ready" : "pending"}
                </span>
              )}
              {isClassicFallback && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-500/20 text-blue-400">
                  Zero-shot only
                </span>
              )}
              {capabilityInfo.yolo_failed && (
                <button
                  type="button"
                  onClick={refreshCapabilities}
                  className="text-[10px] text-blue-400 underline cursor-pointer"
                >
                  Retry
                </button>
              )}
            </div>
          )}

          {/* Classic fallback info */}
          {isClassicFallback && classicFallbackReason && (
            <p className="text-[10px] text-zinc-500">
              {classicFallbackReason}. Auto-detect will use YOLO-World zero-shot until a session OBB detector is trained.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default DetectionModeSelector;
