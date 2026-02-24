import { useCallback, useEffect, useState } from "react";
import { Label } from "./ui/label";
import { Slider } from "./ui/slider";
import { Input } from "./ui/input";
import { Switch } from "./ui/switch";

export type DetectionMode = "manual" | "auto";
export type DetectionPreset = "balanced" | "precision" | "recall" | "single_object";

interface DetectionModeSelectorProps {
  mode: DetectionMode;
  onModeChange: (mode: DetectionMode) => void;
  autoConfidence: number;
  onAutoConfidenceChange: (value: number) => void;
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
  autoConfidence,
  onAutoConfidenceChange,
  detectionPreset = "balanced",
  onDetectionPresetChange,
  disabled = false,
  className = "",
  onClassNameChange,
  samEnabled = false,
  onSamEnabledChange,
}: DetectionModeSelectorProps) {
  const [capabilityInfo, setCapabilityInfo] = useState<{
    mode: string;
    gpu: boolean;
    yolo_ready: boolean;
    sam2_ready: boolean;
    yolo_failed: boolean;
    sam2_failed: boolean;
    yolo_error?: string | null;
  } | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);

  const refreshCapabilities = useCallback(() => {
    window.api.checkSuperAnnotator().then((result) => {
      setCapabilityInfo({
        mode: result.mode,
        gpu: result.gpu,
        yolo_ready: result.yolo_ready,
        sam2_ready: result.sam2_ready,
        yolo_failed: !!result.yolo_failed,
        sam2_failed: !!result.sam2_failed,
        yolo_error: result.yolo_error,
      });
    }).catch(() => {
      setCapabilityInfo({
        mode: "classic_fallback",
        gpu: false,
        yolo_ready: false,
        sam2_ready: false,
        yolo_failed: false,
        sam2_failed: false,
      });
    });
  }, []);

  useEffect(() => {
    refreshCapabilities();
  }, [refreshCapabilities]);

  // Re-check capabilities after models load (triggered by first auto-detect)
  useEffect(() => {
    const recheck = () => {
      refreshCapabilities();
    };
    window.addEventListener("super-annotator-ready", recheck);
    return () => window.removeEventListener("super-annotator-ready", recheck);
  }, [refreshCapabilities]);

  const modeOptions: { value: DetectionMode; label: string; desc: string }[] = [
    { value: "manual", label: "Manual", desc: "Draw bounding boxes manually" },
    { value: "auto", label: "Auto (AI)", desc: "AI detection (YOLO-World, or OpenCV fallback if unavailable)" },
  ];

  const sam2Available = capabilityInfo?.mode === "auto_high_performance";

  const yoloStatusHint = capabilityInfo
    ? capabilityInfo.yolo_ready
      ? "YOLO ready."
      : capabilityInfo.yolo_failed
        ? `YOLO unavailable: ${capabilityInfo.yolo_error ?? "initialization failed"}`
        : capabilityInfo.mode === "classic_fallback"
          ? "YOLO not attempted: requires more free RAM (>2GB) or GPU."
          : "YOLO pending: click Initialize models."
    : null;
  const sam2StatusHint = capabilityInfo
    ? capabilityInfo.sam2_ready
      ? "SAM2 ready."
      : sam2Available
        ? "SAM2 pending: initialize models to enable mask refinement."
        : "SAM2 system-gated: requires GPU + sufficient memory."
    : null;

  const isClassicFallback = capabilityInfo?.mode === "classic_fallback";
  const classicFallbackReason = capabilityInfo
    ? capabilityInfo.yolo_failed
      ? `Python error: ${capabilityInfo.yolo_error ?? "initialization failed"}`
      : "Requires >1.5 GB free RAM or GPU"
    : null;

  const handleInitializeModels = async () => {
    setIsInitializing(true);
    try {
      await window.api.initSuperAnnotator();
    } finally {
      refreshCapabilities();
      window.dispatchEvent(new CustomEvent("super-annotator-ready"));
      setIsInitializing(false);
    }
  };

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
            Click and drag on the canvas to draw bounding boxes.
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

          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-zinc-400">
                Confidence: {(autoConfidence * 100).toFixed(0)}%
              </Label>
            </div>
            <Slider
              value={[autoConfidence]}
              onValueChange={([value]) => onAutoConfidenceChange(value)}
              min={0.2}
              max={0.9}
              step={0.05}
              disabled={disabled}
              className="w-full"
            />
          </div>

          {/* SAM2 refinement toggle */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex flex-col gap-0.5">
              <Label className="text-xs text-zinc-400">
                SAM2 mask refinement
              </Label>
              <p className="text-[10px] text-zinc-500">
                {sam2Available
                  ? "Precise pixel masks for crowded scenes"
                  : "Requires GPU + SAM2 model"}
              </p>
            </div>
            <Switch
              checked={samEnabled}
              onCheckedChange={(checked) => onSamEnabledChange?.(checked)}
              disabled={disabled || !sam2Available}
            />
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
              <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium ${
                capabilityInfo.sam2_ready
                  ? "bg-green-500/20 text-green-400"
                  : sam2Available
                    ? "bg-yellow-500/20 text-yellow-400"
                    : "bg-zinc-500/20 text-zinc-400"
              }`}>
                SAM2 {capabilityInfo.sam2_ready ? "ready" : sam2Available ? "pending" : "N/A"}
              </span>
              {capabilityInfo.gpu && (
                <span className="text-[10px] text-zinc-500">GPU</span>
              )}
              {isClassicFallback && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-500/20 text-blue-400">
                  OpenCV fallback
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

          {/* Small status hint — suppress when classic_fallback since button sub-text already explains it */}
          {yoloStatusHint && !isClassicFallback && (
            <p className={`text-[10px] ${
              capabilityInfo?.yolo_failed ? "text-red-400/90" : "text-zinc-500"
            }`}>
              Status: {yoloStatusHint}
            </p>
          )}
          {sam2StatusHint && (
            <p className="text-[10px] text-zinc-500">
              SAM2: {sam2StatusHint}
            </p>
          )}

          {/* Initialize models — always visible in auto mode */}
          {capabilityInfo === null ? (
            <button
              type="button"
              disabled
              className="h-8 px-3 rounded-md text-xs font-medium text-white border border-zinc-700 bg-zinc-800/60 opacity-60 cursor-not-allowed"
            >
              Checking environment...
            </button>
          ) : isClassicFallback ? (
            <div className="flex flex-col gap-1">
              <button
                type="button"
                disabled
                className="h-8 px-3 rounded-md text-xs font-medium text-white border border-zinc-700 bg-zinc-800/60 opacity-60 cursor-not-allowed"
              >
                Initialize models (unavailable)
              </button>
              <p className="text-[10px] text-zinc-500">{classicFallbackReason}. Auto-detect will use OpenCV.</p>
            </div>
          ) : (
            <button
              type="button"
              onClick={handleInitializeModels}
              disabled={disabled || isInitializing || !!capabilityInfo.yolo_ready}
              className={`h-8 px-3 rounded-md text-xs font-medium text-white border border-zinc-600 ${
                disabled || isInitializing || !!capabilityInfo.yolo_ready
                  ? "bg-zinc-800/60 opacity-60 cursor-not-allowed"
                  : "bg-zinc-700 hover:bg-zinc-600"
              }`}
            >
              {capabilityInfo.yolo_ready
                ? "Models initialized"
                : isInitializing
                  ? "Initializing models..."
                  : "Initialize models"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default DetectionModeSelector;
