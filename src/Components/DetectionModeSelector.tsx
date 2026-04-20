import { useCallback, useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Switch } from "./ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";
import { selectSam2Enabled } from "../state/hardwareSlice";
import type { ObbDetectionPreset, ObbDetectionSettings } from "@/types/Image";
import { DEFAULT_OBB_DETECTION_SETTINGS, normalizeObbDetectionSettings } from "@/lib/obbDetectorSettings";

export type DetectionMode = "manual" | "auto";
export type DetectionPreset = ObbDetectionPreset;

interface DetectionModeSelectorProps {
  mode: DetectionMode;
  onModeChange: (mode: DetectionMode) => void;
  detectionPreset?: DetectionPreset;
  onDetectionPresetChange?: (preset: DetectionPreset) => void;
  obbDetectionSettings?: ObbDetectionSettings;
  onObbDetectionSettingsChange?: (settings: ObbDetectionSettings) => void;
  obbDetectionRecommendation?: string;
  disabled?: boolean;
  className?: string;
  onClassNameChange?: (name: string) => void;
  samEnabled?: boolean;
  onSamEnabledChange?: (enabled: boolean) => void;
}

type CapabilityInfo = {
  available: boolean;
  mode: "unknown" | "auto_high_performance" | "auto_lite" | "classic_fallback";
  gpu: boolean;
  yolo_ready: boolean;
  sam2_ready: boolean;
  runtimeState?: "not_started" | "checking" | "not_initialized" | "initializing" | "ready" | "failed";
  statusSource?: "local_estimate" | "python_probe" | "python_check";
  pythonPath?: string;
  usingRepoVenv?: boolean;
  yolo_failed?: boolean;
  sam2_failed?: boolean;
  yolo_error?: string | null;
  sam2_error?: string | null;
  error?: string;
};

const INITIAL_CAPABILITY_INFO: CapabilityInfo = {
  available: true,
  mode: "unknown",
  gpu: false,
  yolo_ready: false,
  sam2_ready: false,
  yolo_failed: false,
  sam2_failed: false,
  runtimeState: "checking",
  statusSource: "local_estimate",
};

type HardwareSelectorState = {
  probed: boolean;
  device: "cpu" | "mps" | "cuda" | null;
  ramGb: number | null;
  gpuName: string | null;
  sam2Enabled: boolean;
};

const capabilityInfoEquals = (
  prev: CapabilityInfo,
  next: CapabilityInfo
): boolean => (
  prev.mode === next.mode &&
  prev.gpu === next.gpu &&
  prev.yolo_ready === next.yolo_ready &&
  prev.sam2_ready === next.sam2_ready &&
  !!prev.yolo_failed === !!next.yolo_failed &&
  !!prev.sam2_failed === !!next.sam2_failed &&
  (prev.yolo_error ?? null) === (next.yolo_error ?? null) &&
  (prev.sam2_error ?? null) === (next.sam2_error ?? null) &&
  (prev.runtimeState ?? null) === (next.runtimeState ?? null) &&
  (prev.statusSource ?? null) === (next.statusSource ?? null) &&
  (prev.pythonPath ?? null) === (next.pythonPath ?? null) &&
  (prev.usingRepoVenv ?? null) === (next.usingRepoVenv ?? null)
);

export function DetectionModeSelector({
  mode,
  onModeChange,
  detectionPreset = "balanced",
  onDetectionPresetChange,
  obbDetectionSettings,
  onObbDetectionSettingsChange,
  obbDetectionRecommendation,
  disabled = false,
  className = "",
  onClassNameChange,
  samEnabled = false,
  onSamEnabledChange,
}: DetectionModeSelectorProps) {
  const reduxSam2Enabled = useSelector(selectSam2Enabled);
  const hardware = useSelector(
    (state: any) => state.hardware as HardwareSelectorState
  );

  const [capabilityInfo, setCapabilityInfo] = useState<CapabilityInfo>(INITIAL_CAPABILITY_INFO);
  const resolvedDetectionSettings = normalizeObbDetectionSettings({
    ...DEFAULT_OBB_DETECTION_SETTINGS,
    ...obbDetectionSettings,
    detectionPreset,
  });

  const setCapabilityInfoIfChanged = useCallback((next: CapabilityInfo) => {
    setCapabilityInfo((prev) => (capabilityInfoEquals(prev, next) ? prev : next));
  }, []);

  const refreshCapabilities = useCallback(() => {
    setCapabilityInfo((prev) => (
      prev.runtimeState === "ready" || prev.runtimeState === "initializing"
        ? prev
        : { ...prev, runtimeState: "checking" }
    ));

    window.api.checkSuperAnnotator().then((result) => {
      setCapabilityInfoIfChanged({
        ...INITIAL_CAPABILITY_INFO,
        ...result,
        mode: result.mode ?? "unknown",
        runtimeState: result.runtimeState ?? (result.yolo_ready || result.sam2_ready ? "ready" : "not_initialized"),
        statusSource: result.statusSource ?? "python_check",
      });
    }).catch((error) => {
      setCapabilityInfoIfChanged({
        ...INITIAL_CAPABILITY_INFO,
        mode: "unknown",
        runtimeState: "failed",
        error: error instanceof Error ? error.message : "Failed to query AI runtime.",
      });
    });
  }, [setCapabilityInfoIfChanged]);

  useEffect(() => {
    window.api.initSuperAnnotator().catch(() => {});
    refreshCapabilities();
  }, [refreshCapabilities]);

  useEffect(() => {
    if (
      capabilityInfo.runtimeState === "ready" ||
      capabilityInfo.runtimeState === "failed" ||
      capabilityInfo.yolo_failed
    ) {
      return;
    }
    const id = setInterval(refreshCapabilities, 4000);
    return () => clearInterval(id);
  }, [capabilityInfo.runtimeState, capabilityInfo.yolo_failed, refreshCapabilities]);

  const modeOptions: { value: DetectionMode; label: string; desc: string }[] = [
    { value: "manual", label: "Manual", desc: "Draw bounding boxes manually" },
    { value: "auto", label: "Auto (AI)", desc: "AI detection with YOLO-World zero-shot fallback or the session OBB detector" },
  ];

  const hardwareChecking = !hardware.probed;
  const hardwareGpuDetected = hardware.device !== null && hardware.device !== "cpu";
  const sam2Available = capabilityInfo.mode === "auto_high_performance";
  const isClassicFallback = capabilityInfo.mode === "classic_fallback";

  const classicFallbackReason = capabilityInfo.yolo_failed
    ? `Python error: ${capabilityInfo.yolo_error ?? "initialization failed"}`
    : "Requires >1 GB free RAM or GPU";

  const hardwareStatusLabel = hardwareChecking
    ? "Checking hardware"
    : capabilityInfo.mode === "auto_high_performance"
      ? "High performance"
      : capabilityInfo.mode === "auto_lite"
        ? "Lite mode"
        : capabilityInfo.mode === "classic_fallback"
          ? "Fallback"
          : hardwareGpuDetected
            ? "GPU detected"
            : "CPU only";

  const hardwareStatusClass = hardwareChecking
    ? "bg-yellow-500/20 text-yellow-400"
    : capabilityInfo.mode === "auto_high_performance"
      ? "bg-green-500/20 text-green-400"
      : capabilityInfo.mode === "auto_lite"
        ? "bg-amber-500/20 text-amber-300"
        : "bg-zinc-500/20 text-zinc-300";

  const yoloStatusLabel = capabilityInfo.yolo_ready
    ? "YOLO ready"
    : capabilityInfo.yolo_failed
      ? "YOLO unavailable"
      : capabilityInfo.runtimeState === "initializing"
        ? "YOLO initializing"
        : capabilityInfo.runtimeState === "checking"
          ? "YOLO checking"
          : isClassicFallback
            ? "YOLO N/A"
            : "YOLO not initialized";

  const yoloStatusClass = capabilityInfo.yolo_ready
    ? "bg-green-500/20 text-green-400"
    : capabilityInfo.yolo_failed
      ? "bg-red-500/20 text-red-400"
      : capabilityInfo.runtimeState === "initializing" || capabilityInfo.runtimeState === "checking"
        ? "bg-yellow-500/20 text-yellow-400"
        : isClassicFallback
          ? "bg-zinc-500/20 text-zinc-400"
          : "bg-blue-500/20 text-blue-300";

  const showSam2Badge = sam2Available || capabilityInfo.sam2_ready;
  const sam2StatusLabel = capabilityInfo.sam2_ready
    ? "SAM2 ready"
    : capabilityInfo.runtimeState === "initializing"
      ? "SAM2 initializing"
      : capabilityInfo.runtimeState === "checking"
        ? "SAM2 checking"
        : "SAM2 not initialized";

  const sam2StatusClass = capabilityInfo.sam2_ready
    ? "bg-green-500/20 text-green-400"
    : capabilityInfo.runtimeState === "initializing" || capabilityInfo.runtimeState === "checking"
      ? "bg-yellow-500/20 text-yellow-400"
      : "bg-blue-500/20 text-blue-300";

  const interpreterWarning = (() => {
    if (capabilityInfo.usingRepoVenv !== false) return null;
    const pythonPath = capabilityInfo.pythonPath ?? "python";
    const lowerPath = pythonPath.toLowerCase();
    if (lowerPath.endsWith("biovision_backend.exe") || lowerPath.endsWith("biovision_backend")) {
      return `Bundled backend not found at ${pythonPath}. Reinstall the app or rebuild the backend (npm run backend:build).`;
    }
    return `Using fallback Python interpreter: ${pythonPath}. GPU support may be misdetected if that environment is missing torch or psutil.`;
  })();

  const sam2HelperText = hardwareChecking
    ? "Checking hardware support in the active Python environment"
    : sam2Available
      ? "Precise pixel masks for crowded scenes"
      : !hardwareGpuDetected
        ? "GPU not detected in the active Python environment"
        : !reduxSam2Enabled
          ? "Requires GPU support and at least 8 GB RAM"
          : capabilityInfo.mode === "auto_lite"
            ? "GPU detected, but the runtime is currently in lite mode"
            : "Available after the runtime enters high-performance mode";

  const sam2TooltipMessage = hardwareChecking
    ? "Checking hardware support in the active Python environment."
    : !hardwareGpuDetected
      ? "GPU not detected in the active Python environment."
      : !reduxSam2Enabled
        ? "SAM2 requires a GPU (CUDA or Apple MPS) and at least 8 GB of system RAM."
        : capabilityInfo.mode === "auto_lite"
          ? "GPU detected, but the runtime is currently in lite mode."
          : capabilityInfo.mode === "unknown"
            ? "Checking runtime capability."
            : "SAM2 is ready.";

  const sam2Disabled = disabled || hardwareChecking || !reduxSam2Enabled || !sam2Available;
  const isCustomDetection = resolvedDetectionSettings.detectionPreset === "custom";

  const updateDetectionSettings = useCallback((patch: Partial<ObbDetectionSettings>) => {
    const next = normalizeObbDetectionSettings({
      ...resolvedDetectionSettings,
      ...patch,
    });
    onDetectionPresetChange?.(next.detectionPreset ?? "balanced");
    onObbDetectionSettingsChange?.(next);
  }, [onDetectionPresetChange, onObbDetectionSettingsChange, resolvedDetectionSettings]);

  return (
    <div className="flex flex-col gap-3 p-3 bg-zinc-800 rounded-lg">
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

      {mode === "manual" && (
        <div className="flex flex-col gap-2 pt-2 border-t border-zinc-700">
          <p className="text-xs text-zinc-500">
            Drag to draw a box. Select a box to drag, resize, or rotate it - use the rotation handle above the box to create oriented (OBB) boxes.
          </p>
        </div>
      )}

      {mode === "auto" && (
        <div className="flex flex-col gap-3 pt-2 border-t border-zinc-700">
          <div className="flex flex-col gap-1.5">
            <Label className="text-xs text-zinc-400">
              Detection goal
            </Label>
            <select
              value={resolvedDetectionSettings.detectionPreset}
              onChange={(e) => updateDetectionSettings({ detectionPreset: e.target.value as DetectionPreset })}
              disabled={disabled}
              className="h-8 rounded-md border border-zinc-700 bg-zinc-900 px-2 text-xs text-white"
            >
              <option value="balanced">Balanced</option>
              <option value="precision">Precision (fewer false positives)</option>
              <option value="recall">Recall (find more objects)</option>
              <option value="single_object">Single object focus</option>
              <option value="custom">Custom</option>
            </select>
            {obbDetectionRecommendation && (
              <p className="text-[10px] text-zinc-500">
                {obbDetectionRecommendation}
              </p>
            )}
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

          <div className="flex items-center justify-between pt-1">
            <div className="flex flex-col gap-0.5">
              <Label className="text-xs text-zinc-400">
                SAM2 mask refinement
              </Label>
              <p className="text-[10px] text-zinc-500">
                {sam2HelperText}
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
                  {sam2TooltipMessage}
                </TooltipContent>
              )}
            </Tooltip>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium ${hardwareStatusClass}`}>
              {hardwareStatusLabel}
            </span>
            <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium ${yoloStatusClass}`}>
              {yoloStatusLabel}
            </span>
            {showSam2Badge && (
              <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium ${sam2StatusClass}`}>
                {sam2StatusLabel}
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

          <details className="rounded-md border border-zinc-700/80 bg-zinc-900/70">
            <summary className="cursor-pointer px-3 py-2 text-xs font-semibold text-zinc-300 select-none">
              Advanced OBB Detection
            </summary>
            <div className="grid grid-cols-2 gap-2 px-3 pb-3 pt-1">
              <div className="flex flex-col gap-1">
                <Label className="text-[10px] text-zinc-500">Confidence</Label>
                <Input
                  type="number"
                  min={0.01}
                  max={0.99}
                  step={0.01}
                  value={resolvedDetectionSettings.conf ?? DEFAULT_OBB_DETECTION_SETTINGS.conf}
                  onChange={(e) => updateDetectionSettings({ conf: Number(e.target.value) })}
                  disabled={disabled || !isCustomDetection}
                  className="h-8 bg-zinc-950 border-zinc-700 text-xs text-white"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-[10px] text-zinc-500">NMS IoU</Label>
                <Input
                  type="number"
                  min={0.05}
                  max={0.95}
                  step={0.01}
                  value={resolvedDetectionSettings.nmsIou ?? DEFAULT_OBB_DETECTION_SETTINGS.nmsIou}
                  onChange={(e) => updateDetectionSettings({ nmsIou: Number(e.target.value) })}
                  disabled={disabled || !isCustomDetection}
                  className="h-8 bg-zinc-950 border-zinc-700 text-xs text-white"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-[10px] text-zinc-500">Max objects</Label>
                <Input
                  type="number"
                  min={1}
                  max={250}
                  step={1}
                  value={resolvedDetectionSettings.maxObjects ?? DEFAULT_OBB_DETECTION_SETTINGS.maxObjects}
                  onChange={(e) => updateDetectionSettings({ maxObjects: Number(e.target.value) })}
                  disabled={disabled || !isCustomDetection}
                  className="h-8 bg-zinc-950 border-zinc-700 text-xs text-white"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-[10px] text-zinc-500">OBB resolution</Label>
                <select
                  value={resolvedDetectionSettings.imgsz ?? DEFAULT_OBB_DETECTION_SETTINGS.imgsz}
                  onChange={(e) => updateDetectionSettings({ imgsz: Number(e.target.value) as 640 | 960 | 1280 })}
                  disabled={disabled || !isCustomDetection}
                  className="h-8 rounded-md border border-zinc-700 bg-zinc-950 px-2 text-xs text-white"
                >
                  <option value={640}>640</option>
                  <option value={960}>960</option>
                  <option value={1280}>1280</option>
                </select>
              </div>
              {!isCustomDetection && (
                <p className="col-span-2 text-[10px] text-zinc-500">
                  Preset modes manage these values automatically. Switch to Custom to override them.
                </p>
              )}
            </div>
          </details>

          {interpreterWarning && (
            <p className="text-[10px] text-amber-300">
              {interpreterWarning}
            </p>
          )}

          {isClassicFallback && (
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
