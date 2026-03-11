import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Info, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Input } from "@/Components/ui/input";
import { Label } from "@/Components/ui/label";
import { Progress } from "@/Components/ui/progress";
import { ScrollArea } from "@/Components/ui/scroll-area";
import { Slider } from "@/Components/ui/slider";
import { Switch } from "@/Components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/Components/ui/dialog";

const SCHEMA_DEFAULTS: Record<string, { flip_prob: number; vertical_flip_prob: number; rotate_180_prob: number }> = {
  directional: { flip_prob: 0.5, vertical_flip_prob: 0.0, rotate_180_prob: 0.0 },
  bilateral:   { flip_prob: 0.5, vertical_flip_prob: 0.0, rotate_180_prob: 0.0 },
  axial:       { flip_prob: 0.5, vertical_flip_prob: 0.0, rotate_180_prob: 0.5 },
  invariant:   { flip_prob: 0.5, vertical_flip_prob: 0.25, rotate_180_prob: 0.0 },
};

interface TrainModelDialogProps {
  open: boolean;
  setOpen: (value: boolean) => void;
  handleTrainConfirm: () => Promise<void>;
  setModelName: (name: string) => void;
  isTraining?: boolean;
  modelName: string;
  preflightSummary?: string;
  preflightWarning?: string;
  predictorType?: "dlib" | "cnn";
  setPredictorType?: (type: "dlib" | "cnn") => void;
  // OBB detector training (Phase 3 - step before landmarker)
  obbDetectorReady?: boolean;
  /** Show the OBB detector step — true when user has finalized boxes (flat or OBB) */
  showObbStep?: boolean;
  isTrainingObb?: boolean;
  handleTrainObbDetector?: () => Promise<void>;
  obbTrainingMessage?: string;
  obbHyperparams?: { iou: number; cls: number; box: number };
  onObbHyperparamsChange?: (v: { iou: number; cls: number; box: number }) => void;
  cnnVariants?: {
    id: string;
    label: string;
    description: string;
    selectable: boolean;
    recommended?: boolean;
    reason?: string | null;
    recommendationReason?: string | null;
  }[];
  cnnVariant?: string;
  setCnnVariant?: (variant: string) => void;
  cnnVariantWarning?: string;
  speciesId?: string;
  augmentationPolicy?: AugmentationPolicy;
  onAugmentationPolicyChange?: (policy: AugmentationPolicy) => void;
  orientationMode?: string;
  trainingProgress?: {
    percent: number;
    stage: string;
    message: string;
    predictorType: "dlib" | "cnn";
    modelName: string;
    details?: {
      substage?: string;
      epoch?: number;
      epochs?: number;
      loss?: number;
      lr?: number;
      elapsed_sec?: number;
      eta_sec?: number;
      samples_per_sec?: number;
      split?: string;
      eval_mode?: string;
      records_total?: number;
      records_done?: number;
      batch_size?: number;
      workers?: number;
      amp_enabled?: boolean;
      device?: string;
    };
  } | null;
}

export const TrainModelDialog: React.FC<TrainModelDialogProps> = ({
  open,
  setOpen,
  handleTrainConfirm,
  modelName,
  setModelName,
  isTraining = false,
  preflightSummary,
  preflightWarning,
  predictorType = "dlib",
  setPredictorType,
  cnnVariants = [],
  cnnVariant = "simplebaseline",
  setCnnVariant,
  cnnVariantWarning,
  speciesId,
  augmentationPolicy,
  onAugmentationPolicyChange,
  orientationMode,
  trainingProgress = null,
  obbDetectorReady = false,
  showObbStep = false,
  isTrainingObb = false,
  handleTrainObbDetector,
  obbTrainingMessage,
  obbHyperparams = { iou: 0.3, cls: 1.5, box: 5.0 },
  onObbHyperparamsChange,
}) => {
  const [touched, setTouched] = useState(false);

  // Local editable copy of augmentationPolicy; syncs from props when dialog opens
  const [localAugPolicy, setLocalAugPolicy] = useState<AugmentationPolicy>({
    gravity_aligned: true,
    rotation_range: [-15, 15],
    scale_range: [0.9, 1.1],
    flip_prob: 0.5,
    vertical_flip_prob: 0.0,
    rotate_180_prob: 0.0,
  });

  useEffect(() => {
    if (!open) {
      setTouched(false);
      return;
    }
    const sd = SCHEMA_DEFAULTS[orientationMode ?? "invariant"] ?? SCHEMA_DEFAULTS.invariant;
    const gravAlign = augmentationPolicy?.gravity_aligned ?? true;
    const maxRot = !gravAlign && orientationMode === "invariant" ? 180 : 15;
    const rawRange = augmentationPolicy?.rotation_range ?? [-15, 15];
    const clampedRange: [number, number] = [
      Math.max(-maxRot, rawRange[0]),
      Math.min(maxRot,  rawRange[1]),
    ];
    setLocalAugPolicy({
      gravity_aligned:    gravAlign,
      rotation_range:     clampedRange,
      scale_range:        augmentationPolicy?.scale_range        ?? [0.9, 1.1],
      flip_prob:           augmentationPolicy?.flip_prob           ?? sd.flip_prob,
      vertical_flip_prob: augmentationPolicy?.vertical_flip_prob ?? sd.vertical_flip_prob,
      rotate_180_prob:    augmentationPolicy?.rotate_180_prob    ?? sd.rotate_180_prob,
    });
  }, [open, augmentationPolicy, orientationMode]);

  const handleAugPolicyChange = useCallback((patch: Partial<AugmentationPolicy>) => {
    setLocalAugPolicy((prev) => {
      const next = { ...prev, ...patch };
      onAugmentationPolicyChange?.(next);
      return next;
    });
  }, [onAugmentationPolicyChange]);

  const trimmed = useMemo(() => modelName.trim(), [modelName]);

  const nameOk = useMemo(() => /^[a-zA-Z0-9._-]+$/.test(trimmed), [trimmed]);
  const canTrain =
    trimmed.length > 0 &&
    nameOk &&
    !isTraining &&
    (!showObbStep || obbDetectorReady);

  const helperText = useMemo(() => {
    if (!touched)
      return "Use letters, numbers, hyphen (-), underscore (_), dot (.), or colon (:).";
    if (!trimmed) return "Model name is required.";
    if (!nameOk) return "Only letters, numbers, ., _, -, : are allowed (no spaces).";
    return "Looks good.";
  }, [touched, trimmed, nameOk]);

  const handleClose = useCallback(() => {
    if (isTraining) return;
    setModelName("");
    setOpen(false);
  }, [isTraining, setModelName, setOpen]);

  const onTrain = useCallback(async () => {
    if (!canTrain) return;
    await handleTrainConfirm();
  }, [canTrain, handleTrainConfirm]);

  const selectedCnnVariant = useMemo(
    () => cnnVariants.find((v) => v.id === cnnVariant) ?? null,
    [cnnVariant, cnnVariants]
  );
  const progressDetails = trainingProgress?.details ?? null;
  const progressEpochLabel = useMemo(() => {
    if (!progressDetails) return null;
    const epoch = Number(progressDetails.epoch);
    const epochs = Number(progressDetails.epochs);
    if (Number.isFinite(epoch) && Number.isFinite(epochs) && epochs > 0) {
      return `Epoch ${Math.round(epoch)}/${Math.round(epochs)}`;
    }
    return null;
  }, [progressDetails]);
  const progressLossLabel = useMemo(() => {
    if (!progressDetails) return null;
    const loss = Number(progressDetails.loss);
    if (Number.isFinite(loss)) return `Loss ${loss.toFixed(4)}`;
    return null;
  }, [progressDetails]);
  const progressLrLabel = useMemo(() => {
    if (!progressDetails) return null;
    const lr = Number(progressDetails.lr);
    if (Number.isFinite(lr) && lr > 0) return `LR ${lr.toExponential(2)}`;
    return null;
  }, [progressDetails]);
  const progressEtaLabel = useMemo(() => {
    if (!progressDetails) return null;
    const eta = Number(progressDetails.eta_sec);
    if (Number.isFinite(eta) && eta >= 0) return `ETA ${Math.round(eta)}s`;
    return null;
  }, [progressDetails]);
  const cnnSupported = useMemo(
    () => cnnVariants.length === 0 || cnnVariants.some((v) => v.selectable),
    [cnnVariants]
  );

  // Schema-aware gate variables for the Augmentation Studio
  const gravityAligned = localAugPolicy.gravity_aligned ?? true;
  const rotationMax = !gravityAligned && orientationMode === "invariant" ? 180 : 15;
  const showVerticalFlip = !gravityAligned && orientationMode === "invariant";
  const showRotate180 = !gravityAligned && orientationMode === "axial";
  const showDirectionalHint = orientationMode === "directional";
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  const [showScrollTopFade, setShowScrollTopFade] = useState(false);
  const [showScrollBottomFade, setShowScrollBottomFade] = useState(false);

  useEffect(() => {
    if (!open) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleClose();
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        onTrain();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, handleClose, onTrain]);

  useEffect(() => {
    if (!open) {
      setShowScrollTopFade(false);
      setShowScrollBottomFade(false);
      return;
    }

    let viewport: HTMLDivElement | null = null;
    let resizeObserver: ResizeObserver | null = null;
    let frameId = 0;
    const syncScrollFades = () => {
      if (!viewport) return;
      const maxScrollTop = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
      const hasOverflow = maxScrollTop > 6;
      setShowScrollTopFade(hasOverflow && viewport.scrollTop > 6);
      setShowScrollBottomFade(hasOverflow && viewport.scrollTop < maxScrollTop - 6);
    };

    frameId = window.requestAnimationFrame(() => {
      viewport = scrollAreaRef.current?.querySelector("[data-radix-scroll-area-viewport]") as HTMLDivElement | null;
      if (!viewport) return;
      syncScrollFades();
      viewport.addEventListener("scroll", syncScrollFades, { passive: true });
      window.addEventListener("resize", syncScrollFades);
      if (typeof ResizeObserver !== "undefined") {
        resizeObserver = new ResizeObserver(syncScrollFades);
        resizeObserver.observe(viewport);
        const content = viewport.firstElementChild as HTMLElement | null;
        if (content) {
          resizeObserver.observe(content);
        }
      }
    });

    return () => {
      window.cancelAnimationFrame(frameId);
      if (viewport) {
        viewport.removeEventListener("scroll", syncScrollFades);
      }
      resizeObserver?.disconnect();
      window.removeEventListener("resize", syncScrollFades);
    };
  }, [
    open,
    predictorType,
    obbDetectorReady,
    showObbStep,
    isTraining,
    preflightSummary,
    preflightWarning,
    cnnVariantWarning,
    orientationMode,
    speciesId,
    trainingProgress?.percent,
    localAugPolicy,
  ]);

  return (
    <Dialog open={open} onOpenChange={(value) => !isTraining && setOpen(value)}>
      <DialogContent className="flex h-[92vh] max-h-[92vh] min-h-0 flex-col overflow-hidden p-0 sm:max-w-lg">
        <DialogHeader className="shrink-0">
          <div className="px-6 pt-6">
          <DialogTitle className="text-sm font-bold">
            Train new model
          </DialogTitle>
          <DialogDescription className="text-xs">
            Give your model a clear, versioned name (Ctrl/Cmd+Enter to start).
          </DialogDescription>
          </div>
        </DialogHeader>
        <div className="relative min-h-0 flex-1 overflow-hidden px-6">
          <div
            className={cn(
              "pointer-events-none absolute inset-x-1 top-0 z-10 h-8 rounded-t-xl bg-gradient-to-b from-background via-background/92 to-transparent transition-opacity",
              showScrollTopFade ? "opacity-100" : "opacity-0"
            )}
          />
          <div
            className={cn(
              "pointer-events-none absolute inset-x-1 bottom-0 z-10 h-10 rounded-b-xl bg-gradient-to-t from-background via-background/92 to-transparent transition-opacity",
              showScrollBottomFade ? "opacity-100" : "opacity-0"
            )}
          />
          <ScrollArea
            ref={scrollAreaRef}
            className="h-full min-h-0 rounded-xl border border-border/60 bg-muted/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]"
          >
            <div className="space-y-1 px-4 pb-6 pt-3">

        {/* Step 1: OBB Detector — shown when session has OBB annotations or detector already trained */}
        {speciesId && (
          <details className="rounded-md border border-border/60 bg-muted/20">
            <summary className="cursor-pointer px-3 py-2 text-xs font-semibold select-none">
              Session Orientation Policy
            </summary>
            <div className="flex flex-col gap-3 px-3 pb-3 pt-1">
              <div className="flex items-start gap-3">
                <div>
                  <Label className="text-xs font-medium">Gravity-aligned imaging</Label>
                  <p className="text-[11px] text-muted-foreground">
                    Keeps upright specimens on a tight rotation prior by default.
                  </p>
                </div>
                <p className="flex-1 text-[11px] text-muted-foreground">
                  This shared session setting affects OBB synthetic export and OBB detector training, and also informs CNN augmentation defaults.
                </p>
                <Switch
                  checked={localAugPolicy.gravity_aligned ?? true}
                  onCheckedChange={(v) => {
                    const newPolicy: Partial<AugmentationPolicy> = { gravity_aligned: v };
                    if (v) newPolicy.rotation_range = [-15, 15];
                    handleAugPolicyChange(newPolicy);
                  }}
                  disabled={isTraining}
                />
              </div>

              <div className="flex flex-col gap-1">
                <Label className="text-xs text-muted-foreground">
                  Shared rotation range: {(localAugPolicy.rotation_range?.[0] ?? -15).toFixed(0)}° to {(localAugPolicy.rotation_range?.[1] ?? 15).toFixed(0)}°
                </Label>
                <p className="text-[11px] text-muted-foreground">
                  Used by OBB synthetic export, YOLO OBB augmentation, and CNN rotation augmentation. Dlib remains capped separately in Python.
                </p>
                <Slider
                  value={localAugPolicy.rotation_range ?? [-15, 15]}
                  onValueChange={([lo, hi]) => handleAugPolicyChange({ rotation_range: [lo, hi] })}
                  min={-rotationMax}
                  max={rotationMax}
                  step={1}
                  disabled={isTraining}
                />
              </div>
            </div>
          </details>
        )}

        {(showObbStep || obbDetectorReady) && (
          <div className={`rounded-md border px-3 py-2 mb-1 ${obbDetectorReady ? "border-green-500/40 bg-green-500/5" : "border-amber-500/40 bg-amber-500/5"}`}>
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <p className="text-xs font-semibold">
                  {obbDetectorReady ? "✓ Step 1: OBB Detector — ready" : "Step 1: Train OBB Detector"}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  {obbDetectorReady
                    ? (obbTrainingMessage ?? "Retrain to update with new annotations.")
                    : (obbTrainingMessage ?? "Train the orientation detector before landmarking.")}
                </p>
              </div>
              {handleTrainObbDetector && (
                <Button
                  size="sm"
                  variant="outline"
                  className="shrink-0 text-xs h-7"
                  disabled={isTrainingObb || isTraining}
                  onClick={handleTrainObbDetector}
                >
                  {isTrainingObb
                    ? <><Loader2 className="mr-1 h-3 w-3 animate-spin" />Training…</>
                    : obbDetectorReady ? "Retrain OBB" : "Train OBB"}
                </Button>
              )}
            </div>
            {/* Advanced OBB hyperparameters accordion */}
            {onObbHyperparamsChange && (
              <details className="mt-2 rounded-md border border-border/60 bg-muted/20">
                <summary className="cursor-pointer px-3 py-2 text-xs font-semibold select-none">
                  Advanced Training Hyperparameters
                </summary>
                <div className="flex flex-col gap-3 px-3 pb-3 pt-1">
                  <div className="flex flex-col gap-1">
                    <Label className="text-xs text-muted-foreground">
                      NMS IoU threshold: {obbHyperparams.iou.toFixed(2)}
                      <span className="ml-1 text-[10px] opacity-60">(lower = suppress more overlapping boxes)</span>
                    </Label>
                    <Slider
                      value={[obbHyperparams.iou]}
                      onValueChange={([v]) => onObbHyperparamsChange({ ...obbHyperparams, iou: v })}
                      min={0.1} max={0.9} step={0.05}
                      disabled={isTrainingObb || isTraining}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label className="text-xs text-muted-foreground">
                      Classification loss weight: {obbHyperparams.cls.toFixed(1)}
                      <span className="ml-1 text-[10px] opacity-60">(higher = sharper confidence scores)</span>
                    </Label>
                    <Slider
                      value={[obbHyperparams.cls]}
                      onValueChange={([v]) => onObbHyperparamsChange({ ...obbHyperparams, cls: v })}
                      min={0.1} max={3.0} step={0.1}
                      disabled={isTrainingObb || isTraining}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label className="text-xs text-muted-foreground">
                      Box regression loss weight: {obbHyperparams.box.toFixed(1)}
                    </Label>
                    <Slider
                      value={[obbHyperparams.box]}
                      onValueChange={([v]) => onObbHyperparamsChange({ ...obbHyperparams, box: v })}
                      min={1.0} max={10.0} step={0.5}
                      disabled={isTrainingObb || isTraining}
                    />
                  </div>
                </div>
              </details>
            )}
          </div>
        )}
        {/* Step 2 label — only once OBB detector is ready */}
        {obbDetectorReady && (
          <p className="text-[11px] font-semibold text-muted-foreground px-1">
            Step 2: Train Landmark Predictor
          </p>
        )}

        {showObbStep && !obbDetectorReady && (
          <p className="text-[11px] text-muted-foreground px-1 py-2">
            Train the OBB detector above before configuring the landmark predictor.
          </p>
        )}

        {false && speciesId && (
          <details className="rounded-md border border-border/60 bg-muted/20">
            <summary className="cursor-pointer px-3 py-2 text-xs font-semibold select-none">
              Session Orientation Policy
            </summary>
            <div className="flex flex-col gap-3 px-3 pb-3 pt-1">
              <div className="flex items-start gap-3">
                <div>
                  <Label className="text-xs font-medium">Gravity-aligned imaging</Label>
                  <p className="text-[11px] text-muted-foreground">
                    Keeps upright specimens on a tight rotation prior by default.
                  </p>
                </div>
                <p className="flex-1 text-[11px] text-muted-foreground">
                  This shared session setting affects OBB synthetic export and OBB detector training, and also informs CNN augmentation defaults.
                </p>
                <Switch
                  checked={localAugPolicy.gravity_aligned ?? true}
                  onCheckedChange={(v) => {
                    const newPolicy: Partial<AugmentationPolicy> = { gravity_aligned: v };
                    if (v) newPolicy.rotation_range = [-15, 15];
                    handleAugPolicyChange(newPolicy);
                  }}
                  disabled={isTraining}
                />
              </div>

              <div className="flex flex-col gap-1">
                <Label className="text-xs text-muted-foreground">
                  Shared rotation range: {(localAugPolicy.rotation_range?.[0] ?? -15).toFixed(0)}° to {(localAugPolicy.rotation_range?.[1] ?? 15).toFixed(0)}°
                </Label>
                <p className="text-[11px] text-muted-foreground">
                  Used by OBB synthetic export, YOLO OBB augmentation, and CNN rotation augmentation. Dlib remains capped separately in Python.
                </p>
                <Slider
                  value={localAugPolicy.rotation_range ?? [-15, 15]}
                  onValueChange={([lo, hi]) => handleAugPolicyChange({ rotation_range: [lo, hi] })}
                  min={-rotationMax}
                  max={rotationMax}
                  step={1}
                  disabled={isTraining}
                />
              </div>
            </div>
          </details>
        )}

        {(!showObbStep || obbDetectorReady) && <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="model-name" className="text-sm font-medium">
              Model name
            </Label>
            <div className="relative">
              <Info className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                id="model-name"
                autoFocus
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                onBlur={() => setTouched(true)}
                placeholder="e.g. fossil_landmarks_v1"
                disabled={isTraining}
                className={cn(
                  "pl-10",
                  touched &&
                  (!trimmed || !nameOk) &&
                  "border-destructive focus-visible:ring-destructive"
                )}
              />
            </div>
            <p
              className={cn(
                "text-xs",
                touched && (!trimmed || !nameOk)
                  ? "text-destructive"
                  : "text-muted-foreground"
              )}
            >
              {helperText}
            </p>
          </div>

          {/* Predictor type selection */}
          {setPredictorType && (
            <div className="space-y-2">
              <Label className="text-sm font-medium">Predictor type</Label>
              <div className="grid gap-2">
                <label
                  className={cn(
                    "flex cursor-pointer items-start gap-3 rounded-md border p-3 transition-colors",
                    predictorType === "dlib"
                      ? "border-primary bg-primary/5"
                      : "border-border/50 hover:border-border"
                  )}
                >
                  <input
                    type="radio"
                    name="predictorType"
                    value="dlib"
                    checked={predictorType === "dlib"}
                    onChange={() => setPredictorType("dlib")}
                    disabled={isTraining}
                    className="mt-0.5 accent-primary"
                  />
                  <div>
                    <p className="text-xs font-semibold">dlib Shape Predictor</p>
                    <p className="text-[11px] text-muted-foreground">
                      Fast controlled-image fallback. Best when crops are highly standardized and appearance variation is limited.
                    </p>
                  </div>
                </label>
                <label
                  className={cn(
                    "flex cursor-pointer items-start gap-3 rounded-md border p-3 transition-colors",
                    predictorType === "cnn"
                      ? "border-primary bg-primary/5"
                      : "border-border/50 hover:border-border"
                  )}
                >
                  <input
                    type="radio"
                    name="predictorType"
                    value="cnn"
                    checked={predictorType === "cnn"}
                    onChange={() => setPredictorType("cnn")}
                    disabled={isTraining || !cnnSupported}
                    className="mt-0.5 accent-primary"
                  />
                  <div>
                    <p className="text-xs font-semibold">CNN Landmark Model</p>
                    <p className="text-[11px] text-muted-foreground">
                      Recommended for general use. Stronger visual generalization, heatmap-based localization, and dataset-size-aware backbone selection.
                    </p>
                    {!cnnSupported && (
                      <p className="mt-1 text-[11px] text-amber-600 dark:text-amber-400">
                        CNN training is unavailable on this system.
                      </p>
                    )}
                  </div>
                </label>
              </div>
              {predictorType === "cnn" && setCnnVariant && (
                <div className="space-y-2 rounded-md border border-border/60 bg-muted/30 p-3">
                  <Label htmlFor="cnn-variant" className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    CNN architecture
                  </Label>
                  <select
                    id="cnn-variant"
                    value={cnnVariant}
                    onChange={(e) => setCnnVariant(e.target.value)}
                    disabled={isTraining}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-xs text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    {cnnVariants.map((variant) => (
                      <option
                        key={variant.id}
                        value={variant.id}
                        disabled={!variant.selectable}
                      >
                        {variant.label}
                        {variant.recommended ? " (Recommended)" : ""}
                        {!variant.selectable ? " - unavailable" : ""}
                      </option>
                    ))}
                  </select>
                  {selectedCnnVariant && (
                    <div className="space-y-1">
                      <p className="text-[11px] text-muted-foreground">
                        {selectedCnnVariant.description}
                      </p>
                      {selectedCnnVariant.recommendationReason && (
                        <p className="text-[11px] text-emerald-700 dark:text-emerald-400">
                          {selectedCnnVariant.recommendationReason}
                        </p>
                      )}
                    </div>
                  )}
                  {!selectedCnnVariant?.selectable && selectedCnnVariant?.reason && (
                    <p className="text-[11px] text-amber-600 dark:text-amber-400">
                      {selectedCnnVariant.reason}
                    </p>
                  )}
                  {cnnVariantWarning && (
                    <p className="text-[11px] text-amber-600 dark:text-amber-400">
                      {cnnVariantWarning}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Augmentation & Schema Studio — only shown when a session is active and CNN is selected */}
          {speciesId && predictorType === "cnn" && (
            <details className="rounded-md border border-border/60 bg-muted/20">
              <summary className="cursor-pointer px-3 py-2 text-xs font-semibold select-none">
                CNN Augmentation Studio
              </summary>
              <div className="flex flex-col gap-3 px-3 pb-3 pt-1">
                <p className="text-[11px] text-muted-foreground">
                  These controls affect CNN augmentation only. Shared session orientation settings are configured above.
                </p>
                <div className="hidden">
                  <div>
                    <Label className="text-xs font-medium">Gravity-aligned imaging</Label>
                    <p className="text-[11px] text-muted-foreground">
                      Locks synthetic rotation to ±15° (for specimens always upright)
                    </p>
                  </div>
                  <p className="flex-1 text-[11px] text-muted-foreground">
                    Upright sessions should usually keep this range tight so OBB export and training do not learn unnecessary tilt.
                  </p>
                  <Switch
                    checked={localAugPolicy.gravity_aligned ?? true}
                    onCheckedChange={(v) => {
                      const newPolicy: Partial<AugmentationPolicy> = { gravity_aligned: v };
                      if (v) newPolicy.rotation_range = [-15, 15];
                      handleAugPolicyChange(newPolicy);
                    }}
                    disabled={isTraining}
                  />
                </div>

                {/* Rotation range — CNN micro-augmentation only; Dlib is always capped at ±6° in Python */}
                <div className="hidden flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    CNN rotation: {(localAugPolicy.rotation_range?.[0] ?? -15).toFixed(0)}° to {(localAugPolicy.rotation_range?.[1] ?? 15).toFixed(0)}°
                  </Label>
                  <p className="text-[11px] text-muted-foreground">
                    Shared session policy: this same range now drives OBB synthetic export and YOLO OBB augmentation in addition to CNN augmentation. Dlib remains capped separately in Python.
                  </p>
                  <Slider
                    value={localAugPolicy.rotation_range ?? [-15, 15]}
                    onValueChange={([lo, hi]) => handleAugPolicyChange({ rotation_range: [lo, hi] })}
                    min={-rotationMax}
                    max={rotationMax}
                    step={1}
                    disabled={isTraining}
                  />
                </div>
                {/* Scale range */}
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Scale range: {(localAugPolicy.scale_range?.[0] ?? 0.9).toFixed(2)}× to {(localAugPolicy.scale_range?.[1] ?? 1.1).toFixed(2)}×
                  </Label>
                  <Slider
                    value={localAugPolicy.scale_range ?? [0.9, 1.1]}
                    onValueChange={([lo, hi]) => handleAugPolicyChange({ scale_range: [lo, hi] })}
                    min={0.7}
                    max={1.3}
                    step={0.01}
                    disabled={isTraining}
                  />
                </div>

                {/* Horizontal flip probability */}
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Horizontal flip: {((localAugPolicy.flip_prob ?? 0.5) * 100).toFixed(0)}%
                  </Label>
                  <Slider
                    value={[localAugPolicy.flip_prob ?? 0.5]}
                    onValueChange={([v]) => handleAugPolicyChange({ flip_prob: v })}
                    min={0}
                    max={1}
                    step={0.05}
                    disabled={isTraining}
                  />
                  {showDirectionalHint && (
                    <p className="text-[10px] text-amber-500">
                      Tip: Set to 0% for strictly directional specimens where left/right carries biological meaning.
                    </p>
                  )}
                </div>

                {/* Vertical flip — invariant schema only, gravity OFF */}
                {showVerticalFlip && (
                  <div className="flex flex-col gap-1">
                    <Label className="text-xs text-muted-foreground">
                      Vertical flip: {((localAugPolicy.vertical_flip_prob ?? 0) * 100).toFixed(0)}%
                    </Label>
                    <Slider
                      value={[localAugPolicy.vertical_flip_prob ?? 0]}
                      onValueChange={([v]) => handleAugPolicyChange({ vertical_flip_prob: v })}
                      min={0}
                      max={1}
                      step={0.05}
                      disabled={isTraining}
                    />
                  </div>
                )}

                {/* Rotate 180° — axial schema only, gravity OFF */}
                {showRotate180 && (
                  <div className="flex flex-col gap-1">
                    <Label className="text-xs text-muted-foreground">
                      Rotate 180°: {((localAugPolicy.rotate_180_prob ?? 0) * 100).toFixed(0)}%
                    </Label>
                    <Slider
                      value={[localAugPolicy.rotate_180_prob ?? 0]}
                      onValueChange={([v]) => handleAugPolicyChange({ rotate_180_prob: v })}
                      min={0}
                      max={1}
                      step={0.05}
                      disabled={isTraining}
                    />
                  </div>
                )}
              </div>
            </details>
          )}

          {(preflightSummary || preflightWarning) && (
            <div className="rounded-md border border-border/70 bg-muted/30 p-3">
              {preflightSummary && (
                <p className="text-[11px] text-muted-foreground">
                  {preflightSummary}
                </p>
              )}
              {preflightWarning && (
                <p className="mt-1 text-[11px] text-amber-600 dark:text-amber-400">
                  {preflightWarning}
                </p>
              )}
            </div>
          )}

          {isTraining && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <p className="font-bold text-foreground">
                  {trainingProgress?.message || "Training in progress..."}
                </p>
                <span className="text-muted-foreground">
                  {Math.max(0, Math.min(100, Math.round(trainingProgress?.percent ?? 0)))}%
                </span>
              </div>
              {trainingProgress?.stage && (
                <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                  Stage: {trainingProgress.stage}
                </p>
              )}
              <Progress
                className="h-2"
                value={Math.max(0, Math.min(100, Math.round(trainingProgress?.percent ?? 0)))}
              />
              {(progressEpochLabel || progressLossLabel || progressLrLabel || progressEtaLabel) && (
                <div className="grid grid-cols-2 gap-1 text-[11px] text-muted-foreground">
                  <span>{progressEpochLabel || " "}</span>
                  <span className="text-right">{progressEtaLabel || " "}</span>
                  <span>{progressLossLabel || " "}</span>
                  <span className="text-right">{progressLrLabel || " "}</span>
                </div>
              )}
            </div>
          )}
            </div>}
            </div>
          </ScrollArea>
        </div>

        {(!showObbStep || obbDetectorReady) && <DialogFooter className="shrink-0 gap-2 px-6 pb-6 pt-2 sm:gap-0">
          <Button
            variant="outline"
            onClick={handleClose}
            disabled={isTraining}
          >
            Cancel
          </Button>
          <Button disabled={!canTrain} onClick={onTrain}>
            {isTraining ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Training...
              </>
            ) : (
              "Train model"
            )}
          </Button>
        </DialogFooter>}

      </DialogContent>
    </Dialog>
  );
};
