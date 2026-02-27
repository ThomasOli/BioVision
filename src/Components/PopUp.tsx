import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Info, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Input } from "@/Components/ui/input";
import { Label } from "@/Components/ui/label";
import { Progress } from "@/Components/ui/progress";
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
  cnnVariants?: {
    id: string;
    label: string;
    description: string;
    selectable: boolean;
    recommended?: boolean;
    reason?: string | null;
  }[];
  cnnVariant?: string;
  setCnnVariant?: (variant: string) => void;
  cnnVariantWarning?: string;
  skipParity?: boolean;
  setSkipParity?: (value: boolean) => void;
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
  skipParity = false,
  setSkipParity,
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
  const progressParityLabel = useMemo(() => {
    if (!progressDetails) return null;
    const done = Number(progressDetails.records_done);
    const total = Number(progressDetails.records_total);
    const split = typeof progressDetails.split === "string" ? progressDetails.split : "";
    const mode = typeof progressDetails.eval_mode === "string" ? progressDetails.eval_mode : "";
    if (Number.isFinite(done) && Number.isFinite(total) && total > 0) {
      const scope = [split, mode].filter(Boolean).join("/");
      return `${scope || "parity"} ${Math.round(done)}/${Math.round(total)}`;
    }
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

  return (
    <Dialog open={open} onOpenChange={(value) => !isTraining && setOpen(value)}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-sm font-bold">
            Train new model
          </DialogTitle>
          <DialogDescription className="text-xs">
            Give your model a clear, versioned name (Ctrl/Cmd+Enter to start).
          </DialogDescription>
        </DialogHeader>

        {/* Step 1: OBB Detector — shown when session has OBB annotations or detector already trained */}
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
          </div>
        )}
        {/* Step 2 label when OBB step is shown */}
        {showObbStep && (
          <p className="text-[11px] font-semibold text-muted-foreground px-1">
            Step 2: Train Landmark Predictor
          </p>
        )}

        <div className="space-y-4 py-4">
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
                      Fast training - best for standardized, controlled-conditions images (mean shape predictor)
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
                    <p className="text-xs font-semibold">CNN (SimpleBaseline default)</p>
                    <p className="text-[11px] text-muted-foreground">
                      Slower training - robust to varied lighting and orientations - requires PyTorch
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
                    <p className="text-[11px] text-muted-foreground">
                      {selectedCnnVariant.description}
                    </p>
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

          {setSkipParity && (
            <div className="space-y-2">
              <Label className="text-sm font-medium">Evaluation options</Label>
              <label className="flex items-start gap-3 rounded-md border border-border/50 p-3">
                <input
                  type="checkbox"
                  checked={skipParity}
                  onChange={(e) => setSkipParity(e.target.checked)}
                  disabled={isTraining}
                  className="mt-0.5 accent-primary"
                />
                <div>
                  <p className="text-xs font-semibold">Skip pipeline parity (faster)</p>
                  <p className="text-[11px] text-muted-foreground">
                    Trains and saves model without running GT/detected parity checks.
                  </p>
                </div>
              </label>
            </div>
          )}

          {/* Augmentation & Schema Studio — only shown when a session is active and CNN is selected */}
          {speciesId && predictorType === "cnn" && (
            <details className="rounded-md border border-border/60 bg-muted/20">
              <summary className="cursor-pointer px-3 py-2 text-xs font-semibold select-none">
                Augmentation &amp; Schema Studio
              </summary>
              <div className="flex flex-col gap-3 px-3 pb-3 pt-1">
                {/* Gravity-aligned toggle */}
                <div className="flex items-center justify-between">
                  <div>
                    <Label className="text-xs font-medium">Gravity-aligned imaging</Label>
                    <p className="text-[11px] text-muted-foreground">
                      Locks synthetic rotation to ±15° (for specimens always upright)
                    </p>
                  </div>
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
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    CNN rotation: {(localAugPolicy.rotation_range?.[0] ?? -15).toFixed(0)}° to {(localAugPolicy.rotation_range?.[1] ?? 15).toFixed(0)}°
                  </Label>
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
              {(progressEpochLabel || progressLossLabel || progressLrLabel || progressEtaLabel || progressParityLabel) && (
                <div className="grid grid-cols-2 gap-1 text-[11px] text-muted-foreground">
                  <span>{progressEpochLabel || " "}</span>
                  <span className="text-right">{progressEtaLabel || " "}</span>
                  <span>{progressLossLabel || " "}</span>
                  <span className="text-right">{progressLrLabel || " "}</span>
                  {progressParityLabel && <span className="col-span-2">{progressParityLabel}</span>}
                </div>
              )}
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
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
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
