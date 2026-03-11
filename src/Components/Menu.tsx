import React, { useEffect, useState } from "react";
import { useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import { useDispatch, useSelector } from "react-redux";
import { Copy, FolderOpen, Loader2, Home, Trash2 } from "lucide-react";
import { toast } from "sonner";

import UploadImages from "./UploadImages";
import type { RootState } from "../state/store";
import store from "../state/store";
import { clearFiles } from "../state/filesState/fileSlice";
import Landmark from "./Landmark";
import { TrainModelDialog } from "./PopUp";
import { DetectionModeSelector, DetectionMode, DetectionPreset } from "./DetectionModeSelector";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { Separator } from "@/Components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/Components/ui/tooltip";
import { sidebarContainer, sidebarItem, buttonHover, buttonTap, cardHover } from "@/lib/animations";

interface MenuProps {
  onOpacityChange: (selectedOpacity: number) => void;
  onColorChange: (selectedColor: string) => void;
  onSwitchChange: () => void;
  onNavigateToLanding?: () => void;
  openTrainDialogOnMount?: boolean;
  onTrainDialogOpened?: () => void;
  // Detection mode
  detectionMode?: DetectionMode;
  onDetectionModeChange?: (mode: DetectionMode) => void;
  detectionPreset?: DetectionPreset;
  onDetectionPresetChange?: (preset: DetectionPreset) => void;
  // Auto mode class name
  className?: string;
  onClassNameChange?: (name: string) => void;
  // SAM2 refinement toggle
  samEnabled?: boolean;
  onSamEnabledChange?: (enabled: boolean) => void;
}

type CnnVariantOption = {
  id: string;
  label: string;
  description: string;
  selectable: boolean;
  recommended?: boolean;
  reason?: string | null;
  recommendationReason?: string | null;
};

const CNN_VARIANT_LIGHT_TO_HEAVY = [
  "mobilenet_v3_large",
  "efficientnet_b0",
  "resnet50",
  "hrnet_w32",
] as const;

const getDatasetSizeBucket = (count: number): "starvation" | "balanced" | "deep" => {
  if (count < 250) return "starvation";
  if (count < 1000) return "balanced";
  return "deep";
};

const getRecommendedCnnVariantId = (bucket: "starvation" | "balanced" | "deep"): string => {
  if (bucket === "starvation") return "mobilenet_v3_large";
  if (bucket === "balanced") return "efficientnet_b0";
  return "resnet50";
};

const getRecommendationReason = (bucket: "starvation" | "balanced" | "deep"): string => {
  if (bucket === "starvation") return "Recommended for small datasets";
  if (bucket === "balanced") return "Recommended for medium datasets";
  return "Recommended for large datasets";
};

const Menu: React.FC<MenuProps> = ({
  onColorChange,
  onOpacityChange,
  onSwitchChange,
  onNavigateToLanding,
  openTrainDialogOnMount,
  onTrainDialogOpened,
  detectionMode = "manual",
  onDetectionModeChange,
  detectionPreset = "balanced",
  onDetectionPresetChange,
  className = "",
  onClassNameChange,
  samEnabled = false,
  onSamEnabledChange,
}) => {
  const dispatch = useDispatch();
  const [openTrainDialog, setOpenTrainDialog] = useState(false);

  // Handle opening train dialog from navigation
  useEffect(() => {
    if (openTrainDialogOnMount) {
      setOpenTrainDialog(true);
      onTrainDialogOpened?.();
    }
  }, [openTrainDialogOnMount, onTrainDialogOpened]);
  const [modelName, setModelName] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<{
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
  } | null>(null);
  const [modelPath, setModelPath] = useState("");
  const [preflightSummary, setPreflightSummary] = useState("");
  const [preflightWarning, setPreflightWarning] = useState("");
  const [predictorType, setPredictorType] = useState<"dlib" | "cnn">("dlib");
  const [cnnVariants, setCnnVariants] = useState<CnnVariantOption[]>([]);
  const [cnnVariant, setCnnVariant] = useState<string>("simplebaseline");
  const [cnnVariantTouched, setCnnVariantTouched] = useState(false);
  const [cnnVariantWarning, setCnnVariantWarning] = useState<string>("");
  const [obbDetectorReady, setObbDetectorReady] = useState(false);
  const [isTrainingObb, setIsTrainingObb] = useState(false);
  const [obbTrainingMessage, setObbTrainingMessage] = useState<string>("");
  const [obbHyperparams, setObbHyperparams] = useState({ iou: 0.3, cls: 1.5, box: 5.0 });
  const [augmentationPolicy, setAugmentationPolicy] = useState<AugmentationPolicy | undefined>(undefined);
  const [orientationMode, setOrientationMode] = useState<string | undefined>(undefined);
  const [sessionImageCountHint, setSessionImageCountHint] = useState(0);

  const activeSpeciesId = useSelector((state: RootState) => state.species.activeSpeciesId);
  const workspaceImageCount = useSelector((state: RootState) => state.files.fileArray.length);
  const finalizedImageCount = useSelector((state: RootState) =>
    state.files.fileArray.filter((img) => img.isFinalized === true).length
  );
  const hasWorkspaceData = workspaceImageCount > 0;

  // showObbStep: unlocks the "Train OBB Detector" step only once the user has
  // finalized at least one image (clicked "Finalize This Image").
  const hasFinalizedBoxes = useSelector((state: RootState) =>
    state.files.fileArray.some((img) => img.isFinalized === true)
  );
  const canTrain = hasFinalizedBoxes && !isTraining;
  const showObbStep = hasFinalizedBoxes;

  useEffect(() => {
    const fetchProjectRoot = async () => {
      try {
        const result = await window.api.getProjectRoot();
        if (result?.projectRoot) setModelPath(result.projectRoot);
      } catch (err) {
        console.error("Failed to load model path", err);
        toast.error("Failed to load model location.");
      }
    };
    fetchProjectRoot();
  }, []);

  // Load augmentationPolicy and orientationMode from session when the train dialog opens
  useEffect(() => {
    if (!openTrainDialog || !activeSpeciesId) return;
    setCnnVariantTouched(false);
    window.api.sessionLoad(activeSpeciesId).then((result) => {
      if (result?.ok) {
        if (result.meta?.augmentationPolicy) {
          setAugmentationPolicy(result.meta.augmentationPolicy as AugmentationPolicy);
        }
        const mode = result.meta?.orientationPolicy?.mode;
        if (typeof mode === "string") setOrientationMode(mode);
        setObbDetectorReady(Boolean(result.meta?.obbDetectorReady));
        setSessionImageCountHint(
          typeof result.meta?.imageCount === "number" ? result.meta.imageCount : 0
        );
      }
    }).catch(() => {/* ignore */});
  }, [openTrainDialog, activeSpeciesId]);

  useEffect(() => {
    const loadCnnVariants = async () => {
      try {
        const result = await window.api.getCnnVariants();
        if (!result?.ok || !Array.isArray(result.variants) || result.variants.length === 0) {
          return;
        }
        const parsed = result.variants as CnnVariantOption[];
        setCnnVariants(parsed);
        const selectable = parsed.filter((v) => v.selectable);
        const requestedDefault = result.defaultVariant;
        const hasRequested =
          !!requestedDefault &&
          selectable.some((v) => v.id === requestedDefault);

        // Python-recommended default (or first selectable as fallback)
        const resolvedDefault = hasRequested
          ? String(requestedDefault)
          : selectable[0]?.id ?? parsed[0]?.id ?? "simplebaseline";
        setCnnVariant(resolvedDefault);
        const warningParts: string[] = [];
        if (result.warning) warningParts.push(String(result.warning));
        if (!result.torchAvailable) {
          warningParts.push("PyTorch is not available.");
        } else if (!result.torchvisionAvailable) {
          warningParts.push("torchvision is not available.");
        } else if (result.device && result.device !== "cuda") {
          warningParts.push(
            `Detected ${String(result.device).toUpperCase()} runtime; heavy CNN variants are system-gated.`
          );
        } else if (
          result.device === "cuda" &&
          typeof result.gpuMemoryGb === "number" &&
          result.gpuMemoryGb < 6
        ) {
          warningParts.push(
            `GPU memory (${result.gpuMemoryGb.toFixed(1)} GB) is below the recommended threshold for high-capacity variants.`
          );
        }
        setCnnVariantWarning(warningParts.join(" "));
      } catch (err) {
        console.warn("Failed to fetch CNN variants:", err);
      }
    };
    void loadCnnVariants();
  }, []);

  const datasetSizeCount = useMemo(() => {
    if (finalizedImageCount > 0) return finalizedImageCount;
    if (sessionImageCountHint > 0) return sessionImageCountHint;
    return workspaceImageCount;
  }, [finalizedImageCount, sessionImageCountHint, workspaceImageCount]);

  const cnnDatasetBucket = useMemo(
    () => getDatasetSizeBucket(datasetSizeCount),
    [datasetSizeCount]
  );

  const recommendedCnnVariantId = useMemo(
    () => getRecommendedCnnVariantId(cnnDatasetBucket),
    [cnnDatasetBucket]
  );

  const recommendedSelectableCnnVariantId = useMemo(() => {
    const preferredIdx = CNN_VARIANT_LIGHT_TO_HEAVY.indexOf(
      recommendedCnnVariantId as (typeof CNN_VARIANT_LIGHT_TO_HEAVY)[number]
    );
    if (preferredIdx >= 0) {
      for (let i = preferredIdx; i >= 0; i -= 1) {
        const fallbackId = CNN_VARIANT_LIGHT_TO_HEAVY[i];
        if (cnnVariants.some((variant) => variant.id === fallbackId && variant.selectable)) {
          return fallbackId;
        }
      }
    }
    return (
      cnnVariants.find((variant) => variant.selectable)?.id ??
      recommendedCnnVariantId
    );
  }, [cnnVariants, recommendedCnnVariantId]);

  const datasetAwareCnnVariants = useMemo(() => {
    const recommendationReason = getRecommendationReason(cnnDatasetBucket);
    return cnnVariants.map((variant) => {
      const isRecommended = variant.id === recommendedSelectableCnnVariantId && variant.selectable;
      let description = variant.description;
      if (variant.id === "hrnet_w32") {
        description = `${variant.description} High-capacity / experimental.`;
      }
      return {
        ...variant,
        recommended: isRecommended,
        recommendationReason: isRecommended ? recommendationReason : null,
        description,
      };
    });
  }, [cnnVariants, cnnDatasetBucket, recommendedSelectableCnnVariantId]);

  const handleCnnVariantChange = useCallback((variantId: string) => {
    setCnnVariantTouched(true);
    setCnnVariant(variantId);
  }, []);

  useEffect(() => {
    if (predictorType !== "cnn" || datasetAwareCnnVariants.length === 0) return;
    const current = datasetAwareCnnVariants.find((v) => v.id === cnnVariant);
    if (!current?.selectable) {
      if (recommendedSelectableCnnVariantId) {
        setCnnVariant(recommendedSelectableCnnVariantId);
      }
      return;
    }
    if (!cnnVariantTouched && current.id !== recommendedSelectableCnnVariantId && recommendedSelectableCnnVariantId) {
      setCnnVariant(recommendedSelectableCnnVariantId);
    }
  }, [
    predictorType,
    datasetAwareCnnVariants,
    cnnVariant,
    cnnVariantTouched,
    recommendedSelectableCnnVariantId,
  ]);

  // Check OBB detector readiness when species or train dialog opens
  useEffect(() => {
    if (!openTrainDialog || !activeSpeciesId) {
      setObbDetectorReady(false);
      return;
    }
    window.api
      .checkModelCompatibility({ speciesId: activeSpeciesId, modelName: modelName || "_preflight" })
      .then((r) => setObbDetectorReady(r?.obbDetectorReady ?? false))
      .catch(() => setObbDetectorReady(false));
  }, [activeSpeciesId, openTrainDialog, modelName]);

  const handleTrainObbDetector = async () => {
    if (!activeSpeciesId || isTrainingObb) return;
    setIsTrainingObb(true);
    setObbTrainingMessage("Exporting OBB dataset and starting training…");
    try {
      const result = await window.api.trainObbDetector(activeSpeciesId, {
        iou: obbHyperparams.iou,
        cls: obbHyperparams.cls,
        box: obbHyperparams.box,
      });
      if (result?.ok) {
        setObbDetectorReady(true);
        const mapStr = typeof result.map50 === "number" ? ` (mAP50: ${result.map50.toFixed(3)})` : "";
        const warningText = Array.isArray(result.warnings) && result.warnings.length > 0
          ? ` Warnings: ${result.warnings.join(" ")}`
          : "";
        setObbTrainingMessage(`OBB detector trained${mapStr} — ready.${warningText}`);
      } else {
        setObbTrainingMessage(`OBB training failed: ${result?.error ?? "unknown error"}`);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setObbTrainingMessage(`OBB training error: ${message}`);
    } finally {
      setIsTrainingObb(false);
    }
  };

  const handleSelectModelPath = async () => {
    try {
      const result = await window.api.selectProjectRoot();
      if (!result?.canceled && result?.projectRoot) {
        setModelPath(result.projectRoot);
        toast.success("Model location updated.");
      }
    } catch (err) {
      console.error("Failed to select model path", err);
      toast.error("Failed to select model location.");
    }
  };

  const handleCopyPath = async () => {
    if (!modelPath) return;
    try {
      await navigator.clipboard.writeText(modelPath);
      toast.success("Path copied.");
    } catch (err) {
      console.error("Clipboard copy failed", err);
      toast.error("Could not copy path.");
    }
  };

  const handleOpenFolder = async () => {
    if (!modelPath) return;
    try {
      // @ts-expect-error - API may not exist in preload yet
      if (window.api?.openPath) await window.api.openPath(modelPath);
      else toast.info("Open folder is not implemented yet.");
    } catch (err) {
      console.error("Failed to open folder", err);
      toast.error("Could not open folder.");
    }
  };

  const runPreflight = async (autosaveWorkspace: boolean) => {
    const name = modelName.trim();
    if (!name) {
      setPreflightSummary("Enter a model name to run preflight.");
      setPreflightWarning("");
      return null;
    }

    const latestFileArray = store.getState().files.fileArray;

    if (autosaveWorkspace && !activeSpeciesId && hasWorkspaceData) {
      await window.api.saveLabels(latestFileArray);
    }

    const preflight = await window.api.trainingPreflight({
      speciesId: activeSpeciesId ?? undefined,
      modelName: name,
      workspaceImages: latestFileArray.length,
    });

    if (!preflight.ok) {
      throw new Error(preflight.error || "Preflight failed.");
    }

    setPreflightSummary(
      `Preflight: ${preflight.totalTrainableImages ?? 0} trainable image(s) - Landmarks: ${preflight.landmarkMessage}`
    );

    const warningText = (preflight.warnings || []).join(" | ");
    setPreflightWarning(warningText);
    return preflight;
  };
  const handleTrainConfirm = async () => {
    const name = modelName.trim();
    if (!name) return;

    let unsubscribeTrainProgress: (() => void) | null = null;
    try {
      setIsTraining(true);
      setTrainingProgress({
        percent: 0,
        stage: "preflight",
        message: "Starting training...",
        predictorType,
        modelName: name,
      });

      unsubscribeTrainProgress = window.api.onTrainProgress((data) => {
        if (data.modelName !== name) return;
        if (data.predictorType !== predictorType) return;
        setTrainingProgress((prev) => {
          const prevPct = Number(prev?.percent ?? 0);
          const nextPct = Number(data?.percent ?? 0);
          const monotonicPct = Math.max(
            0,
            Math.min(100, Math.max(prevPct, nextPct))
          );
          return {
            ...data,
            percent: monotonicPct,
          };
        });
      });

      const preflight = await runPreflight(true);
      if (!preflight) return;

      toast.info("Training from session data...");
      const result = await window.api.trainModel(name, {
        speciesId: activeSpeciesId ?? undefined,
        predictorType,
        cnnVariant: predictorType === "cnn" ? cnnVariant : undefined,
      });
      if (!result.ok) throw new Error(result.error);
      console.log("Training output:", result.output);
      toast.success("Training complete.");

      setOpenTrainDialog(false);
    } catch (err) {
      console.error(err);
      toast.error(`Training failed. ${String(err)}`);
    } finally {
      unsubscribeTrainProgress?.();
      setIsTraining(false);
      setTrainingProgress(null);
    }
  };
  useEffect(() => {
    if (!openTrainDialog) return;
    runPreflight(false).catch((err) => {
      setPreflightWarning(String(err));
    });
    // runPreflight is recreated on render but captures the same vars already listed as deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openTrainDialog, modelName, activeSpeciesId, hasWorkspaceData, predictorType]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === "n") {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent("open-upload-dialog"));
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    const openUploadDialog = () => {
      document.getElementById("btn-upload")?.click();
    };
    window.addEventListener("open-upload-dialog", openUploadDialog);
    return () =>
      window.removeEventListener("open-upload-dialog", openUploadDialog);
  }, []);

  return (
      <div className="flex h-screen w-full flex-col overflow-hidden bg-background">
        <TrainModelDialog
          handleTrainConfirm={handleTrainConfirm}
          open={openTrainDialog}
          setOpen={setOpenTrainDialog}
          modelName={modelName}
          isTraining={isTraining}
          setModelName={setModelName}
          preflightSummary={preflightSummary}
          preflightWarning={preflightWarning}
          predictorType={predictorType}
          setPredictorType={setPredictorType}
          cnnVariants={datasetAwareCnnVariants}
          cnnVariant={cnnVariant}
          setCnnVariant={handleCnnVariantChange}
          cnnVariantWarning={cnnVariantWarning}
          trainingProgress={trainingProgress}
          obbDetectorReady={obbDetectorReady}
          showObbStep={showObbStep}
          isTrainingObb={isTrainingObb}
          handleTrainObbDetector={handleTrainObbDetector}
          obbTrainingMessage={obbTrainingMessage}
          obbHyperparams={obbHyperparams}
          onObbHyperparamsChange={setObbHyperparams}
          speciesId={activeSpeciesId ?? undefined}
          augmentationPolicy={augmentationPolicy}
          onAugmentationPolicyChange={(policy) => {
            setAugmentationPolicy(policy);
            if (activeSpeciesId) {
              window.api.sessionUpdateAugmentation(activeSpeciesId, policy).catch(() => {/* ignore */});
            }
          }}
          orientationMode={orientationMode}
        />

        <div className="flex-1 overflow-y-auto">
          <motion.div
            variants={sidebarContainer}
            initial="hidden"
            animate="visible"
            className="flex flex-col gap-4 p-4"
          >
            {/* Header */}
            <motion.div variants={sidebarItem}>
              <div className="flex items-center gap-2">
                {onNavigateToLanding && (
                  <motion.div {...buttonHover} {...buttonTap}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="inline-flex">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={onNavigateToLanding}
                            className="shrink-0"
                          >
                            <Home className="h-5 w-5" />
                          </Button>
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="right">
                        Back to Home
                      </TooltipContent>
                    </Tooltip>
                  </motion.div>
                )}
                <div className="flex-1 text-center">
                  <h1 className="text-xl font-bold text-foreground">
                    Auto Landmarking
                  </h1>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Select Model - Import - Annotate
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Model Card */}
            <motion.div variants={sidebarItem}>
              <motion.div variants={cardHover} initial="initial" whileHover="hover">
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                      Model
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-semibold text-foreground">
                          Model location
                        </p>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <p className="truncate font-mono text-xs text-muted-foreground">
                              {modelPath || "Loading..."}
                            </p>
                          </TooltipTrigger>
                          <TooltipContent side="bottom" align="start">
                            {modelPath || "Loading..."}
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <motion.div {...buttonHover} {...buttonTap}>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleSelectModelPath}
                          className="shrink-0 font-semibold"
                        >
                          Browse...
                        </Button>
                      </motion.div>
                    </div>

                    <div className="flex gap-2">
                      <motion.div {...buttonHover} {...buttonTap}>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={handleCopyPath}
                          disabled={!modelPath}
                          className="text-xs font-semibold text-primary"
                        >
                          <Copy className="mr-1.5 h-3 w-3" />
                          Copy path
                        </Button>
                      </motion.div>
                      <motion.div {...buttonHover} {...buttonTap}>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={handleOpenFolder}
                          disabled={!modelPath}
                          className="text-xs font-semibold text-primary"
                        >
                          <FolderOpen className="mr-1.5 h-3 w-3" />
                          Open folder
                        </Button>
                      </motion.div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>

            {/* Image Upload Card */}
            <motion.div variants={sidebarItem}>
              <motion.div variants={cardHover} initial="initial" whileHover="hover">
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                      Image Upload
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <UploadImages />
                    <Separator />
                    <div className="flex items-center justify-between">
                      <p className="text-xs text-muted-foreground">
                        {workspaceImageCount
                          ? `${workspaceImageCount} image(s) loaded`
                          : "No images loaded"}
                      </p>
                      {workspaceImageCount > 0 && (
                        <motion.div {...buttonHover} {...buttonTap}>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={async () => {
                              if (activeSpeciesId) {
                                await window.api.sessionDeleteAllImages(activeSpeciesId);
                              }
                              dispatch(clearFiles());
                              toast.success("All images removed.");
                            }}
                            className="h-7 text-xs font-bold text-destructive hover:bg-destructive/10 hover:text-destructive"
                          >
                            <Trash2 className="mr-1 h-3 w-3" />
                            Delete all
                          </Button>
                        </motion.div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>

            {/* Landmark Controls */}
            <motion.div variants={sidebarItem}>
              <Landmark
                onOpacityChange={onOpacityChange}
                onColorChange={onColorChange}
                onSwitchChange={onSwitchChange}
              />
            </motion.div>

            {/* Detection Mode */}
            {onDetectionModeChange && (
              <motion.div variants={sidebarItem}>
                <motion.div variants={cardHover} initial="initial" whileHover="hover">
                  <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                        Detection Mode
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <DetectionModeSelector
                        mode={detectionMode}
                        onModeChange={onDetectionModeChange}
                        detectionPreset={detectionPreset}
                        onDetectionPresetChange={onDetectionPresetChange}
                        className={className}
                        onClassNameChange={onClassNameChange}
                        samEnabled={samEnabled}
                        onSamEnabledChange={onSamEnabledChange}
                      />
                    </CardContent>
                  </Card>
                </motion.div>
              </motion.div>
            )}

            {/* Training Card */}
            <motion.div variants={sidebarItem}>
              <motion.div variants={cardHover} initial="initial" whileHover="hover">
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                      Training
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p
                      className={cn(
                        "text-xs",
                        "text-muted-foreground"
                      )}
                    >
                      {canTrain
                        ? "Ready to train from current session data."
                        : "Annotate images or import a pre-annotated folder to enable training."}
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>

            {/* Spacer for footer */}
            <div className="h-24" />
          </motion.div>
        </div>

        {/* Sticky footer */}
        <div className="border-t bg-background p-4">
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              className="w-full font-bold"
              disabled={!canTrain}
              onClick={() => setOpenTrainDialog(true)}
            >
              {isTraining ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Training...
                </>
              ) : (
                "Train model"
              )}
            </Button>
          </motion.div>
        </div>
      </div>
  );
};

export default Menu;
