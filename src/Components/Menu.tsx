import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useSelector } from "react-redux";
import { Copy, FolderOpen, Loader2, Home } from "lucide-react";
import { toast } from "sonner";

import UploadImages from "./UploadImages";
import type { RootState } from "../state/store";
import Landmark from "./Landmark";
import { TrainModelDialog } from "./PopUp";
import { DetectionModeSelector, DetectionMode, DetectionPreset } from "./DetectionModeSelector";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { Input } from "@/Components/ui/input";
import { Separator } from "@/Components/ui/separator";
import { ScrollArea } from "@/Components/ui/scroll-area";
import { Switch } from "@/Components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
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
  autoConfidence?: number;
  onAutoConfidenceChange?: (value: number) => void;
  detectionPreset?: DetectionPreset;
  onDetectionPresetChange?: (preset: DetectionPreset) => void;
  // Auto mode class name
  className?: string;
  onClassNameChange?: (name: string) => void;
  // SAM2 refinement toggle
  samEnabled?: boolean;
  onSamEnabledChange?: (enabled: boolean) => void;
}

const Menu: React.FC<MenuProps> = ({
  onColorChange,
  onOpacityChange,
  onSwitchChange,
  onNavigateToLanding,
  openTrainDialogOnMount,
  onTrainDialogOpened,
  detectionMode = "manual",
  onDetectionModeChange,
  autoConfidence = 0.5,
  onAutoConfidenceChange,
  detectionPreset = "balanced",
  onDetectionPresetChange,
  className = "",
  onClassNameChange,
  samEnabled = false,
  onSamEnabledChange,
}) => {
  const [openTrainDialog, setOpenTrainDialog] = useState(false);

  // Handle opening train dialog from navigation
  useEffect(() => {
    if (openTrainDialogOnMount) {
      setOpenTrainDialog(true);
      onTrainDialogOpened?.();
    }
  }, [openTrainDialogOnMount, onTrainDialogOpened]);
  const [modelName, setModelName] = useState("");
  const [xmlImportModelName, setXmlImportModelName] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [modelPath, setModelPath] = useState("");
  const [showAdvancedImport, setShowAdvancedImport] = useState(false);
  const [useImportedXmlForTraining, setUseImportedXmlForTraining] = useState(false);
  const [hasImportedPreAnnotated, setHasImportedPreAnnotated] = useState(false);
  const [importedPreAnnotatedCount, setImportedPreAnnotatedCount] = useState(0);
  const [importedXmlModelName, setImportedXmlModelName] = useState<string | null>(null);
  const [preAnnotatedStatus, setPreAnnotatedStatus] = useState("");
  const [dlibXmlStatus, setDlibXmlStatus] = useState("");
  const [preflightSummary, setPreflightSummary] = useState("");
  const [preflightWarning, setPreflightWarning] = useState("");

  const fileArray = useSelector((state: RootState) => state.files.fileArray);
  const activeSpeciesId = useSelector((state: RootState) => state.species.activeSpeciesId);
  const hasWorkspaceData = (fileArray?.length ?? 0) > 0;
  const hasImportedXml = Boolean(importedXmlModelName);
  const canUseImportedXml =
    hasImportedXml && modelName.trim().length > 0 && importedXmlModelName === modelName.trim();
  const canTrain = (useImportedXmlForTraining ? canUseImportedXml : (hasWorkspaceData || hasImportedPreAnnotated)) && !isTraining;

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

  const handleImportPreAnnotated = async () => {
    try {
      const result = await window.api.importPreAnnotatedDataset({
        speciesId: activeSpeciesId ?? undefined,
      });
      if (result.canceled) return;
      if (!result.ok) throw new Error(result.error || "Import failed.");

      const warningSuffix = result.warnings?.length
        ? ` Warnings: ${result.warnings.join(" | ")}`
        : "";
      const status = `Imported ${result.importedImages ?? 0} image(s), ${result.importedLabels ?? 0} label file(s).${warningSuffix}`;
      setHasImportedPreAnnotated(true);
      setImportedPreAnnotatedCount(result.importedImages ?? 0);
      setPreAnnotatedStatus(status);

      if (result.warnings && result.warnings.length > 0) {
        toast.warning(status);
      } else {
        toast.success(status);
      }
    } catch (err) {
      console.error(err);
      toast.error(`Pre-annotated import failed. ${String(err)}`);
    }
  };

  const handleImportDlibXml = async (currentModelName: string) => {
    try {
      const name = currentModelName.trim();
      if (!name) {
        toast.error("Enter a model name before importing dlib XML.");
        return;
      }

      const result = await window.api.importDlibXml({
        modelName: name,
        speciesId: activeSpeciesId ?? undefined,
      });
      if (result.canceled) return;
      if (!result.ok) throw new Error(result.error || "dlib XML import failed.");

      const trainStats = result.trainStats
        ? `${result.trainStats.num_images} images, ${result.trainStats.num_boxes} boxes`
        : "train XML imported";
      const testStats = result.testStats
        ? ` Test: ${result.testStats.num_images} images, ${result.testStats.num_boxes} boxes.`
        : " No test XML provided.";
      const warningSuffix = result.warnings?.length
        ? ` Warnings: ${result.warnings.join(" | ")}`
        : "";

      const status = `XML imported for "${name}". Train: ${trainStats}.${testStats}${warningSuffix}`;
      setImportedXmlModelName(name);
      setModelName(name);
      setShowAdvancedImport(true);
      setDlibXmlStatus(status);

      if (result.warnings && result.warnings.length > 0) {
        toast.warning(status);
      } else {
        toast.success(status);
      }
    } catch (err) {
      console.error(err);
      toast.error(`dlib XML import failed. ${String(err)}`);
    }
  };

  const runPreflight = async (autosaveWorkspace: boolean) => {
    const name = modelName.trim();
    if (!name) {
      setPreflightSummary("Enter a model name to run preflight.");
      setPreflightWarning("");
      return null;
    }

    if (autosaveWorkspace && !useImportedXmlForTraining && !activeSpeciesId && hasWorkspaceData) {
      await window.api.saveLabels(fileArray);
    }

    const preflight = await window.api.trainingPreflight({
      speciesId: activeSpeciesId ?? undefined,
      modelName: name,
      useImportedXml: useImportedXmlForTraining,
      workspaceImages: fileArray.length,
      importedImagesHint: importedPreAnnotatedCount,
    });

    if (!preflight.ok) {
      throw new Error(preflight.error || "Preflight failed.");
    }

    if (preflight.useImportedXml) {
      setPreflightSummary(
        `Preflight: Train XML ${preflight.trainXmlImages ?? 0} image(s)` +
        `${preflight.testXmlImages !== undefined ? ` • Test XML ${preflight.testXmlImages} image(s)` : ""}` +
        ` • Landmark IDs: ${preflight.landmarkMessage}`
      );
    } else {
      setPreflightSummary(
        `Preflight: Workspace ${preflight.workspaceImages ?? fileArray.length} • Imported ${preflight.importedImages ?? importedPreAnnotatedCount} • Total trainable ${preflight.totalTrainableImages ?? 0} • Landmark IDs: ${preflight.landmarkMessage}`
      );
    }

    const warningText = (preflight.warnings || []).join(" | ");
    setPreflightWarning(warningText);
    return preflight;
  };

  const handleTrainConfirm = async () => {
    const name = modelName.trim();
    if (!name) return;

    try {
      setIsTraining(true);
      const preflight = await runPreflight(true);
      if (!preflight) return;

      if (useImportedXmlForTraining) {
        if (!canUseImportedXml) {
          throw new Error("Import dlib XML for this exact model name before training.");
        }
        toast.info("Training from imported dlib XML...");
        const result = await window.api.trainModel(name, {
          speciesId: activeSpeciesId ?? undefined,
          useImportedXml: true,
        });
        if (!result.ok) throw new Error(result.error);
        console.log("Training output:", result.output);
      } else {
        toast.info("Training from combined app data...");
        const result = await window.api.trainModel(name, {
          speciesId: activeSpeciesId ?? undefined,
        });
        if (!result.ok) throw new Error(result.error);
        console.log("Training output:", result.output);
      }

      setOpenTrainDialog(false);
      toast.success("Training complete.");
    } catch (err) {
      console.error(err);
      toast.error(`Training failed. ${String(err)}`);
    } finally {
      setIsTraining(false);
    }
  };

  useEffect(() => {
    if (!openTrainDialog) return;
    runPreflight(false).catch((err) => {
      setPreflightWarning(String(err));
    });
  }, [openTrainDialog, modelName, useImportedXmlForTraining, activeSpeciesId, importedPreAnnotatedCount, hasWorkspaceData]);

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
    <TooltipProvider>
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
        />

        <ScrollArea className="flex-1">
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
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={onNavigateToLanding}
                          className="shrink-0"
                        >
                          <Home className="h-5 w-5" />
                        </Button>
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
                    Select Model • Import • Annotate
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
                    <p className="text-xs text-muted-foreground">
                      {fileArray?.length
                        ? `${fileArray.length} image(s) loaded`
                        : "No images loaded"}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="text-xs"
                        onClick={handleImportPreAnnotated}
                        disabled={isTraining}
                      >
                        Import pre-annotated
                      </Button>
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                      {preAnnotatedStatus || "Pre-annotated import: not imported."}
                    </p>
                    <div className="rounded-md border border-border/70 p-2">
                      <div className="flex items-center justify-between">
                        <p className="text-[11px] font-medium text-foreground">
                          Advanced
                        </p>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-[11px]"
                          onClick={() => setShowAdvancedImport((v) => !v)}
                          disabled={isTraining}
                        >
                          {showAdvancedImport ? "Hide" : "Show"}
                        </Button>
                      </div>
                      {showAdvancedImport && (
                        <div className="mt-2 space-y-2">
                          <Input
                            value={xmlImportModelName}
                            onChange={(e) => setXmlImportModelName(e.target.value)}
                            placeholder="Model name for XML import"
                            disabled={isTraining}
                            className="h-8 text-xs"
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            className="text-xs"
                            onClick={() => handleImportDlibXml(xmlImportModelName)}
                            disabled={isTraining}
                          >
                            Import dlib XML (Advanced)
                          </Button>
                          <p className="text-[11px] text-muted-foreground">
                            {dlibXmlStatus || "Import dlib XML only if you already have train/test XML ready."}
                          </p>
                          <div className="flex items-center justify-between">
                            <p className="text-[11px] text-muted-foreground">
                              Train directly from imported XML
                            </p>
                            <Switch
                              checked={useImportedXmlForTraining}
                              onCheckedChange={setUseImportedXmlForTraining}
                              disabled={isTraining || !canUseImportedXml}
                            />
                          </div>
                          {!canUseImportedXml && (
                            <p className="text-[11px] text-muted-foreground">
                              Import XML for this exact model name to enable XML training.
                            </p>
                          )}
                        </div>
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
            {onDetectionModeChange && onAutoConfidenceChange && (
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
                        autoConfidence={autoConfidence}
                        onAutoConfidenceChange={onAutoConfidenceChange}
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
                        ? `Ready to train (${useImportedXmlForTraining ? "imported XML (advanced)" : "combined app data"}).`
                        : useImportedXmlForTraining
                          ? "XML mode selected: import dlib XML for this model name first."
                          : "Import pre-annotated data and/or annotate images in app to train."}
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>

            {/* Spacer for footer */}
            <div className="h-24" />
          </motion.div>
        </ScrollArea>

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
    </TooltipProvider>
  );
};

export default Menu;
