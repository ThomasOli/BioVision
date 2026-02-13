import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useSelector } from "react-redux";
import { Copy, FolderOpen, Loader2, Home } from "lucide-react";
import { toast } from "sonner";

import UploadImages from "./UploadImages";
import type { RootState } from "../state/store";
import Landmark from "./Landmark";
import { TrainModelDialog } from "./PopUp";
import { DetectionModeSelector, DetectionMode } from "./DetectionModeSelector";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { Separator } from "@/Components/ui/separator";
import { ScrollArea } from "@/Components/ui/scroll-area";
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
  // Multi-specimen detection
  detectionMode?: DetectionMode;
  onDetectionModeChange?: (mode: DetectionMode) => void;
  confThreshold?: number;
  onConfThresholdChange?: (value: number) => void;
}

const Menu: React.FC<MenuProps> = ({
  onColorChange,
  onOpacityChange,
  onSwitchChange,
  onNavigateToLanding,
  openTrainDialogOnMount,
  onTrainDialogOpened,
  detectionMode = "single",
  onDetectionModeChange,
  confThreshold = 0.25,
  onConfThresholdChange,
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
  const [isTraining, setIsTraining] = useState(false);
  const [modelPath, setModelPath] = useState("");

  const fileArray = useSelector((state: RootState) => state.files.fileArray);
  const activeSpeciesId = useSelector((state: RootState) => state.species.activeSpeciesId);
  const canTrain = useMemo(
    () => (fileArray?.length ?? 0) > 0 && !isTraining,
    [fileArray, isTraining]
  );

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

  const handleTrainConfirm = async () => {
    const name = modelName.trim();
    if (!name) return;

    try {
      setIsTraining(true);

      if (activeSpeciesId) {
        // Session-scoped: auto-save already persisted annotations to session dir
        toast.info("Training model from session...");
        const result = await window.api.trainModel(name, { speciesId: activeSpeciesId });
        if (!result.ok) throw new Error(result.error);
        console.log("Training output:", result.output);
      } else {
        // Legacy fallback: save labels to flat directory and train
        toast.info("Saving labels...");
        await window.api.saveLabels(fileArray);
        toast.info("Training model...");
        const result = await window.api.trainModel(name);
        if (!result.ok) throw new Error(result.error);
        console.log("Training output:", result.output);
      }

      setOpenTrainDialog(false);
      setModelName("");
      toast.success("Training complete.");
    } catch (err) {
      console.error(err);
      toast.error(`Training failed. ${String(err)}`);
    } finally {
      setIsTraining(false);
    }
  };

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
            {onDetectionModeChange && onConfThresholdChange && (
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
                        confThreshold={confThreshold}
                        onConfThresholdChange={onConfThresholdChange}
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
                        canTrain ? "text-muted-foreground" : "text-destructive"
                      )}
                    >
                      {canTrain
                        ? "Ready to train."
                        : "Add images to enable training."}
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
