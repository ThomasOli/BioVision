import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  Database,
  Trash2,
  Play,
  Edit3,
  RefreshCw,
  HardDrive,
  Calendar,
  MoreVertical,
  Check,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { Input } from "@/Components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/Components/ui/popover";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/Components/ui/dialog";
import { staggerContainer, staggerItem, buttonHover, buttonTap, cardHover, modalContent } from "@/lib/animations";
import { TrainedModel, AppView } from "@/types/Image";

interface MyModelsPageProps {
  onNavigate: (view: AppView) => void;
  onSelectModelForInference: (modelName: string) => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(date: Date): string {
  return new Date(date).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

interface ModelCardProps {
  model: TrainedModel;
  onUse: () => void;
  onDelete: () => void;
  onRename: (newName: string) => void;
}

const ModelCard: React.FC<ModelCardProps> = ({ model, onUse, onDelete, onRename }) => {
  const [isRenaming, setIsRenaming] = useState(false);
  const [newName, setNewName] = useState(model.name);
  const [menuOpen, setMenuOpen] = useState(false);

  const handleRenameSubmit = () => {
    if (newName.trim() && newName !== model.name) {
      onRename(newName.trim());
    }
    setIsRenaming(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleRenameSubmit();
    } else if (e.key === "Escape") {
      setNewName(model.name);
      setIsRenaming(false);
    }
  };

  return (
    <motion.div variants={staggerItem}>
      <motion.div
        variants={cardHover}
        initial="initial"
        whileHover="hover"
        className="h-full"
      >
        <Card className="h-full border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <div className="flex items-start justify-between">
              {isRenaming ? (
                <div className="flex flex-1 items-center gap-2">
                  <Input
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    onKeyDown={handleKeyDown}
                    autoFocus
                    className="h-8 text-sm font-semibold"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={handleRenameSubmit}
                  >
                    <Check className="h-4 w-4 text-green-500" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => {
                      setNewName(model.name);
                      setIsRenaming(false);
                    }}
                  >
                    <X className="h-4 w-4 text-destructive" />
                  </Button>
                </div>
              ) : (
                <>
                  <CardTitle className="text-sm font-semibold">{model.name}</CardTitle>
                  <Popover open={menuOpen} onOpenChange={setMenuOpen}>
                    <PopoverTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent align="end" className="w-40 p-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="w-full justify-start"
                        onClick={() => {
                          setMenuOpen(false);
                          setIsRenaming(true);
                        }}
                      >
                        <Edit3 className="mr-2 h-4 w-4" />
                        Rename
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="w-full justify-start text-destructive hover:text-destructive"
                        onClick={() => {
                          setMenuOpen(false);
                          onDelete();
                        }}
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete
                      </Button>
                    </PopoverContent>
                  </Popover>
                </>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <HardDrive className="h-3 w-3" />
                <span>{formatFileSize(model.size)}</span>
              </div>
              <div className="flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                <span>{formatDate(model.createdAt)}</span>
              </div>
            </div>
            <motion.div {...buttonHover} {...buttonTap}>
              <Button
                size="sm"
                className="w-full"
                onClick={onUse}
              >
                <Play className="mr-2 h-4 w-4" />
                Use for Inference
              </Button>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
};

export const MyModelsPage: React.FC<MyModelsPageProps> = ({
  onNavigate,
  onSelectModelForInference,
}) => {
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteDialog, setDeleteDialog] = useState<{ open: boolean; model: TrainedModel | null }>({
    open: false,
    model: null,
  });

  const loadModels = async () => {
    setLoading(true);
    try {
      const result = await window.api.listModels();
      if (result.ok && result.models) {
        setModels(result.models);
      } else {
        toast.error(result.error || "Failed to load models");
      }
    } catch (err) {
      console.error("Failed to load models:", err);
      toast.error("Failed to load models");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleUseModel = (model: TrainedModel) => {
    onSelectModelForInference(model.name);
    onNavigate("inference");
  };

  const handleDeleteModel = async () => {
    if (!deleteDialog.model) return;

    try {
      const result = await window.api.deleteModel(deleteDialog.model.name);
      if (result.ok) {
        toast.success("Model deleted");
        loadModels();
      } else {
        toast.error(result.error || "Failed to delete model");
      }
    } catch (err) {
      console.error("Failed to delete model:", err);
      toast.error("Failed to delete model");
    } finally {
      setDeleteDialog({ open: false, model: null });
    }
  };

  const handleRenameModel = async (oldName: string, newName: string) => {
    try {
      const result = await window.api.renameModel(oldName, newName);
      if (result.ok) {
        toast.success("Model renamed");
        loadModels();
      } else {
        toast.error(result.error || "Failed to rename model");
      }
    } catch (err) {
      console.error("Failed to rename model:", err);
      toast.error("Failed to rename model");
    }
  };

  return (
    <>
      <div className="flex h-screen w-screen flex-col bg-background">
        {/* Header */}
        <div className="flex items-center justify-between border-b p-4">
          <div className="flex items-center gap-4">
            <motion.div {...buttonHover} {...buttonTap}>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => onNavigate("landing")}
              >
                <ArrowLeft className="h-5 w-5" />
              </Button>
            </motion.div>
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-primary" />
              <h1 className="text-lg font-bold">My Models</h1>
            </div>
          </div>
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              variant="outline"
              size="sm"
              onClick={loadModels}
              disabled={loading}
            >
              <RefreshCw className={cn("mr-2 h-4 w-4", loading && "animate-spin")} />
              Refresh
            </Button>
          </motion.div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {loading ? (
            <div className="flex h-full items-center justify-center">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : models.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <Database className="mb-4 h-16 w-16 text-muted-foreground/50" />
              <h2 className="text-lg font-semibold text-foreground">No models yet</h2>
              <p className="mt-2 max-w-sm text-sm text-muted-foreground">
                Train a model by annotating images and clicking "Train Model" from the
                annotation workspace.
              </p>
              <motion.div {...buttonHover} {...buttonTap} className="mt-6">
                <Button onClick={() => onNavigate("workspace")}>
                  Start Annotating
                </Button>
              </motion.div>
            </div>
          ) : (
            <motion.div
              variants={staggerContainer}
              initial="initial"
              animate="animate"
              className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
            >
              {models.map((model) => (
                <ModelCard
                  key={model.name}
                  model={model}
                  onUse={() => handleUseModel(model)}
                  onDelete={() => setDeleteDialog({ open: true, model })}
                  onRename={(newName) => handleRenameModel(model.name, newName)}
                />
              ))}
            </motion.div>
          )}
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialog.open}
        onOpenChange={(open) => setDeleteDialog({ open, model: open ? deleteDialog.model : null })}
      >
        <DialogContent asChild>
          <motion.div
            variants={modalContent}
            initial="initial"
            animate="animate"
            exit="exit"
            className="sm:max-w-md"
          >
            <DialogHeader>
              <DialogTitle className="text-sm font-bold">Delete Model</DialogTitle>
              <DialogDescription className="text-xs">
                Are you sure you want to delete "{deleteDialog.model?.name}"? This action
                cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="gap-2 sm:gap-0">
              <motion.div {...buttonHover} {...buttonTap}>
                <Button
                  variant="outline"
                  onClick={() => setDeleteDialog({ open: false, model: null })}
                >
                  Cancel
                </Button>
              </motion.div>
              <motion.div {...buttonHover} {...buttonTap}>
                <Button variant="destructive" onClick={handleDeleteModel}>
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete
                </Button>
              </motion.div>
            </DialogFooter>
          </motion.div>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default MyModelsPage;
