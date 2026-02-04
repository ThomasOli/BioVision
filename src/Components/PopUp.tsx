import React, { useEffect, useMemo, useState } from "react";
import { Info, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Input } from "@/Components/ui/input";
import { Label } from "@/Components/ui/label";
import { Progress } from "@/Components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/Components/ui/dialog";

interface TrainModelDialogProps {
  open: boolean;
  setOpen: (value: boolean) => void;
  handleTrainConfirm: () => Promise<void>;
  setModelName: (name: string) => void;
  isTraining?: boolean;
  modelName: string;
}

export const TrainModelDialog: React.FC<TrainModelDialogProps> = ({
  open,
  setOpen,
  handleTrainConfirm,
  modelName,
  setModelName,
  isTraining = false,
}) => {
  const [touched, setTouched] = useState(false);

  useEffect(() => {
    if (!open) setTouched(false);
  }, [open]);

  const trimmed = useMemo(() => modelName.trim(), [modelName]);

  const nameOk = useMemo(() => /^[a-zA-Z0-9._-]+$/.test(trimmed), [trimmed]);
  const canTrain = trimmed.length > 0 && nameOk && !isTraining;

  const helperText = useMemo(() => {
    if (!touched)
      return "Use letters, numbers, hyphen (-), underscore (_), dot (.), or colon (:).";
    if (!trimmed) return "Model name is required.";
    if (!nameOk) return "Only letters, numbers, ., _, -, : are allowed (no spaces).";
    return "Looks good.";
  }, [touched, trimmed, nameOk]);

  const handleClose = () => {
    if (isTraining) return;
    setModelName("");
    setOpen(false);
  };

  const onTrain = async () => {
    if (!canTrain) return;
    await handleTrainConfirm();
  };

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
  }, [open, canTrain, isTraining, trimmed, nameOk]);

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

          {isTraining && (
            <div className="space-y-2">
              <p className="text-xs font-bold text-foreground">
                Training in progress...
              </p>
              <Progress className="h-2" />
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
