import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
} from "@mui/material";

interface TrainModelDialogProps {
  open: boolean;
  setOpen: (value: boolean) => void;
  handleTrainConfirm : () => Promise<void>;
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


  // Helper to close only if not training
  const handleClose = () => {
    if (!isTraining) {
      setModelName("")
      setOpen(false)
    }
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>Train New Model</DialogTitle>

      <DialogContent>
        <TextField
          autoFocus
          margin="dense"
          label="Model name"
          fullWidth
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          placeholder="e.g. fossil_landmarks_v1"
          disabled={isTraining}
        />
      </DialogContent>

      <DialogActions>
        <Button 
          onClick={handleClose} 
          disabled={isTraining}
        >
          Cancel
        </Button>

        <Button
          variant="contained"
          disabled={!modelName.trim() || isTraining}
          onClick={handleTrainConfirm}
        >
          {isTraining ? "Training..." : "Train"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};