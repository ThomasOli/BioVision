import React, { useEffect, useMemo, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Typography,
  Box,
  Stack,
  Divider,
  LinearProgress,
  InputAdornment,
  IconButton,
  Fade,
} from "@mui/material";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

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

  // Reset touched when opening/closing
  useEffect(() => {
    if (!open) setTouched(false);
  }, [open]);

  const trimmed = useMemo(() => modelName.trim(), [modelName]);

  // Basic "safe name" check (adjust if you want different rules)
  const nameOk = useMemo(() => /^[a-zA-Z0-9._-]+$/.test(trimmed), [trimmed]);
  const canTrain = trimmed.length > 0 && nameOk && !isTraining;

  const helperText = useMemo(() => {
    if (!touched) return "Use letters, numbers, hyphen (-), underscore (_), dot (.), or colon (:).";
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

  // Enter to submit (when not training)
  useEffect(() => {
    if (!open) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleClose();
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        // Ctrl+Enter / Cmd+Enter to train
        e.preventDefault();
        onTrain();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, canTrain, isTraining, trimmed, nameOk]);

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="sm"
      fullWidth
      TransitionComponent={Fade}
      transitionDuration={150}
      PaperProps={{
        sx: {
          borderRadius: "16px",
          border: "1px solid #e5e7eb",
          overflow: "hidden",
        },
      }}
    >
      {/* Header */}
      <Box sx={{ px: 2.25, py: 1.5, bgcolor: "#fbfbfb" }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
          <Box sx={{ minWidth: 0 }}>
            <Typography sx={{ fontWeight: 900, fontSize: 14, color: "#0f172a" }}>Train new model</Typography>
            <Typography sx={{ fontSize: 12, color: "#64748b" }}>
              Give your model a clear, versioned name (Ctrl/Cmd+Enter to start).
            </Typography>
          </Box>

          <IconButton
            onClick={handleClose}
            disabled={isTraining}
            aria-label="Close"
            sx={{
              bgcolor: "rgba(255,255,255,0.9)",
              border: "1px solid #e5e7eb",
              "&:hover": { bgcolor: "#fff" },
            }}
          >
            <CloseRoundedIcon />
          </IconButton>
        </Stack>
      </Box>

      <Divider />

      {/* Body */}
      <DialogContent sx={{ pt: 2.25 }}>
        <TextField
          autoFocus
          label="Model name"
          fullWidth
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          onBlur={() => setTouched(true)}
          placeholder="e.g. fossil_landmarks_v1"
          disabled={isTraining}
          error={touched && (!trimmed || !nameOk)}
          helperText={helperText}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <InfoOutlinedIcon fontSize="small" />
              </InputAdornment>
            ),
          }}
        />

        {isTraining && (
          <Box sx={{ mt: 2 }}>
            <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#0f172a", mb: 0.75 }}>
              Training in progress…
            </Typography>
            <LinearProgress />
          </Box>
        )}
      </DialogContent>

      {/* Footer */}
      <DialogActions sx={{ px: 2.25, pb: 2.25, pt: 0.5 }}>
        <Button
          onClick={handleClose}
          disabled={isTraining}
          variant="outlined"
          sx={{ textTransform: "none", borderRadius: "10px", fontWeight: 800 }}
        >
          Cancel
        </Button>

        <Button
          variant="contained"
          disabled={!canTrain}
          onClick={onTrain}
          sx={{
            textTransform: "none",
            borderRadius: "10px",
            fontWeight: 900,
            boxShadow: "none",
            "&:hover": { boxShadow: "none" },
          }}
        >
          {isTraining ? "Training…" : "Train model"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};
