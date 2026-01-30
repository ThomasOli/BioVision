import React, { useEffect, useMemo, useState } from "react";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Tooltip from "@mui/material/Tooltip";
import Divider from "@mui/material/Divider";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import { useSelector } from "react-redux";
import UploadImages from "./UploadImages";
import type { RootState } from "../state/store";
import Landmark from "./Landmark";
import { AnnotatedImage } from "../types/Image";
import { TrainModelDialog } from "./PopUp";

const scrollbarStyles = `
  .custom-scrollbar::-webkit-scrollbar { width: 8px; }
  .custom-scrollbar::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
  .custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
`;

interface MenuProps {
  onOpacityChange: (selectedOpacity: number) => void;
  onColorChange: (selectedColor: string) => void;
  onSwitchChange: () => void;
}

async function saveLabels(fileArray: AnnotatedImage[]) {
  await window.api.saveLabels(fileArray);
}

const cardSx = {
  width: "100%",
  p: "12px",
  boxSizing: "border-box" as const,
  backgroundColor: "#fbfbfb", // updated theme color
  border: "1px solid #e5e7eb",
  borderRadius: "10px",
  boxShadow: "0 1px 0 rgba(255,255,255,0.6)",
  display: "flex",
  flexDirection: "column" as const,
  gap: "10px",
};

const labelSx = {
  fontWeight: 700,
  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  fontSize: "12px",
  color: "#475569",
  letterSpacing: "0.02em",
  textTransform: "uppercase" as const,
};

const Menu: React.FC<MenuProps> = ({ onColorChange, onOpacityChange, onSwitchChange }) => {
  const [openTrainDialog, setOpenTrainDialog] = useState(false);
  const [modelName, setModelName] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [modelPath, setModelPath] = useState("");

  // UI feedback
  const [snackOpen, setSnackOpen] = useState(false);
  const [snackMsg, setSnackMsg] = useState<string>("");
  const [snackSeverity, setSnackSeverity] = useState<"success" | "error" | "info">("info");

  const fileArray = useSelector((state: RootState) => state.files.fileArray);
  const canTrain = useMemo(() => (fileArray?.length ?? 0) > 0 && !isTraining, [fileArray, isTraining]);

  const showSnack = (message: string, severity: "success" | "error" | "info" = "info") => {
    setSnackMsg(message);
    setSnackSeverity(severity);
    setSnackOpen(true);
  };

  useEffect(() => {
    const fetchProjectRoot = async () => {
      try {
        const result = await window.api.getProjectRoot();
        if (result?.projectRoot) setModelPath(result.projectRoot);
      } catch (err) {
        console.error("Failed to load model path", err);
        showSnack("Failed to load model location.", "error");
      }
    };
    fetchProjectRoot();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSelectModelPath = async () => {
    try {
      const result = await window.api.selectProjectRoot();
      if (!result?.canceled && result?.projectRoot) {
        setModelPath(result.projectRoot);
        showSnack("Model location updated.", "success");
      }
    } catch (err) {
      console.error("Failed to select model path", err);
      showSnack("Failed to select model location.", "error");
    }
  };

  const handleCopyPath = async () => {
    if (!modelPath) return;
    try {
      await navigator.clipboard.writeText(modelPath);
      showSnack("Path copied.", "success");
    } catch (err) {
      console.error("Clipboard copy failed", err);
      showSnack("Could not copy path.", "error");
    }
  };

  const handleOpenFolder = async () => {
    if (!modelPath) return;
    try {
      // Optional: if you have an IPC method for this, use it.
      // @ts-expect-error - API may not exist in your preload yet
      if (window.api?.openPath) await window.api.openPath(modelPath);
      else showSnack("Open folder is not implemented yet.", "info");
    } catch (err) {
      console.error("Failed to open folder", err);
      showSnack("Could not open folder.", "error");
    }
  };

  const handleTrainConfirm = async () => {
    const name = modelName.trim();
    if (!name) return;

    try {
      setIsTraining(true);
      showSnack("Saving labels…", "info");

      await saveLabels(fileArray);

      showSnack("Training model…", "info");
      const result = await window.api.trainModel(name);

      if (!result.ok) throw new Error(result.error);

      console.log("Training output:", result.output);

      setOpenTrainDialog(false);
      setModelName("");
      showSnack("Training complete.", "success");
    } catch (err) {
      console.error(err);
      showSnack(`Training failed. ${String(err)}`, "error");
    } finally {
      setIsTraining(false);
    }
  };

  // Keyboard shortcut: Ctrl+N opens upload dialog
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
    return () => window.removeEventListener("open-upload-dialog", openUploadDialog);
  }, []);

  return (
    <>
      <style>{scrollbarStyles}</style>

      <Paper
        elevation={9}
        sx={{
          height: "100vh",
          display: "flex",
          flexDirection: "column",

          // allow parent container to control width
          width: "100%",
          maxWidth: "none",
          minWidth: 0,

          boxSizing: "border-box",
          overflow: "hidden",
          background: "#fbfbfb", // updated theme color
          borderRadius: 0,
        }}
      >
        <TrainModelDialog
          handleTrainConfirm={handleTrainConfirm}
          open={openTrainDialog}
          setOpen={setOpenTrainDialog}
          modelName={modelName}
          isTraining={isTraining}
          setModelName={setModelName}
        />

        {/* Scrollable content */}
        <Box
          className="custom-scrollbar"
          sx={{
            flex: 1,
            minWidth: 0,
            overflowY: "auto",
            overflowX: "hidden",
            p: 2,
            display: "flex",
            flexDirection: "column",
            gap: 1.5,
          }}
        >
          {/* Header */}
          <Box sx={{ mb: 0.5, textAlign: "center" }}>
            <Typography
              variant="h6"
              sx={{
                m: 0,
                textAlign: "center",
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                fontSize: "20px",
                fontWeight: 700,
                color: "#0f172a",
                lineHeight: 1.2,
              }}
            >
              Auto Landmarking
            </Typography>

            <Typography
              variant="body2"
              sx={{
                mt: 0.5,
                textAlign: "center",
                color: "#64748b",
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                fontSize: "12px",
              }}
            >
              Select Model • Import • Annotate
            </Typography>
          </Box>

          {/* Model */}
          <Box sx={cardSx}>
            <Typography sx={labelSx}>Model</Typography>

            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
              <Box sx={{ minWidth: 0 }}>
                <Typography
                  sx={{
                    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    fontWeight: 600,
                    fontSize: "13px",
                    color: "#1f2937",
                    mb: 0.25,
                  }}
                >
                  Model location
                </Typography>

                <Tooltip title={modelPath || "Loading..."} placement="bottom-start">
                  <Typography
                    sx={{
                      fontFamily: '"Fira Code", "SFMono-Regular", Consolas, monospace',
                      color: "#334155",
                      fontSize: "11.5px",
                      maxWidth: "100%",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {modelPath || "Loading..."}
                  </Typography>
                </Tooltip>
              </Box>

              <Box sx={{ display: "flex", gap: 1, flexShrink: 0 }}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleSelectModelPath}
                  sx={{
                    textTransform: "none",
                    borderRadius: "10px",
                    px: 1.25,
                    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    fontSize: "12px",
                    fontWeight: 600,
                    borderColor: "#d1d5db",
                    color: "#111827",
                    "&:hover": { borderColor: "#9ca3af", backgroundColor: "#f9fafb" },
                  }}
                >
                  Browse…
                </Button>
              </Box>
            </Box>

            <Box sx={{ display: "flex", gap: 1 }}>
              <Button
                size="small"
                variant="text"
                onClick={handleCopyPath}
                disabled={!modelPath}
                sx={{
                  textTransform: "none",
                  borderRadius: "10px",
                  px: 1.25,
                  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                  fontSize: "12px",
                  fontWeight: 600,
                  color: "#2563eb",
                  "&:hover": { backgroundColor: "rgba(37, 99, 235, 0.08)" },
                }}
              >
                Copy path
              </Button>

              <Button
                size="small"
                variant="text"
                onClick={handleOpenFolder}
                disabled={!modelPath}
                sx={{
                  textTransform: "none",
                  borderRadius: "10px",
                  px: 1.25,
                  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                  fontSize: "12px",
                  fontWeight: 600,
                  color: "#2563eb",
                  "&:hover": { backgroundColor: "rgba(37, 99, 235, 0.08)" },
                }}
              >
                Open folder
              </Button>
            </Box>
          </Box>

          {/* Input */}
          <Box sx={{ ...cardSx, gap: "12px" }}>
            <Box sx={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
              <Typography sx={labelSx}>Image Upload</Typography>
            </Box>

            <UploadImages />

            <Divider sx={{ borderColor: "#e5e7eb" }} />

            <Typography
              sx={{
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                fontSize: "12px",
                color: "#64748b",
              }}
            >
              {fileArray?.length ? `${fileArray.length} image(s) loaded` : "No images loaded"}
            </Typography>
          </Box>

          {/* Landmark controls */}
          <Landmark onOpacityChange={onOpacityChange} onColorChange={onColorChange} onSwitchChange={onSwitchChange} />

          {/* Training */}
          <Box sx={{ ...cardSx, gap: "10px" }}>
            <Typography sx={labelSx}>Training</Typography>

            <Typography
              sx={{
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                fontSize: "12px",
                color: canTrain ? "#64748b" : "#b91c1c",
              }}
            >
              {canTrain ? "Ready to train." : "Add images to enable training."}
            </Typography>
          </Box>

          {/* Spacer */}
          <Box sx={{ height: 86 }} />
        </Box>

        {/* Sticky footer */}
        <Box
          sx={{
            borderTop: "1px solid #e5e7eb",
            backgroundColor: "#fbfbfb",
            p: 2,
            display: "flex",
            justifyContent: "center",
          }}
        >
          <Button
            variant="contained"
            disabled={!canTrain}
            onClick={() => setOpenTrainDialog(true)}
            sx={{
              backgroundColor: "#3b82f6",
              color: "white",
              fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
              fontWeight: 700,
              fontSize: "14px",
              padding: "10px 18px",
              borderRadius: "10px",
              textTransform: "none",
              width: "100%",
              maxWidth: "280px",
              transition: "all 0.2s ease",
              "&:hover": {
                backgroundColor: "#2563eb",
                transform: "translateY(-1px)",
                boxShadow: "0 10px 22px rgba(59, 130, 246, 0.28)",
              },
              "&:disabled": {
                backgroundColor: "#93c5fd",
                color: "#ffffff",
              },
            }}
            startIcon={isTraining ? <CircularProgress size={16} sx={{ color: "white" }} /> : undefined}
          >
            {isTraining ? "Training…" : "Train model"}
          </Button>
        </Box>
      </Paper>

      {/* Snackbar */}
      <Snackbar
        open={snackOpen}
        autoHideDuration={3500}
        onClose={() => setSnackOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert onClose={() => setSnackOpen(false)} severity={snackSeverity} variant="filled">
          {snackMsg}
        </Alert>
      </Snackbar>
    </>
  );
};

export default Menu;
