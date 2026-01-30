// src/Components/ImageLabelerCarousel.tsx
import React, { useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useDispatch } from "react-redux";
import {
  Button,
  IconButton,
  Box,
  Typography,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from "@mui/material";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";

import ImageLabeler from "./ImageLabeler";
import MagnifiedImageLabeler from "./MagnifiedZoomLabeler";
import { UndoRedoClearContext } from "./UndoRedoClearContext";

import { AppDispatch } from "../state/store";
import { removeFile, updateLabels } from "../state/filesState/fileSlice";

import ChevronLeftRoundedIcon from "@mui/icons-material/ChevronLeftRounded";
import ChevronRightRoundedIcon from "@mui/icons-material/ChevronRightRounded";
import DeleteOutlineRoundedIcon from "@mui/icons-material/DeleteOutlineRounded";
import ZoomInRoundedIcon from "@mui/icons-material/ZoomInRounded";

interface ImageLabelerCarouselProps {
  color: string;
  opacity: number;
  isSwitchOn: boolean;
}

const ImageLabelerCarousel: React.FC<ImageLabelerCarouselProps> = ({ color, opacity, isSwitchOn }) => {
  const { images, setSelectedImage } = useContext(UndoRedoClearContext);
  const dispatch = useDispatch<AppDispatch>();

  const totalImages = images.length;

  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isMagnified, setIsMagnified] = useState<boolean>(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  useEffect(() => {
    if (totalImages === 0) {
      setCurrentIndex(0);
      return;
    }
    setCurrentIndex((prev) => Math.min(prev, totalImages - 1));
  }, [totalImages]);

  const current = useMemo(() => (totalImages ? images[currentIndex] : null), [images, currentIndex, totalImages]);
  const hasLandmarks = Boolean(current?.labels?.length);

  const handleUpdateLabels = useCallback(
    (id: number, labels: { x: number; y: number; id: number }[]) => {
      dispatch(updateLabels({ id, labels }));
    },
    [dispatch]
  );

  const handleDeleteImage = useCallback(
    (id: number) => {
      dispatch(removeFile(id));
      setCurrentIndex((prevIndex) => {
        const newTotal = totalImages - 1;
        if (newTotal <= 0) return 0;
        if (prevIndex >= newTotal) return newTotal - 1;
        return prevIndex;
      });
    },
    [dispatch, totalImages]
  );

  const handleNext = useCallback(() => {
    if (totalImages <= 1) return;
    setCurrentIndex((prev) => (prev + 1) % totalImages);
    setSelectedImage((prev) => (prev + 1) % totalImages);
  }, [totalImages, setSelectedImage]);

  const handlePrev = useCallback(() => {
    if (totalImages <= 1) return;
    setCurrentIndex((prev) => (prev - 1 + totalImages) % totalImages);
    setSelectedImage((prev) => (prev - 1 + totalImages) % totalImages);
  }, [totalImages, setSelectedImage]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") handleNext();
      else if (e.key === "ArrowLeft") handlePrev();
    },
    [handleNext, handlePrev]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const toggleMagnifiedView = () => setIsMagnified((prev) => !prev);

  // Export only CURRENT image's labels (only shown if it has landmarks)
  const exportCurrent = useCallback(async () => {
    if (!current || !current.labels?.length) return;

    const dims = await new Promise<{ width: number; height: number } | null>((resolve) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => resolve(null);
      img.src = current.url;
    });

    const data = {
      imageURL: current.url,
      imageDimensions: dims,
      points: current.labels.map(({ x, y, id }: any) => ({ x: Math.round(x), y: Math.round(y), id })),
    };

    const jsonData = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonData], { type: "application/json" });
    const urlBlob = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = urlBlob;
    a.download = `labeled_data_${Date.now()}.json`;
    a.click();

    URL.revokeObjectURL(urlBlob);
  }, [current]);

  const exportAll = useCallback(() => {
    const data = images.map(({ id, url, labels }) => ({ id, url, labels }));
    const jsonData = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonData], { type: "application/json" });
    const urlBlob = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = urlBlob;
    a.download = `all_labeled_data_${Date.now()}.json`;
    a.click();

    URL.revokeObjectURL(urlBlob);
  }, [images]);

  if (totalImages === 0) {
    return (
      <Paper
        elevation={0}
        sx={{
          width: "min(900px, 100%)",
          p: 3,
          bgcolor: "#fbfbfb",
          border: "1px solid #e5e7eb",
          borderRadius: "14px",
          textAlign: "center",
        }}
      >
        <Typography sx={{ fontWeight: 800, fontSize: 18, color: "#0f172a", mb: 1 }}>No images available.</Typography>
        <Typography sx={{ fontSize: 13, color: "#64748b", maxWidth: 520, mx: "auto" }}>
          Press <strong>Ctrl+N</strong> to upload images, or use the left sidebar to begin labeling.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper
      elevation={0}
      sx={{
        width: "100%",
        height: "100%",
        minWidth: 0,
        minHeight: 0,
        bgcolor: "#fbfbfb",
        border: "1px solid #e5e7eb",
        borderRadius: "14px",
        p: 2,
        display: "flex",
        flexDirection: "column",
        gap: 1.5,
        boxSizing: "border-box",
      }}
    >
      {/* Toolbar */}
      <Box
        sx={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 1,
          minWidth: 0,
        }}
      >
        <Box sx={{ minWidth: 0 }}>
          <Typography sx={{ fontWeight: 800, fontSize: 13, color: "#0f172a" }}>
            Image {currentIndex + 1} / {totalImages}
          </Typography>
          <Typography sx={{ fontSize: 12, color: "#64748b" }}>Use ← / → to navigate • Ctrl+N to add</Typography>
        </Box>

        <Stack direction="row" spacing={1} alignItems="center" sx={{ flexShrink: 0 }}>
          {/* Export current (only if current has landmarks) */}
          {hasLandmarks && (
            <Button
              variant="outlined"
              size="small"
              onClick={exportCurrent}
              sx={{
                textTransform: "none",
                borderRadius: "10px",
                fontWeight: 700,
                whiteSpace: "nowrap",
              }}
            >
              Export Current (JSON)
            </Button>
          )}

          {/* Export all */}
          <Button
            variant="outlined"
            size="small"
            onClick={exportAll}
            sx={{
              textTransform: "none",
              borderRadius: "10px",
              fontWeight: 700,
              whiteSpace: "nowrap",
            }}
          >
            Export All (JSON)
          </Button>

          <Tooltip title="Magnify">
            <IconButton
              onClick={toggleMagnifiedView}
              aria-label="Magnify"
              sx={{
                bgcolor: "rgba(255,255,255,0.9)",
                border: "1px solid #e5e7eb",
                "&:hover": { bgcolor: "#fff" },
                transition: "all 0.15s ease",
              }}
            >
              <ZoomInRoundedIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Delete">
            <IconButton
              onClick={() => setConfirmOpen(true)}
              aria-label="Delete"
              sx={{
                bgcolor: "rgba(255,255,255,0.9)",
                border: "1px solid #e5e7eb",
                color: "#dc2626",
                "&:hover": { bgcolor: "#fff" },
                transition: "all 0.15s ease",
              }}
            >
              <DeleteOutlineRoundedIcon />
            </IconButton>
          </Tooltip>
        </Stack>
      </Box>

      {/* Image area */}
      <Box
        sx={{
          position: "relative",
          width: "100%",
          flex: 1,
          minHeight: 0,
          minWidth: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "#fff",
          border: "1px solid #e5e7eb",
          borderRadius: "14px",
          overflow: "hidden",
        }}
      >
        <IconButton
          onClick={handlePrev}
          disabled={totalImages === 1}
          aria-label="Previous"
          sx={{
            position: "absolute",
            left: 10,
            top: "50%",
            transform: "translateY(-50%)",
            bgcolor: "rgba(255,255,255,0.9)",
            border: "1px solid #e5e7eb",
            "&:hover": { bgcolor: "#fff" },
            transition: "all 0.15s ease",
          }}
        >
          <ChevronLeftRoundedIcon />
        </IconButton>

        {current && (
          <Box sx={{ width: "100%", height: "100%", minWidth: 0, minHeight: 0 }}>
            <ImageLabeler
              key={current.id}
              imageURL={current.url}
              onPointsChange={(newPoints) => handleUpdateLabels(current.id, newPoints)}
              color={color}
              opacity={opacity}
              mode={isSwitchOn}
            />
          </Box>
        )}

        <IconButton
          onClick={handleNext}
          disabled={totalImages === 1}
          aria-label="Next"
          sx={{
            position: "absolute",
            right: 10,
            top: "50%",
            transform: "translateY(-50%)",
            bgcolor: "rgba(255,255,255,0.9)",
            border: "1px solid #e5e7eb",
            "&:hover": { bgcolor: "#fff" },
            transition: "all 0.15s ease",
          }}
        >
          <ChevronRightRoundedIcon />
        </IconButton>
      </Box>

      {/* Delete confirmation dialog */}
      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)} aria-labelledby="confirm-delete-title">
        <DialogTitle id="confirm-delete-title" sx={{ fontWeight: 800 }}>
          Delete this image?
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ color: "#64748b" }}>
            This will remove the current image and its labels from the session. This cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 2, pb: 2 }}>
          <Button
            onClick={() => setConfirmOpen(false)}
            variant="outlined"
            sx={{ textTransform: "none", borderRadius: "10px", fontWeight: 700 }}
          >
            Cancel
          </Button>
          <Button
            onClick={() => {
              if (current) handleDeleteImage(current.id);
              setConfirmOpen(false);
            }}
            variant="contained"
            sx={{
              textTransform: "none",
              borderRadius: "10px",
              fontWeight: 800,
              bgcolor: "#dc2626",
              "&:hover": { bgcolor: "#b91c1c" },
            }}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {current && (
        <MagnifiedImageLabeler
          imageURL={current.url}
          onPointsChange={(newPoints) => handleUpdateLabels(current.id, newPoints)}
          color={color}
          opacity={opacity}
          open={isMagnified}
          onClose={toggleMagnifiedView}
          mode={isSwitchOn}
        />
      )}
    </Paper>
  );
};

export default ImageLabelerCarousel;
