// src/Components/UploadImages.tsx
import React, { ChangeEvent, useState, DragEvent } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import LinearProgress from "@mui/material/LinearProgress";
import IconButton from "@mui/material/IconButton";
import Alert from "@mui/material/Alert";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import FolderIcon from "@mui/icons-material/Folder";
import DeleteIcon from "@mui/icons-material/Delete";
import ClearAllIcon from "@mui/icons-material/ClearAll";
import ImageIcon from "@mui/icons-material/Image";
import { useSelector, useDispatch } from "react-redux";
import { addFiles } from "../state/filesState/fileSlice";
import { RootState } from "../state/store";

interface UploadImagesProps {}

const fontFamily = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

// More robust than name-only (avoids collisions)
const fileKey = (f: File) => `${f.name}__${f.size}__${f.lastModified}`;

const UploadImages: React.FC<UploadImagesProps> = () => {
  const files = useSelector((state: RootState) => state.files.fileArray);
  const dispatch = useDispatch();

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [isError, setIsError] = useState(false);
  const [disableClear, setDisableClear] = useState(true);

  // UI: drag state for dropzone highlight
  const [isDragging, setIsDragging] = useState(false);

  // Merge helper (keeps existing queue, appends unique new items, preserves existing previews)
  const appendToQueue = (incomingFiles: File[], incomingPreviews: string[]) => {
    const existingKeys = new Set(selectedFiles.map(fileKey));

    const keptFiles: File[] = [...selectedFiles];
    const keptPreviews: string[] = [...previews];

    for (let i = 0; i < incomingFiles.length; i++) {
      const f = incomingFiles[i];
      const p = incomingPreviews[i];

      const k = fileKey(f);
      if (existingKeys.has(k)) {
        // Avoid leaking blob URLs we just created for duplicates
        if (p?.startsWith("blob:")) URL.revokeObjectURL(p);
        continue;
      }
      existingKeys.add(k);
      keptFiles.push(f);
      keptPreviews.push(p);
    }

    setSelectedFiles(keptFiles);
    setPreviews(keptPreviews);
    setDisableClear(keptFiles.length === 0);
    setProgress(0);
    setMessage("");
    setIsError(false);
  };

  const handleSelectFolder = async () => {
    const result = await window.api.selectImageFolder();
    if (result.canceled) return;

    // NOTE: result.image = File[], result.images = { path: string }[]
    const incomingFiles: File[] = result.image ?? [];
    const incomingPreviews: string[] = (result.images ?? []).map((img: any) => `file://${img.path}`);

    appendToQueue(incomingFiles, incomingPreviews);

    // Reset the input so selecting the same files again still fires onChange
    const fileInput = document.getElementById("btn-upload") as HTMLInputElement;
    if (fileInput) fileInput.value = "";
  };

  const handleSelectFiles = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const incomingFiles = Array.from(event.target.files);
      const incomingPreviews = incomingFiles.map((file) => URL.createObjectURL(file));

      appendToQueue(incomingFiles, incomingPreviews);

      // Reset input so selecting same file again still triggers
      event.target.value = "";
    }
  };

  const handleUpload = () => {
    if (selectedFiles.length === 0) {
      setMessage("No files to upload.");
      setIsError(true);
      return;
    }

    dispatch(addFiles(selectedFiles));
    console.log("Uploaded Files:", selectedFiles);
    console.log("Redux Store Files:", files);

    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += 50;
      setProgress(currentProgress);

      if (currentProgress >= 100) {
        clearInterval(interval);

        // revoke blob URLs
        previews.forEach((p) => {
          if (p.startsWith("blob:")) URL.revokeObjectURL(p);
        });

        setSelectedFiles([]);
        setPreviews([]);
        setMessage("Upload completed!");
        setIsError(false);
        setDisableClear(true);

        setTimeout(() => setMessage(""), 3000);
      }
    }, 500);
  };

  const handleRemoveImage = (index: number) => {
    const updatedFiles = [...selectedFiles];
    const updatedPreviews = [...previews];

    updatedFiles.splice(index, 1);
    const removedPreview = updatedPreviews.splice(index, 1)[0];

    if (removedPreview?.startsWith("blob:")) {
      URL.revokeObjectURL(removedPreview);
    }

    setSelectedFiles(updatedFiles);
    setPreviews(updatedPreviews);

    if (updatedFiles.length === 0) {
      setDisableClear(true);
      const fileInput = document.getElementById("btn-upload") as HTMLInputElement;
      if (fileInput) fileInput.value = "";
    }
  };

  const handleClearAll = () => {
    previews.forEach((preview) => {
      if (preview.startsWith("blob:")) URL.revokeObjectURL(preview);
    });

    setSelectedFiles([]);
    setPreviews([]);
    setDisableClear(true);
    setProgress(0);
    setMessage("");
    setIsError(false);

    const fileInput = document.getElementById("btn-upload") as HTMLInputElement;
    if (fileInput) fileInput.value = "";
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDragEnter = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleFolderDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);

    const items = event.dataTransfer.items;
    if (!items) return;

    const incomingFiles: File[] = [];
    let isFolder = false;

    for (let i = 0; i < items.length; i++) {
      const entry = items[i].webkitGetAsEntry();
      if (!entry) continue;

      if (entry.isFile) {
        const file = items[i].getAsFile();
        if (file) incomingFiles.push(file);
      } else if (entry.isDirectory) {
        isFolder = true;
        await processEntry(entry, incomingFiles);
      }
    }

    if (isFolder || incomingFiles.length > 0) {
      const incomingPreviews = incomingFiles.map((file) => URL.createObjectURL(file));
      appendToQueue(incomingFiles, incomingPreviews);
    }
  };

  const processEntry = async (entry: FileSystemEntry, filesArray: File[]) => {
    return new Promise<void>((resolve, reject) => {
      if (entry.isFile) {
        const fileEntry = entry as FileSystemFileEntry;
        fileEntry.file(
          (file: File) => {
            filesArray.push(file);
            resolve();
          },
          (err) => {
            console.error("Error reading file entry:", err);
            reject(err);
          }
        );
      } else if (entry.isDirectory) {
        const directoryReader = (entry as FileSystemDirectoryEntry).createReader();
        const readEntries = () => {
          directoryReader.readEntries(
            async (entries: FileSystemEntry[]) => {
              if (entries.length === 0) {
                resolve();
              } else {
                for (let i = 0; i < entries.length; i++) {
                  await processEntry(entries[i], filesArray);
                }
                readEntries();
              }
            },
            (err) => {
              console.error("Error reading directory entries:", err);
              reject(err);
            }
          );
        };
        readEntries();
      }
    });
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        fontFamily,
        gap: 1.5,
        width: "100%",
      }}
    >
      {/* Dropzone */}
      <Box
        onClick={() => document.getElementById("btn-upload")?.click()}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleFolderDrop}
        sx={{
          border: "2px dashed",
          borderColor: isDragging ? "#3b82f6" : "#cbd5e1",
          borderRadius: "12px",
          width: "90%",
          maxWidth: "320px",
          minHeight: "80px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: "6px",
          padding: "12px",
          backgroundColor: isDragging ? "#eff6ff" : "#f8fafc",
          transition: "all 0.18s ease",
          cursor: "pointer",
          userSelect: "none",
          "&:hover": {
            borderColor: "#3b82f6",
            backgroundColor: "#eff6ff",
          },
        }}
      >
        <ImageIcon sx={{ fontSize: 32, color: isDragging ? "#2563eb" : "#94a3b8" }} />
        <Typography
          variant="body1"
          sx={{
            color: isDragging ? "#1d4ed8" : "#64748b",
            fontWeight: 600,
            fontSize: "13px",
            textAlign: "center",
          }}
        >
          {isDragging ? "Drop to Add Images" : "Drag & Drop Images Here"}
        </Typography>
        <Typography sx={{ color: "#94a3b8", fontSize: "12px" }}>or click to browse</Typography>
      </Box>

      {/* Selected summary + Clear */}
      {selectedFiles.length > 0 && (
        <Box
          sx={{
            width: "100%",
            maxWidth: "320px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            mt: 0.5,
          }}
        >
          <Typography sx={{ fontFamily, fontSize: "12px", color: "#64748b", fontWeight: 600 }}>
            Selected: {selectedFiles.length}
          </Typography>

          <Button
            variant="text"
            size="small"
            onClick={handleClearAll}
            disabled={disableClear}
            startIcon={<ClearAllIcon />}
            sx={{
              textTransform: "none",
              borderRadius: "10px",
              fontFamily,
              fontSize: "12px",
              fontWeight: 700,
              color: "#ef4444",
              "&:hover": { backgroundColor: "#fef2f2" },
              "&:disabled": { color: "#cbd5e1" },
            }}
          >
            Clear
          </Button>
        </Box>
      )}

      {/* Preview list */}
      {previews.length > 0 && (
        <Box
          sx={{
            maxHeight: "210px",
            overflowY: "auto",
            width: "100%",
            maxWidth: "320px",
            display: "flex",
            flexDirection: "column",
            gap: "10px",
            p: 1,
            backgroundColor: "#f8fafc",
            borderRadius: "12px",
            border: "1px solid #e5e7eb",
            "&::-webkit-scrollbar": { width: "8px" },
            "&::-webkit-scrollbar-track": { backgroundColor: "#f1f5f9", borderRadius: "4px" },
            "&::-webkit-scrollbar-thumb": {
              backgroundColor: "#cbd5e1",
              borderRadius: "4px",
              "&:hover": { backgroundColor: "#94a3b8" },
            },
          }}
        >
          {previews.map((preview, index) => (
            <Box
              key={`${preview}-${index}`}
              sx={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                p: 1,
                backgroundColor: "#ffffff",
                borderRadius: "12px",
                border: "1px solid #e5e7eb",
                transition: "box-shadow 0.15s ease, border-color 0.15s ease",
                "&:hover": {
                  boxShadow: "0 4px 12px rgba(15, 23, 42, 0.08)",
                  borderColor: "#d1d5db",
                },
              }}
            >
              <img
                src={preview}
                alt={`Preview ${index}`}
                style={{
                  width: "64px",
                  height: "48px",
                  objectFit: "cover",
                  borderRadius: "10px",
                  border: "1px solid #e5e7eb",
                }}
              />

              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography
                  sx={{
                    fontFamily,
                    fontSize: "12.5px",
                    fontWeight: 700,
                    color: "#0f172a",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {selectedFiles[index]?.name || `Image ${index + 1}`}
                </Typography>
                <Typography sx={{ fontFamily, fontSize: "11.5px", color: "#64748b" }}>
                  Ready to upload
                </Typography>
              </Box>

              <IconButton
                onClick={() => handleRemoveImage(index)}
                size="small"
                aria-label="Remove image"
                sx={{
                  color: "#ef4444",
                  borderRadius: "10px",
                  "&:hover": { backgroundColor: "#fef2f2" },
                }}
              >
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Box>
          ))}
        </Box>
      )}

      {/* File + Folder buttons */}
      <Box sx={{ display: "flex", gap: 1, width: "100%", maxWidth: "320px" }}>
        <label htmlFor="btn-upload" style={{ flex: 1 }}>
          <input
            id="btn-upload"
            name="btn-upload"
            style={{ display: "none" }}
            type="file"
            accept="image/*"
            multiple
            onChange={handleSelectFiles}
          />
          <Button
            variant="outlined"
            component="span"
            fullWidth
            startIcon={<ImageIcon />}
            sx={{
              borderColor: "#d1d5db",
              color: "#0f172a",
              fontWeight: 700,
              fontSize: "13px",
              borderRadius: "12px",
              textTransform: "none",
              minHeight: 36,
              "&:hover": {
                borderColor: "#3b82f6",
                backgroundColor: "#eff6ff",
              },
            }}
          >
            Images
          </Button>
        </label>

        <Button
          variant="outlined"
          onClick={handleSelectFolder}
          startIcon={<FolderIcon />}
          sx={{
            flex: 1,
            borderColor: "#d1d5db",
            color: "#0f172a",
            fontWeight: 700,
            fontSize: "13px",
            borderRadius: "12px",
            textTransform: "none",
            minHeight: 36,
            "&:hover": {
              borderColor: "#3b82f6",
              backgroundColor: "#eff6ff",
            },
          }}
        >
          Folder
        </Button>
      </Box>

      {/* Progress */}
      {progress > 0 && progress < 100 && (
        <Box sx={{ width: "100%", maxWidth: "320px" }}>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{
              height: "8px",
              borderRadius: "999px",
              backgroundColor: "#e2e8f0",
              "& .MuiLinearProgress-bar": {
                backgroundColor: "#3b82f6",
                borderRadius: "999px",
              },
            }}
          />
          <Typography
            variant="body2"
            sx={{
              mt: 0.75,
              textAlign: "center",
              color: "#64748b",
              fontSize: "12px",
              fontWeight: 600,
              fontFamily,
            }}
          >
            {`${progress}%`}
          </Typography>
        </Box>
      )}

      {/* Upload button */}
      {previews.length > 0 && (
        <Button
          variant="contained"
          disabled={selectedFiles.length === 0}
          onClick={handleUpload}
          startIcon={<CloudUploadIcon />}
          sx={{
            width: "100%",
            maxWidth: "320px",
            backgroundColor: "#3b82f6",
            color: "white",
            fontWeight: 800,
            fontSize: "14px",
            padding: "10px 16px",
            borderRadius: "12px",
            textTransform: "none",
            transition: "all 0.15s ease",
            "&:hover": {
              backgroundColor: "#2563eb",
              boxShadow: "0 10px 22px rgba(59, 130, 246, 0.25)",
            },
            "&:disabled": {
              backgroundColor: "#e2e8f0",
              color: "#cbd5e1",
            },
          }}
        >
          Upload files
        </Button>
      )}

      {/* Message */}
      {message && (
        <Box sx={{ width: "100%", maxWidth: "320px" }}>
          <Alert
            severity={isError ? "error" : "success"}
            sx={{
              borderRadius: "12px",
              fontFamily,
              "& .MuiAlert-message": { fontSize: "13px", fontWeight: 600 },
            }}
          >
            {message}
          </Alert>
        </Box>
      )}
    </Box>
  );
};

export default UploadImages;
