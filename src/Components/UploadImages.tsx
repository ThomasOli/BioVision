// src/Components/UploadImages.tsx
import React, { ChangeEvent, useState, DragEvent } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import LinearProgress from "@mui/material/LinearProgress";
import { ipcRenderer } from "electron";
import { useSelector, useDispatch } from "react-redux";
import {
  clearFiles,
  addFiles,
  removeFile,
} from "../state/filesState/fileSlice";
import { RootState } from "../state/store";

interface UploadImagesProps {}

const UploadImages: React.FC<UploadImagesProps> = () => {
  const files = useSelector((state: RootState) => state.files.fileArray);
  const dispatch = useDispatch();

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [isError, setIsError] = useState(false);
  const [disableClear, setDisableClear] = useState(true);

  const handleSelectFolder = () => {
    ipcRenderer
      .invoke("open-folder-dialog")
      .then((result) => {
        if (!result.canceled && result.filePaths.length > 0) {
          const folderPath = result.filePaths[0];

          const fs = (window as any).require("fs");

          fs.readdir(
            folderPath,
            { encoding: "utf8" },
            async (err: NodeJS.ErrnoException | null, files: string[]) => {
              if (err) {
                console.error(err);
                setIsError(true);
                setMessage("Failed to read the folder.");
                return;
              }

              const imageFiles = files.filter((file) =>
                /\.(jpg|jpeg|png|gif)$/i.test(file)
              );

              const filePreviews: string[] = [];
              const fileObjects: File[] = [];

              for (const fileName of imageFiles) {
                try {
                  const filePath = `${folderPath}/${fileName}`;
                  const fileData = fs.readFileSync(filePath); // Synchronously read file
                  const uint8Array = new Uint8Array(fileData); // Convert Buffer to Uint8Array
                  const file = new File([uint8Array], fileName, {
                    type: "image/jpeg",
                  }); // Create File object
                  fileObjects.push(file);
                  const objectUrl = URL.createObjectURL(file);
                  filePreviews.push(objectUrl);
                } catch (readErr) {
                  console.error(
                    `Failed to read or process file ${fileName}:`,
                    readErr
                  );
                  setIsError(true);
                  setMessage(`Failed to read or process file ${fileName}.`);
                }
              }

              if (fileObjects.length > 0) {
                setSelectedFiles(fileObjects);
                setPreviews(filePreviews);
                setProgress(0);
                setMessage("");
                setDisableClear(false);
              } else {
                setIsError(true);
                setMessage(
                  "No valid image files found in the selected folder."
                );
              }
            }
          );
        }
      })
      .catch((err) => {
        console.error(err);
        setIsError(true);
        setMessage("An error occurred while selecting the folder.");
      });
  };

  const handleSelectFiles = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const filesArray = Array.from(event.target.files);

      const filePreviews = filesArray.map((file) => URL.createObjectURL(file));

      setSelectedFiles(filesArray);
      setPreviews(filePreviews);
      setProgress(0);
      setMessage("");
      setDisableClear(false);
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

    // Simulate upload progress
    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += 10;
      setProgress(currentProgress);
      if (currentProgress >= 100) {
        clearInterval(interval);
        setSelectedFiles([]);
        setPreviews([]);
        setMessage("Upload completed!");
        setDisableClear(false);
      }
    }, 500);
  };

  const handleRemoveImage = (index: number) => {
    const updatedFiles = [...selectedFiles];
    const updatedPreviews = [...previews];

    const removedFile = updatedFiles.splice(index, 1)[0];
    const removedPreview = updatedPreviews.splice(index, 1)[0];

    // Revoke the object URL to free memory
    URL.revokeObjectURL(removedPreview);

    setSelectedFiles(updatedFiles);
    setPreviews(updatedPreviews);

    if (updatedFiles.length === 0) {
      setDisableClear(true);
    }
  };

  const handleClearAll = () => {
    // Revoke all object URLs
    previews.forEach((preview) => URL.revokeObjectURL(preview));
    setSelectedFiles([]);
    setPreviews([]);
    setDisableClear(true);
    dispatch(clearFiles());
    setMessage("");
    setIsError(false);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleFolderDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const items = event.dataTransfer.items;

    if (items) {
      const newFilesArray: File[] = [];
      let isFolder = false;

      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry(); // Now, this is allowed because we've extended the interface

        if (entry) {
          if (entry.isFile) {
            const file = items[i].getAsFile();
            if (file) {
              newFilesArray.push(file);
            }
          } else if (entry.isDirectory) {
            isFolder = true;
            await processEntry(entry, newFilesArray);
          }
        }
      }

      if (isFolder || newFilesArray.length > 0) {
        const filteredNewFiles = newFilesArray.filter(
          (file) =>
            !selectedFiles.some(
              (existingFile) => existingFile.name === file.name
            )
        );

        const mergedFiles = [...selectedFiles, ...filteredNewFiles];
        const filePreviews = mergedFiles.map((file) =>
          URL.createObjectURL(file)
        );

        setSelectedFiles(mergedFiles);
        setPreviews(filePreviews);
        setProgress(0);
        setMessage("");
        setDisableClear(false);
      }
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
        const directoryReader = (
          entry as FileSystemDirectoryEntry
        ).createReader();
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
    <div
      style={{ display: "flex", flexDirection: "column", alignItems: "center" }}
    >
      <Box
        border={2}
        borderRadius={5}
        borderColor="grey.300"
        width="250px"
        height="100px"
        display="flex"
        justifyContent="center"
        alignItems="center"
        mb={2}
        onDragOver={handleDragOver}
        onDrop={handleFolderDrop}
      >
        <Typography variant="body1" color="textSecondary">
          Drag & Drop Images Here
        </Typography>
      </Box>
      {previews.length > 0 && (
        <div
          style={{
            height: "200px",
            overflowY: "scroll",
            width: "310px",
            marginBottom: "20px",
          }}
        >
          {previews.map((preview, index) => (
            <div
              key={index}
              style={{
                display: "flex",
                alignItems: "center",
                margin: "10px 0",
              }}
            >
              <img
                src={preview}
                alt={`Preview ${index}`}
                style={{ width: "200px", height: "100px" }}
              />
              <Button
                onClick={() => handleRemoveImage(index)}
                variant="outlined"
                sx={{
                  marginLeft: "20px",
                  marginRight: "20px",
                  backgroundColor: "transparent",
                  transition: "background-color 0.2s ease",
                  "&:hover": {
                    backgroundColor: "#D1EEF8",
                  },
                  fontSize: "10px",
                }}
              >
                Remove
              </Button>
            </div>
          ))}
        </div>
      )}
      <label htmlFor="btn-upload" style={{ display: "flex", gap: "10px" }}>
        <input
          id="btn-upload"
          name="btn-upload"
          style={{ display: "none" }}
          type="file"
          accept="image/*"
          multiple
          onChange={handleSelectFiles}
        />
        <Button className="btn-choose" variant="outlined" component="span">
          Choose Images
        </Button>
        <Button
          className="btn-choose-folder"
          variant="outlined"
          component="span"
          onClick={handleSelectFolder}
        >
          Choose Folder
        </Button>
      </label>
      {previews.length > 0 && (
        <Button
          className="btn-clear"
          color="secondary"
          variant="outlined"
          component="span"
          onClick={handleClearAll}
          style={{ marginTop: "10px", marginBottom: "10px" }}
          disabled={disableClear}
        >
          Clear All
        </Button>
      )}
      {progress > 0 && progress < 100 && (
        <Box className="my20" width="100%">
          <LinearProgress variant="determinate" value={progress} />
          <Typography
            variant="body2"
            color="textSecondary"
            align="center"
          >{`${progress}%`}</Typography>
        </Box>
      )}
      <Button
        className="btn-upload"
        color="primary"
        variant="contained"
        component="span"
        disabled={selectedFiles.length === 0}
        onClick={handleUpload}
      >
        Upload
      </Button>
      {message && (
        <Typography
          variant="subtitle2"
          className={`upload-message ${isError ? "error" : ""}`}
        >
          {message}
        </Typography>
      )}
    </div>
  );
};

export default UploadImages;
