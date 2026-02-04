import React, { ChangeEvent, useState, DragEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Folder, Trash2, X, ImageIcon, Loader2 } from "lucide-react";
import { useDispatch } from "react-redux";
import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Progress } from "@/Components/ui/progress";
import { ScrollArea } from "@/Components/ui/scroll-area";
import { addFiles } from "../state/filesState/fileSlice";
import { staggerContainer, staggerItem, dropzoneActive, buttonHover, buttonTap } from "@/lib/animations";
import { toast } from "sonner";

const fileKey = (f: File) => `${f.name}__${f.size}__${f.lastModified}`;

const UploadImages: React.FC = () => {
  const dispatch = useDispatch();

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const appendToQueue = (incomingFiles: File[], incomingPreviews: string[]) => {
    const existingKeys = new Set(selectedFiles.map(fileKey));

    const keptFiles: File[] = [...selectedFiles];
    const keptPreviews: string[] = [...previews];

    for (let i = 0; i < incomingFiles.length; i++) {
      const f = incomingFiles[i];
      const p = incomingPreviews[i];

      const k = fileKey(f);
      if (existingKeys.has(k)) {
        if (p?.startsWith("blob:")) URL.revokeObjectURL(p);
        continue;
      }
      existingKeys.add(k);
      keptFiles.push(f);
      keptPreviews.push(p);
    }

    setSelectedFiles(keptFiles);
    setPreviews(keptPreviews);
    setProgress(0);
  };

  const handleSelectFolder = async () => {
    const result = await window.api.selectImageFolder();
    if (result.canceled || !result.images?.length) return;

    const incomingFiles: File[] = [];
    const incomingPreviews: string[] = [];

    // Convert base64 data to File objects
    for (const img of result.images) {
      try {
        // Decode base64 to binary
        const binaryString = atob(img.data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: img.mimeType });

        // Create File object with the original filename
        const file = new File([blob], img.filename, { type: img.mimeType });
        // Attach the original path for later use (e.g., training)
        Object.defineProperty(file, 'path', { value: img.path, writable: false });

        incomingFiles.push(file);
        incomingPreviews.push(URL.createObjectURL(blob));
      } catch (err) {
        console.error(`Failed to process image: ${img.filename}`, err);
      }
    }

    if (incomingFiles.length > 0) {
      appendToQueue(incomingFiles, incomingPreviews);
    } else {
      toast.error("No valid images found in the folder.");
    }

    const fileInput = document.getElementById("btn-upload") as HTMLInputElement;
    if (fileInput) fileInput.value = "";
  };

  const handleSelectFiles = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const incomingFiles = Array.from(event.target.files);
      const incomingPreviews = incomingFiles.map((file) =>
        URL.createObjectURL(file)
      );

      appendToQueue(incomingFiles, incomingPreviews);
      event.target.value = "";
    }
  };

  const handleUpload = () => {
    if (selectedFiles.length === 0) {
      toast.error("No files to upload.");
      return;
    }

    setIsUploading(true);
    dispatch(addFiles(selectedFiles));

    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += 50;
      setProgress(currentProgress);

      if (currentProgress >= 100) {
        clearInterval(interval);

        previews.forEach((p) => {
          if (p.startsWith("blob:")) URL.revokeObjectURL(p);
        });

        setSelectedFiles([]);
        setPreviews([]);
        setIsUploading(false);
        toast.success("Upload completed!");
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
    setProgress(0);

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

    for (let i = 0; i < items.length; i++) {
      const entry = items[i].webkitGetAsEntry();
      if (!entry) continue;

      if (entry.isFile) {
        const file = items[i].getAsFile();
        if (file) incomingFiles.push(file);
      } else if (entry.isDirectory) {
        await processEntry(entry, incomingFiles);
      }
    }

    if (incomingFiles.length > 0) {
      const incomingPreviews = incomingFiles.map((file) =>
        URL.createObjectURL(file)
      );
      appendToQueue(incomingFiles, incomingPreviews);
    }
  };

  const processEntry = async (
    entry: FileSystemEntry,
    filesArray: File[]
  ): Promise<void> => {
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
    <div className="flex w-full flex-col items-center gap-3">
      {/* Dropzone */}
      <motion.div
        variants={dropzoneActive}
        animate={isDragging ? "active" : "inactive"}
        onClick={() => document.getElementById("btn-upload")?.click()}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleFolderDrop}
        className={cn(
          "flex min-h-[80px] w-full max-w-[320px] cursor-pointer flex-col items-center justify-center gap-1.5 rounded-xl border-2 border-dashed p-3 transition-all",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-border bg-muted/30 hover:border-primary hover:bg-primary/5"
        )}
      >
        <ImageIcon
          className={cn(
            "h-8 w-8 transition-colors",
            isDragging ? "text-primary" : "text-muted-foreground"
          )}
        />
        <p
          className={cn(
            "text-center text-sm font-semibold transition-colors",
            isDragging ? "text-primary" : "text-muted-foreground"
          )}
        >
          {isDragging ? "Drop to Add Images" : "Drag & Drop Images Here"}
        </p>
        <p className="text-xs text-muted-foreground">or click to browse</p>
      </motion.div>

      {/* Selected summary + Clear */}
      <AnimatePresence>
        {selectedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex w-full max-w-[320px] items-center justify-between"
          >
            <span className="text-xs font-semibold text-muted-foreground">
              Selected: {selectedFiles.length}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearAll}
              className="h-7 text-xs font-bold text-destructive hover:bg-destructive/10 hover:text-destructive"
            >
              <X className="mr-1 h-3 w-3" />
              Clear
            </Button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Preview list */}
      <AnimatePresence>
        {previews.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="w-full max-w-[320px]"
          >
            <ScrollArea className="h-[210px] rounded-xl border bg-muted/30 p-2">
              <motion.div
                variants={staggerContainer}
                initial="hidden"
                animate="visible"
                className="flex flex-col gap-2"
              >
                {previews.map((preview, index) => (
                  <motion.div
                    key={`${preview}-${index}`}
                    variants={staggerItem}
                    layout
                    className="flex items-center gap-2.5 rounded-xl border bg-card p-2 transition-shadow hover:shadow-md"
                  >
                    <img
                      src={preview}
                      alt={`Preview ${index}`}
                      className="h-12 w-16 rounded-lg border object-cover"
                    />

                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-bold text-foreground">
                        {selectedFiles[index]?.name || `Image ${index + 1}`}
                      </p>
                      <p className="text-[11px] text-muted-foreground">
                        Ready to upload
                      </p>
                    </div>

                    <motion.div {...buttonHover} {...buttonTap}>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-destructive hover:bg-destructive/10 hover:text-destructive"
                        onClick={() => handleRemoveImage(index)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </motion.div>
                  </motion.div>
                ))}
              </motion.div>
            </ScrollArea>
          </motion.div>
        )}
      </AnimatePresence>

      {/* File + Folder buttons */}
      <div className="flex w-full max-w-[320px] gap-2">
        <label htmlFor="btn-upload" className="flex-1">
          <input
            id="btn-upload"
            name="btn-upload"
            className="hidden"
            type="file"
            accept="image/*"
            multiple
            onChange={handleSelectFiles}
          />
          <motion.div {...buttonHover} {...buttonTap} className="w-full">
            <Button
              variant="outline"
              className="w-full font-bold"
              asChild
            >
              <span>
                <ImageIcon className="mr-2 h-4 w-4" />
                Images
              </span>
            </Button>
          </motion.div>
        </label>

        <motion.div {...buttonHover} {...buttonTap} className="flex-1">
          <Button
            variant="outline"
            className="w-full font-bold"
            onClick={handleSelectFolder}
          >
            <Folder className="mr-2 h-4 w-4" />
            Folder
          </Button>
        </motion.div>
      </div>

      {/* Progress */}
      <AnimatePresence>
        {progress > 0 && progress < 100 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="w-full max-w-[320px] space-y-1"
          >
            <Progress value={progress} className="h-2" />
            <p className="text-center text-xs font-semibold text-muted-foreground">
              {progress}%
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Upload button */}
      <AnimatePresence>
        {previews.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="w-full max-w-[320px]"
          >
            <motion.div {...buttonHover} {...buttonTap}>
              <Button
                className="w-full font-bold"
                disabled={selectedFiles.length === 0 || isUploading}
                onClick={handleUpload}
              >
                {isUploading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Upload className="mr-2 h-4 w-4" />
                )}
                {isUploading ? "Uploading..." : "Upload files"}
              </Button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default UploadImages;
